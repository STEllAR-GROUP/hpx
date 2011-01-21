//  Copyright (c) 2007-2010 Chirag Dekate
// 
//  Distributed under the GPL. 

#include <cstring>
#include <iostream>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <sys/time.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

#include "bhtree/bhtree.hpp"

using namespace hpx;
using namespace std;
namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////
// Helpers
typedef hpx::naming::gid_type gid_type;

/// Global variables
static double dtime; // length of one time step
static double eps; // potential softening parameter
static double tolerance; // tolerance for stopping recursion, should be less than 0.57 for 3D case to bound error warren salmon 94
static double half_dt;
//http://www.cs.toronto.edu/~wayne/research/thesis/msc/node5.html
// force softening, i.e.,  replacing r^2 with (r ^ 2 + softening ^ 2) in the denominator of the gravitational force computation 
// for some small constant (softening factor epsilon (eps)) , usually chosen to approximate the average inter-particle separation. 
// This is done because it allows a smaller N to approximate a larger N, and also to eliminate the singularity at r=0
static double softening_2; // softening ^ 2 
static double inv_tolerance_2;
static int iter;
static int num_bodies;     // number of particles in system
static int num_iterations; // number of time steps to run

template <typename T>
T square(const T& value)
{
    return value * value;
}

/*
 * The N-Body tree is composed of two different kinds of particles
 * CELL - intermediate nodes
 * PAR - Terminal nodes which contain the body/particle.
 */
enum 
{
    CELL, PAR,
};

///////// Main Class Definitions
class TreeLeaf;
static TreeLeaf **particles; // the n particles

class TreeNode 
{
public:
    int node_type; /* Each node has a node type which is either NDE or PAR*/
    double mass;   /* Each node has a mass */
    double p[3];   /* position of each node is stored in a vector of type double named p */
};



class TreeLeaf    /* Terminal / leaf nodes extend intermediate nodes */
{                                 /* in addition to the mass, px, py, pz the internal */
public:                       /* nodes also have velocity vector of type double named v and */
    int node_type; /* Each node has a node type which is either NDE or PAR*/
    double mass;   /* Each node has a mass */
    double p[3];   /* position of each node is stored in a vector of type double named p */
    double v[3];              /* acceleration vector of type double named a  */
    double a[3];
    TreeLeaf()                /* TreeLeaf Constructor */
    {
        node_type = PAR;     /* Initialize TreeLeaf node to PARTICLE type */
        mass = 0.0;          /* Initialize mass to zero */
        p[0] = 0.0; /* Initialize position vector elements to 0 */
        p[1] = 0.0; /* Initialize position vector elements to 0 */
        p[2] = 0.0; /* Initialize position vector elements to 0 */
        v[0] = 0.0; /* Initialize velocity vector to 0 */
        v[1] = 0.0; /* Initialize velocity vector to 0 */
        v[2] = 0.0; /* Initialize velocity vector to 0 */
        a[0] = 0.0; /* Initialize acceleration vector to 0 */
        a[1] = 0.0; /* Initialize acceleration vector to 0 */
        a[2] = 0.0; /* Initialize acceleration vector to 0 */
    }
//    void moveParticles(); /* moves particles according to acceleration and velocity calculated */
//    void calculateForce(const IntrTreeNode * const root, const double box_size); /* calculates the acceleration and velocity of each particle in the tree */
    ~TreeLeaf() { }       /* TreeLeaf Destructor */
//private:
//    void forceCalcRe(const TreeNode * const n, double box_size_2); /* traverses tree recursively while calculating force for each particle */
};



static inline void processInput(string input_file) 
{
    ifstream infile;
    infile.open(input_file.c_str());
    if (!infile)                                /* if there is a problem opening file */
    {                                           /* exit gracefully */
        cerr << "Can't open input file " << input_file << endl;
        exit (1);
    }
    infile >> num_bodies;                    /* Read value from first line of the file into num_bodies */
    infile >> num_iterations;                /* Read number of iterations */
    infile >> dtime;                         /* Integration timestep: It's convenient to use a timestep which has an exact representation as a floating-point number; for example, a value n/d, where n is an integer and d is a power of two. To simplify specification of such timesteps, the code accepts fractions as well as ordinary floating-point values. If dtime=0, treecode does a single force calculation, outputs the result, and exits. */
    infile >> eps;                           /* Density smoothing length used in the gravitational force calculation. In effect, the mass distribution is smoothed by replacing each body by a Plummer sphere with scale length eps, and the gravitational field of this smoothed distribution is calculated */ 
    infile >> tolerance;                  
    half_dt = 0.5 * dtime;
    softening_2 = square<double>(eps);
    inv_tolerance_2 = 1.0 / (square<double>(tolerance));
    if (particles == NULL) 
    {
        particles = new TreeLeaf*[num_bodies];
        for (int i = 0; i < num_bodies; ++i)
            particles[i] = new TreeLeaf();
    }
    for(int i = 0; i < num_bodies ; ++i)
    {
        double dat[7] = {0,0,0,0,0,0,0};
        infile >> dat[0] >> dat[1] >> dat[2] >> dat[3] >> dat[4] >> dat[5] >> dat[6];
        particles[i]->mass = dat[0];
        particles[i]->p[0] = dat[1];
        particles[i]->p[1] = dat[2];
        particles[i]->p[2] = dat[3];
        particles[i]->v[0] = dat[4];
        particles[i]->v[1] = dat[5];
        particles[i]->v[2] = dat[6];
    }
    infile.close();
}

static inline void computeRootPos(const int num_bodies, double &box_dim, double center_position[]) 
{
    double minPos[3];
    minPos[0] = 1.0E90;
    minPos[1] = 1.0E90;
    minPos[2] = 1.0E90;
    double maxPos[3];
    maxPos[0] = -1.0E90;
    maxPos[1] = -1.0E90;
    maxPos[2] = -1.0E90;
    for (int i = 0; i < num_bodies; ++i)
    {
        if (minPos[0] > particles[i]->p[0])
            minPos[0] = particles[i]->p[0];
        if (minPos[1] > particles[i]->p[1])
            minPos[1] = particles[i]->p[1];
        if (minPos[2] > particles[i]->p[2])
            minPos[2] = particles[i]->p[2];
        if (maxPos[0] < particles[i]->p[0])
            maxPos[0] = particles[i]->p[0];
        if (maxPos[1] < particles[i]->p[1])
            maxPos[1] = particles[i]->p[1];
        if (maxPos[2] < particles[i]->p[2])
            maxPos[2] = particles[i]->p[2];
    }
    box_dim = maxPos[0] - minPos[0];
    if (box_dim < (maxPos[1] - minPos[1]))
        box_dim = maxPos[1] - minPos[1];
    if (box_dim < (maxPos[2] - minPos[2]))
        box_dim = maxPos[2] - minPos[2];
    center_position[0] = (maxPos[0] + minPos[0]) / 2;
    center_position[1] = (maxPos[1] + minPos[1]) / 2;
    center_position[2] = (maxPos[2] + minPos[2]) / 2;
}

int hpx_main(po::variables_map &vm)
{
    // Time Measurement Variables and Related Calls
    timeval t1, t2;
    double iterChron[6];
    double progChron[6];

    std::string input_file;

    get_option(vm, "input_file", input_file);
    LAPP_(info) << "Nbody, heck yeah!" ;
    if(input_file.size() == 0)
    {
        hpx_finalize();
        return 0;
    }
    LAPP_(info) << "Using input file '" << input_file << "'";


    particles = NULL;
    processInput(input_file);
    
        // get list of all known localities
    std::vector<hpx::naming::gid_type> prefixes;
    hpx::naming::gid_type prefix;
    hpx::applier::applier& appl = hpx::applier::get_applier();
    if (appl.get_remote_prefixes(prefixes)) {
        // create accumulator on any of the remote localities
        prefix = prefixes[0];
    }
    else {
        // create an accumulator locally
        prefix = appl.get_runtime_support_raw_gid();
    }
    
    
    //cout << "num iterations " << num_iterations << endl; 
    for (iter = 0; iter < num_iterations; ++iter) 
    { 
        double box_size, center_position[3];
        computeRootPos(num_bodies, box_size, center_position);
        cout << "center pos " << center_position[0] << " " << center_position[1] <<" " << center_position[2] << endl;
        using hpx::components::IntrTreeNode;
        IntrTreeNode bht_root;
        bht_root.create(naming::id_type(prefix,naming::id_type::unmanaged));
        bht_root.newNode(center_position[0], center_position[1], center_position[2]);
    }


    //  IntrTreeNode *bht_root = IntrTreeNode::newNode(center_position); // create the tree's root

    //  const double sub_box_size = box_size * 0.5;
    //  for (int i = 0; i < num_bodies; ++i) 
    //  {
    //      bht_root->treeNodeInsert(particles[i], sub_box_size); // grow the tree by inserting each body
    //  }
    //}
    hpx_finalize();
    return 0;
}


/// New Main function HPX Style
int main(int argc, char* argv[])
{
    int retcode;
    try {
        // Configure application-specific options
        po::options_description 
            desc_commandline("Usage: nbody [hpx_options] [options]");
        desc_commandline.add_options()
            ("input_file,i", po::value<std::string>(), 
            "asdfasdfasdfasdfasfdasdfadsf")
            ("output_file,o", po::value<std::string>(), 
            "asdfasdfasdfasdfasfdasdfadsf")
            ;
        // Check command line arguments.
        std::string hpx_host("localhost"), agas_host;
        boost::uint16_t hpx_port = HPX_PORT, agas_port = 0;
        int num_threads = 1;
        hpx::runtime::mode mode = hpx::runtime::console;    // default is console mode
        int num_localities = 1;
        // Initialize and run HPX
        retcode = hpx_init(desc_commandline, argc, argv); 
    }
    catch (std::exception& e) {
        std::cerr << "std::exception caught: " << e.what() << "\n";
        return -1;
    }
    catch (...) {
        std::cerr << "unexpected exception caught\n";
        return -2;
    }
    return 0;
}
