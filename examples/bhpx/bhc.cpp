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
#include <vector>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
//#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "tlf/server/tlf.hpp"
#include "tlf/tlf.hpp"

#include "itn/server/itn.hpp"
#include "itn/itn.hpp"


// HPX_REGISTER_COMPONENT_MODULE();
// typedef hpx::components::managed_component<hpx::components::tlf::server::tlf> tlf_type;
// HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(tlf_type, tlf);
// HPX_DEFINE_GET_COMPONENT_TYPE(tlf_type::wrapped_type);
// HPX_REGISTER_ACTION_EX(tlf_type::wrapped_type::set_mass_action, tlf_set_mass_action);
// HPX_REGISTER_ACTION_EX(tlf_type::wrapped_type::set_pos_action, tlf_set_pos_action);
// HPX_REGISTER_ACTION_EX(tlf_type::wrapped_type::set_vel_action, tlf_set_vel_action);
// HPX_REGISTER_ACTION_EX(tlf_type::wrapped_type::set_acc_action, tlf_set_acc_action);
// HPX_REGISTER_ACTION_EX(tlf_type::wrapped_type::print_action, tlf_print_action);
// HPX_REGISTER_ACTION_EX(tlf_type::wrapped_type::get_pos_action, tlf_get_pos_action);
// HPX_REGISTER_ACTION_EX(tlf_type::wrapped_type::get_type_action, tlf_get_type_action);
// 
// 
// HPX_REGISTER_ACTION_EX(hpx::lcos::base_lco_with_value<int>::set_result_action, set_result_action_int);
// HPX_REGISTER_ACTION_EX(hpx::lcos::base_lco_with_value<std::vector<double> >::set_result_action, set_result_action_vector_double)
// HPX_DEFINE_GET_COMPONENT_TYPE(hpx::lcos::base_lco_with_value<std::vector<double> >);
// HPX_DEFINE_GET_COMPONENT_TYPE(hpx::lcos::base_lco_with_value<int>);


using namespace hpx;
using namespace std;
namespace po = boost::program_options;

typedef hpx::naming::gid_type gid_type;

static double dtime; 
static double eps; 
static double tolerance;
static double half_dt;
static double softening_2; 
static double inv_tolerance_2;
static int iter;
static int num_bodies;     
static int num_iterations; 

template <typename T>
T square(const T& value)
{
    return value * value;
}

//*********************    
static inline void computeRootPos(const int num_bodies, double &box_dim, double center_position[], components::distributing_factory::result_type bodies) 
{
   double minPos[3];
   minPos[0] = 1.0E90;
   minPos[1] = 1.0E90;
   minPos[2] = 1.0E90;
   double maxPos[3];
   maxPos[0] = -1.0E90;
   maxPos[1] = -1.0E90;
   maxPos[2] = -1.0E90;
   
   components::distributing_factory::iterator_range_type bod_range = locality_results(bodies);
   components::distributing_factory::iterator_type bBeg = bod_range.first; 
   components::distributing_factory::iterator_type bEnd = bod_range.second;
   for (int i = 0; bBeg != bEnd; ++i, ++bBeg)
   {
       std::vector<double> bPos = components::tlf::stubs::tlf::get_pos((*bBeg));
       
       if (minPos[0] > bPos[0])
           minPos[0] = bPos[0];
       if (minPos[1] > bPos[1])
           minPos[1] = bPos[1];
       if (minPos[2] > bPos[2])
           minPos[2] = bPos[2];
       if (maxPos[0] < bPos[0])
           maxPos[0] = bPos[0];
       if (maxPos[1] < bPos[1])
           maxPos[1] = bPos[1];
       if (maxPos[2] < bPos[2])
           maxPos[2] = bPos[2];
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
//*********************    
int hpx_main(po::variables_map &vm)
{
    std::string input_file;
    get_option(vm, "input_file", input_file);
    LAPP_(info) << "Nbody, heck yeah!" ;
    if(input_file.size() == 0)
    {
        hpx_finalize();
        return 0;
    }
    LAPP_(info) << "Using input file '" << input_file << "'";
        
    ifstream infile;
    infile.open(input_file.c_str());
    if (!infile)                                /* if there is a problem opening file */
    {                                           /* exit gracefully */
        std::cerr << "Can't open input file " << input_file << std::endl;
        exit (1);
    }
    infile >> num_bodies;                   
    infile >> num_iterations;               
    infile >> dtime;                        
    infile >> eps;                          
    infile >> tolerance;                  
    half_dt = 0.5 * dtime;
    softening_2 = square<double>(eps);
    inv_tolerance_2 = 1.0 / (square<double>(tolerance));    
   
    hpx::components::component_type tlf_type = hpx::components::get_component_type<hpx::components::tlf::server::tlf>();
    typedef components::distributing_factory::result_type result_type;
    typedef components::distributing_factory factory;
    factory fac;
    fac.create(applier::get_applier().get_runtime_support_gid());
    result_type bodies = fac.create_components(tlf_type, num_bodies);
    
    components::distributing_factory::iterator_range_type bod_range = locality_results(bodies);
    
    components::distributing_factory::iterator_type bBeg ; 
    components::distributing_factory::iterator_type bEnd = bod_range.second;
    for (bBeg = bod_range.first; bBeg != bEnd; ++bBeg)
    {
        double dat[7] = {0,0,0,0,0,0,0};
        infile >> dat[0] >> dat[1] >> dat[2] >> dat[3] >> dat[4] >> dat[5] >> dat[6];
        components::tlf::stubs::tlf::set_mass((*bBeg),dat[0]);
        components::tlf::stubs::tlf::set_pos((*bBeg), dat[1], dat[2], dat[3]);
        components::tlf::stubs::tlf::set_vel((*bBeg), dat[4], dat[5], dat[6]);
        components::tlf::stubs::tlf::print((*bBeg));
    }
    
            // get list of all known localities
    std::vector<hpx::naming::id_type> prefixes;
    hpx::naming::id_type prefix;
    hpx::applier::applier& appl = hpx::applier::get_applier();
    if (appl.get_remote_prefixes(prefixes)) {
        // create accumulator on any of the remote localities
        prefix = prefixes[0];
    }
    else {
        // create an accumulator locally
        prefix = appl.get_runtime_support_gid();
    }
    
    for (iter = 0; iter < num_iterations; ++iter)
    {
        double box_size, cPos[3];
        computeRootPos(num_bodies, box_size, cPos, bodies);
        hpx::components::itn::itn tree_root;
        tree_root.create(prefix);
        tree_root.new_node(cPos[0], cPos[1], cPos[2]);
        
        const double sub_box_size = box_size * 0.5;
//         for (bBeg = bod_range.first; bBeg != bEnd; ++bBeg)
//         {
//             tree_root.insert_body()
//         }

        
        
        tree_root.free();
    }
       
    infile.close();  
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
            "input file")
            ("output_file,o", po::value<std::string>(), 
            "output file")
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
