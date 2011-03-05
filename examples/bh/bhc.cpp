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

#include "node/server/node.hpp"
#include "node/node.hpp"

using namespace hpx;
using namespace std;
namespace po = boost::program_options;

typedef hpx::naming::gid_type gid_type;

typedef struct {
    double dtime;
    double eps;
    double tolerance;
    double half_dt;
    double softening_2;
    double inv_tolerance_2;
    int iter;
    int num_bodies;
    int num_iterations;
} crit_vals;

//*********************    
int hpx_main(boost::program_options::variables_map &vm)
{
    std::string input_file;
    crit_vals cv;
    
    hpx::get_option(vm, "input_file", input_file);
    LAPP_(info) << "Nbody, heck yeah!" ;
    if(input_file.size() == 0)
    {
        hpx::finalize();
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
    infile >> cv.num_bodies;                   
    infile >> cv.num_iterations;               
    infile >> cv.dtime;                        
    infile >> cv.eps;                          
    infile >> cv.tolerance;                  
    cv.half_dt = 0.5 * cv.dtime;
    cv.softening_2 = cv.eps * cv.eps;
    cv.inv_tolerance_2 = 1.0 / (cv.tolerance * cv.tolerance);    
    
   
    hpx::components::component_type node_type = hpx::components::get_component_type<hpx::components::node::server::node>();
    typedef components::distributing_factory::result_type result_type;
    typedef components::distributing_factory factory;
    factory fac;
    fac.create(applier::get_applier().get_runtime_support_gid());
    result_type bodies = fac.create_components(node_type, cv.num_bodies);
    
    components::distributing_factory::iterator_range_type bod_range = locality_results(bodies);
    
    components::distributing_factory::iterator_type bBeg ; 
    components::distributing_factory::iterator_type bEnd = bod_range.second;
    for (bBeg = bod_range.first; bBeg != bEnd; ++bBeg)
    {
        double dat[7] = {0,0,0,0,0,0,0};
        infile >> dat[0] >> dat[1] >> dat[2] >> dat[3] >> dat[4] >> dat[5] >> dat[6];
        components::node::stubs::node::set_mass((*bBeg),dat[0]);
        components::node::stubs::node::set_pos((*bBeg), dat[1], dat[2], dat[3]);
        components::node::stubs::node::set_vel((*bBeg), dat[4], dat[5], dat[6]);
        components::node::stubs::node::print((*bBeg));
    }       
    infile.close();  

    hpx::finalize();
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
        retcode = hpx::init(desc_commandline, argc, argv); 
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
