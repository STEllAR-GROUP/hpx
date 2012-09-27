//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>

#include <boost/assign/std.hpp>

#include <hpx/include/performance_counters.hpp>
#include <examples/performance_counters/mem_counter.hpp>

#include "sheneos/interpolator.hpp"

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

typedef hpx::performance_counters::server::proc_statm proc_statm_type; 
 
namespace read_mem = hpx::performance_counters::server; 
 
void register_counter_type() 
{ 
    namespace pc = hpx::performance_counters; 
    pc::install_counter_type( 
        "/memory/vm", 
        &read_mem::read_psm_vm, 
        "returns the virtual memory for the pid value of process which calls this counter"
    ); 
    pc::install_counter_type( 
        "/memory/resident", 
        &read_mem::read_psm_resident, 
        "returns the virtual memory for the pid value of process which calls this counter"
    ); 
}

///////////////////////////////////////////////////////////////////////////////
inline bool
eval(char const* expr, sheneos::interpolator& shen, double ye,
    double temp, double rho, std::vector<double>& expected)
{
    std::vector<double> results = shen.interpolate(ye, temp, rho);
    std::cout << expr << std::endl;
    std::cout << std::string(std::strlen(expr), '-') << std::endl;

    if (results.size() != expected.size()) {
        std::cout << "Result size mismatch, got: " << results.size()
                  << ", expected: " << expected.size() << std::endl;
        return false;
    }

    for (std::size_t i = 0; i < results.size(); ++i) {
        std::cout << results[i]
                  << ", expected: " << expected[i]
                  << std::endl;
    }

    return true;
}

int hpx_main(variables_map& vm)
{
    std::string const datafilename = vm["file"].as<std::string>();
    int num_partitions = 27;

    {
        char const* shen_symbolic_name = "/sheneos_client/test";

        using namespace boost::assign;
        std::vector<double> expected;
        expected += 9.809012e+34;       // pressure
        expected += 1.602810e+20;       // energy
        expected += 2.843643e+00;       // entropy
        expected += 4.151515e+01;       // munu
        expected += 3.960476e+20;       // cs2
        expected += 2.052315e+08;       // dedt
        expected += 2.864134e+20;       // dpdrhoe
        expected += 3.556341e+14;       // dpderho

        // create the distributed interpolation object on num_localities
        sheneos::interpolator shen;
        shen.create(datafilename, shen_symbolic_name, num_partitions);

        eval("shen(0.2660725, 63.0, std::pow(10., 14.74994))", shen,
            0.2660725, 63.0, std::pow(10., 14.74994), expected);

        std::cout << std::endl << std::endl;

        // create a second client instance connected to the already existing
        // interpolation object
        sheneos::interpolator shen_connected;
        shen_connected.connect(shen_symbolic_name);

        eval("shen(0.2660725, 63.0, std::pow(10., 14.74994))", shen_connected,
            0.2660725, 63.0, std::pow(10., 14.74994), expected);

        std::cout << std::endl << std::endl;
    }

    hpx::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // register memory counters
    hpx::register_startup_function(&register_counter_type); 
    
    // Configure application-specific options
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("file", value<std::string>()->default_value(
                "sheneos_220r_180t_50y_extT_analmu_20100322_SVNr28.h5"),
            "name of HDF5 data file containing the Shen EOS tables")
        ;

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}

