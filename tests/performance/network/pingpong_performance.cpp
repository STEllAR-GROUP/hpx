//  Copyright (c) 2017 Bibek Wagle
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/serialization.hpp>

#include <cstddef>
#include <complex>
#include <string>
#include <vector>

namespace pingpong
{
    namespace server
    {
        std::complex<double> get_element()
        {
            return std::complex<double>(13.3,-23.8);
        }
    }
}

HPX_PLAIN_ACTION(pingpong::server::get_element, pingpong_get_element_action);
//HPX_ACTION_USES_MESSAGE_COALESCING(pingpong_get_element_action);


int hpx_main(boost::program_options::variables_map& vm)
{
   //Commandline specific code
    std::size_t const n = vm["nparcels"].as<std::size_t>();

    if (0 == hpx::get_locality_id())
    {
        hpx::cout << "Running With nparcel = " << n << "\n" << hpx::flush;
    }

    //Create instance of the actions
    pingpong_get_element_action act;
    std::vector<hpx::future<std::complex<double>>> vec;
    std::vector<std::complex<double>> recieved;
    vec.reserve(n);
    recieved.reserve(n);
    //Find the other locality
    std::vector<hpx::naming::id_type> dummy = hpx::find_remote_localities();
    hpx::naming::id_type other_locality = dummy[0];


    for(std::size_t i=0; i<n; ++i)
    {
        vec.push_back(hpx::async(act,other_locality));
    }

    hpx::when_all(vec).then(
        [&recieved, n](hpx::future<std::vector<hpx::future<std::complex<double>>>> dummy)
        {
            std::vector<hpx::future<std::complex<double>>> number = dummy.get();
            for (std::size_t i = 0; i < n; ++i)
            {
                recieved.push_back(number[i].get());
            }
            hpx::evaluate_active_counters(false, " All Futures Done");
            hpx::cout << "Now Done With Lambda and the last recieved value is "
                      <<recieved[n-1]<< "\n" << hpx::flush;
        }
    ).get();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Configure application-specific options
    boost::program_options::options_description cmdline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ("nparcels,n",
         boost::program_options::value<std::size_t>()->default_value(100),
         "the number of parcels to create")
        ;
    // Initialize and run HPX
    std::vector<std::string> cfg;
    cfg.push_back("hpx.run_hpx_main!=1");
    return hpx::init(cmdline,argc, argv, cfg);
}
