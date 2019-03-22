//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/lightweight_test.hpp>
//
#include <hpx/runtime/basename_registration.hpp>
#include <hpx/runtime/parcelset/rma/rma_object.hpp>
//
#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <type_traits>
#include <vector>
#include <array>

///////////////////////////////////////////////////////////////////////////////
using namespace hpx::parcelset::policies;
using namespace hpx::parcelset::rma;
//
struct dummy_data {
    std::array<char,16384> data;
    //
    dummy_data() {};
};
//
HPX_IS_RMA_ELIGIBLE(dummy_data);
//
const int up = 0;
const int down = 1;
//
std::array<rma_object<dummy_data>, 2> recv;
std::array<rma_object<dummy_data>, 2> send;
//
void test_rdma_1(hpx::id_type loc)
{
    static const char* up_name = "/rdma/up/";
    static const char* down_name = "/rdma/down/";
    //
    std::size_t rank = hpx::get_locality_id();
    std::size_t  num = hpx::get_num_localities(hpx::launch::sync);
    //
    if (rank > 0)
    {
        //recv[up] = hpx::find_from_basename<rma_object<dummy_data>>(down_name, rank - 1);
        send[up] = hpx::parcelset::rma::make_rma_object<dummy_data>();
        //hpx::register_with_basename(down_name, send[up], rank);
    }
    if (rank < num - 1)
    {
        //recv[down] = hpx::find_from_basename<channel_type>(up_name, rank + 1);
        //send[down] = hpx::rma::make_rma_object<dummy_data>();
        //hpx::register_with_basename(up_name, send[down], rank);
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    for (hpx::id_type const& id : hpx::find_remote_localities())
    {
        test_rdma_1(id);
    }

    // compare number of parcels with number of messages generated
    //print_counters("/parcels/count/*/sent");
    //print_counters("/messages/count/*/sent");

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run")
        ;

    // explicitly disable message handlers (parcel coalescing)
    std::vector<std::string> const cfg = {
        "hpx.parcel.message_handlers=0"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
