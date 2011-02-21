//  Copyright (c) 2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <boost/detail/lightweight_test.hpp>
#include <boost/program_options.hpp>

using namespace hpx;
namespace po = boost::program_options;

actions::manage_object_action<boost::uint8_t> const raw_memory =
    actions::manage_object_action<boost::uint8_t>();

///////////////////////////////////////////////////////////////////////////////
int hpx_main(po::variables_map &vm)
{
    components::memory_block mb;
    mb.create(find_here(), sizeof(boost::uint32_t), raw_memory);
    components::access_memory_block<boost::uint32_t> data(mb.get());

    boost::uint32_t* value = data.get_ptr();

    new (value) boost::uint32_t(17);

    // initiate shutdown of the runtime system
    components::stubs::runtime_support::shutdown_all();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    po::options_description desc_commandline("usage: basic [options]");

    // Initialize and run HPX
    BOOST_TEST_EQ(hpx_init(desc_commandline, argc, argv), 0);
    return boost::report_errors();
}

