//  Copyright (c) 2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/program_options.hpp>

using namespace hpx;
namespace po = boost::program_options;

actions::manage_object_action<boost::uint8_t> const raw_memory =
    actions::manage_object_action<boost::uint8_t>();
    
typedef components::server::detail::memory_block::checkin_action
    checkin_action_type;
    
typedef components::server::detail::memory_block::get_action
    get_action_type;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(po::variables_map &vm)
{
    components::memory_block mb;
    naming::id_type here = find_here();
    mb.create(here, sizeof(boost::uint32_t)*2, raw_memory);
    components::access_memory_block<boost::uint32_t> data0(mb.get());
    boost::uint32_t* ptr0 = data0.get_ptr();

    // store a value into memory
    new (&ptr0[0]) boost::uint32_t(17);

    // read a value from memory
    boost::uint32_t& value0 = data0.get();
    BOOST_TEST_EQ(value0, 17);

    // modify a value and check it in (using an eager future)
    components::memory_block_data data1 = mb.get();
    boost::uint32_t* ptr1 = reinterpret_cast<boost::uint32_t*>(data1.get_ptr());
    ptr1[0] = 42; 
    lcos::eager_future<checkin_action_type> ef0(mb.get_gid(), data1); 
    ef0.get();
 
    // read the modified value back (using an eager_future)
    lcos::eager_future<get_action_type> ef1(mb.get_gid());
    components::access_memory_block<boost::uint32_t> data2(ef1.get());
    boost::uint32_t& value1 = data2.get_ptr()[0];
    BOOST_TEST_EQ(value1, 42);

    // initiate shutdown of the runtime system
    hpx::finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    po::options_description
       desc_commandline
          ("usage: " BOOST_PP_STRINGIZE(HPX_APPLICATION_NAME) " [options]");

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv), 0,
      "HPX main exited with non-zero status");
    return boost::report_errors();
}


