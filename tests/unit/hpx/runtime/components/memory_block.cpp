//  Copyright (c) 2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>

hpx::actions::manage_object_action<boost::uint8_t> const raw_memory =
    hpx::actions::manage_object_action<boost::uint8_t>();
    
typedef hpx::components::server::detail::memory_block::checkin_action
    checkin_action_type;
    
typedef hpx::components::server::detail::memory_block::get_action
    get_action_type;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map&)
{
    hpx::components::memory_block mb;
    hpx::naming::id_type here = hpx::find_here();
    mb.create(here, sizeof(boost::uint32_t)*2, raw_memory);
    hpx::components::access_memory_block<boost::uint32_t> data0(mb.get());
    boost::uint32_t* ptr0 = data0.get_ptr();

    // store a value into memory
    new (&ptr0[0]) boost::uint32_t(17);

    // read a value from memory
    boost::uint32_t& value0 = data0.get();
    HPX_TEST_EQ(value0, 17U);

    // modify a value and check it in (using an eager future)
    hpx::components::memory_block_data data1 = mb.get();
    boost::uint32_t* ptr1 = reinterpret_cast<boost::uint32_t*>(data1.get_ptr());
    ptr1[0] = 42; 
    hpx::lcos::eager_future<checkin_action_type> ef0(mb.get_gid(), data1); 
    ef0.get();
 
    // read the modified value back (using an eager_future)
    hpx::lcos::eager_future<get_action_type> ef1(mb.get_gid());
    hpx::components::access_memory_block<boost::uint32_t> data2(ef1.get());
    boost::uint32_t& value1 = data2.get_ptr()[0];
    HPX_TEST_EQ(value1, 42U);

    // initiate shutdown of the runtime system
    hpx::finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv), 0,
      "HPX main exited with non-zero status");
    return hpx::util::report_errors();
}

