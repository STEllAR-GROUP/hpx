//  Copyright (c) 2007-2010 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map&)
{
    // try to access some memory directly
    boost::uint32_t value = 0;

    // store a value to memory
    hpx::naming::id_type memid = hpx::applier::get_applier().get_memory_gid();
    typedef hpx::components::server::memory::store32_action store_action_type;
    hpx::applier::apply<store_action_type>(memid, boost::uint64_t(&value), 1);

    HPX_TEST_EQ(value, 1U);

    // read the value back from memory (using an eager_future)
    typedef hpx::components::server::memory::load32_action load_action_type;
    hpx::lcos::eager_future<load_action_type> ef(memid, boost::uint64_t(&value));

    boost::uint32_t result1 = ef.get();
    HPX_TEST_EQ(result1, value);

    // read the value back from memory (using a lazy_future)
    hpx::lcos::lazy_future<load_action_type> lf(memid, boost::uint64_t(&value));

    boost::uint32_t result2 = lf.get();
    HPX_TEST_EQ(result2, value);

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

