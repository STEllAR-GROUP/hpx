//  Copyright (c) 2010-2011 Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::actions::plain_result_action0;
using hpx::actions::plain_result_action1;
using hpx::actions::plain_result_action2;

using hpx::find_here;
using hpx::init;
using hpx::finalize;

using hpx::naming::id_type;

using hpx::util::report_errors;

using hpx::lcos::lazy_future;

///////////////////////////////////////////////////////////////////////////////
int zero() { return 0; }
typedef plain_result_action0<int, zero> zero_action;
HPX_REGISTER_PLAIN_ACTION(zero_action);

///////////////////////////////////////////////////////////////////////////////
int identity(int x) { return x; }
typedef plain_result_action1<int, int, identity> identity_action;
HPX_REGISTER_PLAIN_ACTION(identity_action);

///////////////////////////////////////////////////////////////////////////////
int sum(int a, int b) { return a + b; }
typedef plain_result_action2<int, int, int, sum> sum_action;
HPX_REGISTER_PLAIN_ACTION(sum_action);

///////////////////////////////////////////////////////////////////////////////
typedef lazy_future<zero_action> zero_lazy_future;
typedef lazy_future<identity_action> identity_lazy_future;
typedef lazy_future<sum_action> sum_lazy_future;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    id_type here = find_here();

    std::size_t iterations = 0;

    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    for (std::size_t i = 0; i < iterations; ++i)
    {
        { // zero
            zero_lazy_future zero(here);
    
            HPX_TEST_EQ(zero.get(), 0);
            HPX_TEST_EQ(zero.get(), 0);
        }
    
        { // identity
            identity_lazy_future identity(here, 42);
    
            HPX_TEST_EQ(identity.get(), 42);
            HPX_TEST_EQ(identity.get(), 42);
        }
    
        { // sum
            sum_lazy_future sum(here, 42, 42);
    
            HPX_TEST_EQ(sum.get(), 84);
            HPX_TEST_EQ(sum.get(), 84);
        }
    }

    // initiate shutdown of the runtime systems on all localities
    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description
        desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");
    
    desc_commandline.add_options()
        ("iterations", value<std::size_t>()->default_value(1 << 6), 
            "the number of times to repeat the test") 
        ;

    // Initialize and run HPX.
    HPX_TEST_EQ_MSG(init(desc_commandline, argc, argv), 0,
      "HPX main exited with non-zero status");
    return report_errors();
}

