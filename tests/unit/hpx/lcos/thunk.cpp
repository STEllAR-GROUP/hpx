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
using hpx::get_runtime;
using hpx::init;
using hpx::finalize;

using hpx::applier::apply;

using hpx::naming::id_type;

using hpx::util::report_errors;

using hpx::lcos::detail::thunk;
using hpx::lcos::thunk_client;
using hpx::lcos::eager_future;
using hpx::lcos::future_value;

///////////////////////////////////////////////////////////////////////////////
int zero(void)
{ 
    std::cout << "Computing 'zero()'" << std::endl;
    return 0; 
}
typedef plain_result_action0<int, zero> zero_action;
HPX_REGISTER_PLAIN_ACTION(zero_action);
typedef thunk<zero_action> zero_thunk;
HPX_REGISTER_THUNK(zero_thunk);

///////////////////////////////////////////////////////////////////////////////
int identity(int x) 
{ 
    std::cout << "Computing 'identity(" << x << ")'" << std::endl;
    return x; 
}
typedef plain_result_action1<int, int, identity> identity_action;
HPX_REGISTER_PLAIN_ACTION(identity_action);
typedef thunk<identity_action> identity_thunk;
HPX_REGISTER_THUNK(identity_thunk);

///////////////////////////////////////////////////////////////////////////////
int sum(int a, int b) 
{
    std::cout << "Computing 'sum(" << a << "," << b << ")'" << std::endl;
    return a + b; 
}
typedef plain_result_action2<int, int, int, sum> sum_action;
HPX_REGISTER_PLAIN_ACTION(sum_action);
typedef thunk<sum_action> sum_thunk;
HPX_REGISTER_THUNK(sum_thunk);

////////////////////////////////////////////////////////////////////////////////
int thunk_test(id_type const& s_id)
{
    std::cout << "test> sum = apply(get, eager_future(get, s_id))" << std::endl;
    int sum = eager_future<sum_thunk_get_action>(s_id).get();
  
    std::cout << "test> print sum" << std::endl;
    std::cout << sum << std::endl;

    std::cout << "test> return sum" << std::endl;
    return sum;
}
typedef plain_result_action1<int, id_type const&, thunk_test> thunk_test_action;
HPX_REGISTER_PLAIN_ACTION(thunk_test_action);

///////////////////////////////////////////////////////////////////////////////
typedef thunk_client<zero_thunk> zero_thunk_type;
typedef thunk_client<identity_thunk> identity_thunk_type;
typedef thunk_client<sum_thunk> sum_thunk_type;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map &vm)
{
    id_type here = find_here();
    id_type there = get_runtime().get_process().next();

    std::cout << ">>> print here, there" << std::endl;
    std::cout << here << " " << there << std::endl << std::endl;

    // Test local eager evaluation
    {
        std::cout << "Testing local eager evaluation:" << std::endl;
        std::cout << ">>> z = thunk(zero) here" << std::endl;
        zero_thunk_type z(here);

        std::cout << ">>> apply(trigger, z)" << std::endl;
        apply<zero_thunk_trigger_action>(z.get_gid());

        std::cout << "  ... do a bunch of stuff ..." << std::endl;

        std::cout << ">>> z_f = eager_future(get, z)" << std::endl;
        future_value<int> z_f =
            eager_future<zero_thunk_get_action>(z.get_gid());

        std::cout << ">>> print f" << std::endl;
        std::cout << z_f.get() << std::endl << std::endl;
    }

    // Test remote lazy evaluation
    {
        std::cout << "Testing remote lazy evaluation:" << std::endl;
        std::cout << ">>> id = thunk(identity, 42) there" << std::endl;
        identity_thunk_type id(there, 42);

        std::cout << "  ... do a bunch of stuff ..." << std::endl;

        std::cout << ">>> id_f = eager_future(get, id)" << std::endl;
        future_value<int> id_f =
            eager_future<identity_thunk_get_action>(id.get_gid());

        std::cout << ">>> print id_f" << std::endl;
        std::cout << id_f.get() << std::endl << std::endl;
    }

    // Test get from a remote action
    {
        std::cout << "Testing get from a remote action:" << std::endl;
        std::cout << ">>> s = thunk(sum, 23, 42) here" << std::endl;
        sum_thunk_type s(here, 23, 42);

        std::cout << ">>> test_f = eager_future(test, s) there" << std::endl;
        future_value<int> test_f =
            eager_future<thunk_test_action>(there, s.get_gid());

        std::cout << ">>> apply(trigger, s)" << std::endl;
        apply<zero_thunk_trigger_action>(s.get_gid());

        std::cout << ">>> print test_f" << std::endl;
        std::cout << test_f.get() << std::endl;
    }

    // initiate shutdown of the runtime systems on all localities
    finalize();

    std::cout << "Test passed" << std::endl;

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
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

