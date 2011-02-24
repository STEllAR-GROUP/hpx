//  Copyright (c) 2010-2011 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

using namespace hpx;
namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////
// int zero(void)
// {
//     return 0;
// }
//
// int identity(int x)
// {
//     return i;
// }
//
// int sum(int a, int b)
// {
//     return a + b;
// }

int zero(void)
{ 
    std::cout << "Computing 'zero()'" << std::endl;

    return 0; 
}
typedef actions::plain_result_action0<int, zero> zero_action;
HPX_REGISTER_PLAIN_ACTION(zero_action);

typedef hpx::lcos::detail::thunk<zero_action> zero_thunk;
HPX_REGISTER_THUNK(zero_thunk);

int identity(int x) 
{ 
    std::cout << "Computing 'identity(" << x << ")'" << std::endl;
    
    return x; 
}
typedef actions::plain_result_action1<int, int, identity> identity_action;
HPX_REGISTER_PLAIN_ACTION(identity_action);

typedef hpx::lcos::detail::thunk<identity_action> identity_thunk;
HPX_REGISTER_THUNK(identity_thunk);

int sum(int a, int b) 
{
    std::cout << "Computing 'sum(" << a << "," << b << ")'" << std::endl;
    
    return a + b; 
}
typedef actions::plain_result_action2<int, int, int, sum> sum_action;
HPX_REGISTER_PLAIN_ACTION(sum_action);

typedef hpx::lcos::detail::thunk<sum_action> sum_thunk;
HPX_REGISTER_THUNK(sum_thunk);

////////////////////////////////////////////////////////////////////////////////
int test(id_type s_id)
{
  std::cout << "test> sum = apply(get, eager_future(get, s_id))" << std::endl;
  int sum = lcos::eager_future<sum_thunk_get_action>(s_id).get();
  
  std::cout << "test> print sum" << std::endl;
  std::cout << sum << std::endl;

  std::cout << "test> return sum" << std::endl;
  return sum;
}
typedef actions::plain_result_action1<int, id_type, test> test_action;
HPX_REGISTER_PLAIN_ACTION(test_action);

///////////////////////////////////////////////////////////////////////////////
typedef lcos::thunk_client<zero_thunk> zero_thunk_type;
typedef lcos::thunk_client<identity_thunk> identity_thunk_type;
typedef lcos::thunk_client<sum_thunk> sum_thunk_type;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(po::variables_map &vm)
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
        applier::apply<zero_thunk_trigger_action>(z.get_gid());

        std::cout << "  ... do a bunch of stuff ..." << std::endl;

        std::cout << ">>> z_f = eager_future(get, z)" << std::endl;
        lcos::future_value<int> z_f =
            lcos::eager_future<zero_thunk_get_action>(z.get_gid());

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
        lcos::future_value<int> id_f =
            lcos::eager_future<identity_thunk_get_action>(id.get_gid());

        std::cout << ">>> print id_f" << std::endl;
        std::cout << id_f.get() << std::endl << std::endl;
    }

    // Test get from a remote action
    {
        std::cout << "Testing get from a remote action:" << std::endl;
        std::cout << ">>> s = thunk(sum, 23, 42) here" << std::endl;
        sum_thunk_type s(here, 23, 42);

        std::cout << ">>> test_f = eager_future(test, s) there" << std::endl;
        lcos::future_value<int> test_f =
            lcos::eager_future<test_action>(there, s.get_gid());

        std::cout << ">>> apply(trigger, s)" << std::endl;
        applier::apply<zero_thunk_trigger_action>(s.get_gid());

        std::cout << ">>> print test_f" << std::endl;
        std::cout << test_f.get() << std::endl;
    }

    // initiate shutdown of the runtime systems on all localities
    hpx::finalize();

    std::cout << "Test passed" << std::endl;

    return 0;
}


///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{

  // Configure application-specific options
  po::options_description
      desc_commandline("Usage: thunk_tests [hpx_options] [options]");

  // Initialize and run HPX
  int retcode = hpx::init(desc_commandline, argc, argv);
  return retcode;
}

