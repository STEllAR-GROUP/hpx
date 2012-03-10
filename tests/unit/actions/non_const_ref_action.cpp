////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <tests/unit/actions/components/non_const_ref_action_component.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;

using hpx::actions::plain_action1;
using hpx::actions::plain_result_action1;
using hpx::actions::plain_direct_action1;
using hpx::actions::plain_direct_result_action1;

using hpx::lcos::eager_future;
using hpx::lcos::async;

using hpx::init;
using hpx::finalize;
using hpx::find_here;

void non_const_ref_void(int && i)
{
    HPX_TEST_EQ(i, 9);
}

int non_const_ref(int && i)
{
    return i;
}

typedef plain_action1<
    int &&
  , non_const_ref_void
> non_const_ref_void_action;

/*
typedef plain_result_action1<
    int
  , int &&
  , non_const_ref
> non_const_ref_result_action;

typedef plain_direct_action1<
    int &&
  , non_const_ref_void
> non_const_ref_void_direct_action;

typedef plain_direct_result_action1<
    int
  , int &&
  , non_const_ref
> non_const_ref_result_direct_action;
*/

HPX_REGISTER_PLAIN_ACTION(non_const_ref_void_action);
/*
HPX_REGISTER_PLAIN_ACTION(non_const_ref_result_action);
HPX_REGISTER_PLAIN_ACTION(non_const_ref_void_direct_action);
HPX_REGISTER_PLAIN_ACTION(non_const_ref_result_direct_action);
*/

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    std::vector<id_type> localities = hpx::find_all_localities();

    int i = 9;

    BOOST_FOREACH(id_type id, localities)
    {
        // testing plain actions
        /*
        async<non_const_ref_void_action>(id, i).get();
        async<non_const_ref_void_action>(id, boost::move(i)).get();
        async<non_const_ref_void_action>(id, 9).get();
        async<non_const_ref_void_direct_action>(id, i).get();
        // Not possible, can't bind a rvalue reference directly to a non-const lvalue
        //async<non_const_ref_void_direct_action>(id, boost::move(i)).get();
        // Not possible (passing a rvalue directly to a non const lvalue ref)
        //async<non_const_ref_void_direct_action>(id, 9).get();
        
        HPX_TEST_EQ(async<non_const_ref_result_action>(id, i).get(), i);
        HPX_TEST_EQ(async<non_const_ref_result_action>(id, boost::move(i)).get(), i);
        HPX_TEST_EQ(async<non_const_ref_result_action>(id, 9).get(), 9);
        HPX_TEST_EQ(async<non_const_ref_result_direct_action>(id, i).get(), i);
        // Not possible, can't bind a rvalue reference directly to a non-const lvalue
        //HPX_TEST_EQ(async<non_const_ref_result_direct_action>(id, boost::move(i)).get(), i);
        // Not possible (passing a rvalue directly to a non const lvalue ref)
        //HPX_TEST_EQ(async<non_const_ref_void_direct_action>(id, 9).get(), 9);
        */
        
        // testing component actions
        /*
        using hpx::test::server::non_const_ref_component;
        async<non_const_ref_component::non_const_ref_void_action>(id, i).get();
        async<non_const_ref_component::non_const_ref_void_action>(id, boost::move(i)).get();
        async<non_const_ref_component::non_const_ref_void_action>(id, 9).get();
        async<non_const_ref_component::non_const_ref_void_direct_action>(id, i).get();
        // Not possible, can't bind a rvalue reference directly to a non-const lvalue
        //async<non_const_ref_component::non_const_ref_void_direct_action>(id, boost::move(i)).get();
        // Not possible (passing a rvalue directly to a non const lvalue ref)
        //async<non_const_ref_component::non_const_ref_void_direct_action>(id, 9).get();
        
        HPX_TEST_EQ(async<non_const_ref_component::non_const_ref_result_action>(id, i).get(), i);
        HPX_TEST_EQ(async<non_const_ref_component::non_const_ref_result_action>(id, boost::move(i)).get(), i);
        HPX_TEST_EQ(async<non_const_ref_component::non_const_ref_result_action>(id, 9).get(), 9);
        HPX_TEST_EQ(async<non_const_ref_component::non_const_ref_result_direct_action>(id, i).get(), i);
        // Not possible, can't bind a rvalue reference directly to a non-const lvalue
        //HPX_TEST_EQ(async<non_const_ref_component::non_const_ref_result_direct_action>(id, boost::move(i)).get(), i);
        // Not possible (passing a rvalue directly to a non const lvalue ref)
        //HPX_TEST_EQ(async<non_const_ref_component::non_const_ref_void_direct_action>(id, 9).get(), 9);
        */
    }

    finalize();

    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");
    
    // we need to explicitly enable the test components used by this test
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.components.test_non_const_ref_component.enabled = 1";

    // Initialize and run HPX.
    return init(desc_commandline, argc, argv);
}

