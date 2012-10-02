//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/foreach.hpp>
#include <boost/assign/std/vector.hpp>

#include <tests/regressions/actions/components/action_move_semantics.hpp>
#include <tests/regressions/actions/components/movable_objects.hpp>

using hpx::test::movable_object;
using hpx::test::non_movable_object;

///////////////////////////////////////////////////////////////////////////////
template <typename Action, typename Object>
std::size_t pass_object(hpx::naming::id_type id)
{
    using hpx::test::action_move_semantics;

    action_move_semantics test;
    test.create(id);

    Object obj;
    obj.reset_count();

    return hpx::async<Action>(test.get_gid(), obj).get();
}

///////////////////////////////////////////////////////////////////////////////
template <typename Action, typename Object>
std::size_t move_object(hpx::naming::id_type id)
{
    using hpx::test::action_move_semantics;

    action_move_semantics test; 
    test.create(id);

    Object obj;
    obj.reset_count();

    return hpx::async<Action>(test.get_gid(), boost::move(obj)).get();
}

///////////////////////////////////////////////////////////////////////////////
template <typename Action, typename Object>
std::size_t return_object(hpx::naming::id_type id)
{
    using hpx::test::action_move_semantics;

    action_move_semantics test;
    test.create(id);

    Object obj(hpx::async<Action>(test.get_gid()).get());
    return obj.get_count();
}

///////////////////////////////////////////////////////////////////////////////
template <typename Action, typename Object>
std::size_t return_move_object(hpx::naming::id_type id)
{
    using hpx::test::action_move_semantics;

    action_move_semantics test;
    test.create(id);

    Object obj(boost::move(hpx::async<Action>(test.get_gid()).move_out()));
    return obj.get_count();
}


int hpx_main(boost::program_options::variables_map&)
{
    using hpx::test::server::action_move_semantics;

    std::vector<hpx::naming::id_type> localities = hpx::find_all_localities();
    BOOST_FOREACH(hpx::naming::id_type id, localities)
    {
        bool is_local = (id == hpx::find_here()) ? true : false;

        // test for movable object ('normal' actions)
        HPX_TEST_EQ((
            pass_object<
                action_move_semantics::test_movable_action, movable_object
            >(id)
        ), is_local ? 1u : 1u);

        // test for movable object (direct actions)
        HPX_TEST_EQ((
            pass_object<
                action_move_semantics::test_movable_direct_action, movable_object
            >(id)
        ), is_local ? 0u : 1u);

        // FIXME: Can we get down to one copy for non-movable objects as well?

        // test for a non-movable object ('normal' actions)
        HPX_TEST_EQ((
            pass_object<
                action_move_semantics::test_non_movable_action, non_movable_object
            >(id)
        ), is_local ? 2u : 4u);

        // test for a non-movable object (direct actions)
        HPX_TEST_EQ((
            pass_object<
                action_move_semantics::test_non_movable_direct_action, non_movable_object
            >(id)
        ), is_local ? 0u : 4u);

        // test for movable object ('normal' actions)
        HPX_TEST_EQ((
            move_object<
                action_move_semantics::test_movable_action, movable_object
            >(id)
        ), is_local ? 0u : 0u);

        // test for movable object (direct actions)
        HPX_TEST_EQ((
            move_object<
                action_move_semantics::test_movable_direct_action, movable_object
            >(id)
        ), is_local ? 0u : 0u);

        // FIXME: Can we get down to one copy for non-movable objects as well?

        // test for a non-movable object ('normal' actions)
        HPX_TEST_EQ((
            move_object<
                action_move_semantics::test_non_movable_action, non_movable_object
            >(id)
        ), is_local ? 4u : 4u);

        // test for a non-movable object (direct actions)
        HPX_TEST_EQ((
            move_object<
                action_move_semantics::test_non_movable_direct_action, non_movable_object
            >(id)
        ), is_local ? 2u : 4u);
        
        HPX_TEST_EQ((
            return_object<
                action_move_semantics::return_test_movable_action, movable_object
            >(id)
        ), is_local ? 1u : 1u);

        HPX_TEST_EQ((
            return_object<
                action_move_semantics::return_test_movable_direct_action, movable_object
            >(id)
        ), is_local ? 1u : 1u);

        HPX_TEST_EQ((
            return_object<
                action_move_semantics::return_test_non_movable_action, non_movable_object
            >(id)
        ), is_local ? 5u : 7u);

        HPX_TEST_EQ((
            return_object<
                action_move_semantics::return_test_non_movable_direct_action, non_movable_object
            >(id)
        ), is_local ? 2u : 7u);
    }

    hpx::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    boost::program_options::options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // we need to explicitly enable the test components used by this test
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.components.action_move_semantics.enabled = 1";

    // Initialize and run HPX.
    return hpx::init(desc_commandline, argc, argv, cfg);
}

