//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/config/compiler_specific.hpp>

#include <boost/assign/std/vector.hpp>

#include <string>
#include <utility>
#include <vector>

#include <tests/regressions/actions/components/action_move_semantics.hpp>
#include <tests/regressions/actions/components/movable_objects.hpp>

using hpx::test::movable_object;
using hpx::test::non_movable_object;

///////////////////////////////////////////////////////////////////////////////
template <typename Action, typename Object>
std::size_t pass_object(hpx::naming::id_type id)
{
    using hpx::test::action_move_semantics;

    action_move_semantics test = action_move_semantics::create(id);

    Object obj;
    obj.reset_count();

    return hpx::async<Action>(test.get_id(), obj).get();
}

///////////////////////////////////////////////////////////////////////////////
template <typename Action, typename Object>
std::size_t move_object(hpx::naming::id_type id)
{
    using hpx::test::action_move_semantics;

    action_move_semantics test = action_move_semantics::create(id);

    Object obj;
    obj.reset_count();

    return hpx::async<Action>(test.get_id(), std::move(obj)).get();
}

///////////////////////////////////////////////////////////////////////////////
template <typename Action, typename Object>
std::size_t return_object(hpx::naming::id_type id)
{
    using hpx::test::action_move_semantics;

    action_move_semantics test = action_move_semantics::create(id);

    Object obj(hpx::async<Action>(test.get_id()).get());
    return obj.get_count();
}

///////////////////////////////////////////////////////////////////////////////
template <typename Action, typename Object>
std::size_t return_move_object(hpx::naming::id_type id)
{
    using hpx::test::action_move_semantics;

    action_move_semantics test = action_move_semantics::create(id);

    Object obj(std::move(hpx::async<Action>(test.get_id()).move_out()));
    return obj.get_count();
}

///////////////////////////////////////////////////////////////////////////////
void test_actions()
{
    using hpx::test::server::action_move_semantics;

    std::vector<hpx::naming::id_type> localities = hpx::find_all_localities();
    for (hpx::naming::id_type const& id : localities)
    {
        bool is_local = id == hpx::find_here();

        // test std::size_t(movable_object const& obj)
        if (is_local)
        {
            HPX_TEST_EQ((
                pass_object<
                    action_move_semantics::test_movable_action, movable_object
                >(id)
            ), 1u); // bind

            HPX_TEST_EQ((
                move_object<
                    action_move_semantics::test_movable_action, movable_object
                >(id)
            ), 0u);
        } else {
            HPX_TEST_EQ((
                pass_object<
                    action_move_semantics::test_movable_action, movable_object
                >(id)
            ), 1u); // transfer_action

            HPX_TEST_EQ((
                move_object<
                    action_move_semantics::test_movable_action, movable_object
                >(id)
            ), 0u);
        }

        // test std::size_t(non_movable_object const& obj)
        if (is_local)
        {
            HPX_TEST_EQ((
                pass_object<
                    action_move_semantics::test_non_movable_action, non_movable_object
                >(id)
            ), 2u); // bind + function

            HPX_TEST_EQ((
                move_object<
                    action_move_semantics::test_non_movable_action, non_movable_object
                >(id)
            ), 2u); // bind + function
        } else {
            HPX_TEST_EQ((
                pass_object<
                    action_move_semantics::test_non_movable_action, non_movable_object
                >(id)
            ), 3u); // transfer_action + bind + function

            HPX_TEST_EQ((
                move_object<
                    action_move_semantics::test_non_movable_action, non_movable_object
                >(id)
            ), 3u); // transfer_action + bind + function
        }

        // test movable_object()
        if (is_local)
        {
            HPX_TEST_EQ((
                return_object<
                    action_move_semantics::return_test_movable_action, movable_object
                >(id)
            ), 0u);
        } else {
            HPX_TEST_EQ((
                return_object<
                    action_move_semantics::return_test_movable_action, movable_object
                >(id)
            ), 0u);
        }

        // test non_movable_object()
        if (is_local)
        {
            //FIXME: bumped number for intel compiler
            HPX_TEST_RANGE((
                return_object<
                    action_move_semantics::return_test_non_movable_action,
                non_movable_object
                >(id)
            ), 1u, 5u); // ?call + set_value + ?return
        } else {
            //FIXME: bumped number for intel compiler
            HPX_TEST_RANGE((
                return_object<
                    action_move_semantics::return_test_non_movable_action,
                non_movable_object
                >(id)
            ), 4u, 8u); // transfer_action + bind + function + ?call +
                    // set_value + ?return
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_direct_actions()
{
    using hpx::test::server::action_move_semantics;

    std::vector<hpx::naming::id_type> localities = hpx::find_all_localities();
    for (hpx::naming::id_type const& id : localities)
    {
        bool is_local = id == hpx::find_here();

        // test std::size_t(movable_object const& obj)
        if (is_local)
        {
            HPX_TEST_EQ((
                pass_object<
                    action_move_semantics::test_movable_direct_action, movable_object
                >(id)
            ), 0u);

            HPX_TEST_EQ((
                move_object<
                    action_move_semantics::test_movable_direct_action, movable_object
                >(id)
            ), 0u);
        } else {
            HPX_TEST_EQ((
                pass_object<
                    action_move_semantics::test_movable_direct_action, movable_object
                >(id)
            ), 1u); // transfer_action

            HPX_TEST_EQ((
                move_object<
                    action_move_semantics::test_movable_direct_action, movable_object
                >(id)
            ), 0u);
        }

        // test std::size_t(non_movable_object const& obj)
        if (is_local)
        {
            HPX_TEST_EQ((
                pass_object<
                    action_move_semantics::test_non_movable_direct_action,
                non_movable_object
                >(id)
            ), 0u);

            HPX_TEST_EQ((
                move_object<
                    action_move_semantics::test_non_movable_direct_action,
                non_movable_object
                >(id)
            ), 0u);
        } else {
            HPX_TEST_EQ((
                pass_object<
                    action_move_semantics::test_non_movable_direct_action,
                non_movable_object
                >(id)
            ), 3u); // transfer_action + bind + function

            HPX_TEST_EQ((
                move_object<
                    action_move_semantics::test_non_movable_direct_action,
                non_movable_object
                >(id)
            ), 3u); // transfer_action + bind + function
        }

        // test movable_object()
        if (is_local)
        {
            HPX_TEST_EQ((
                return_object<
                    action_move_semantics::return_test_movable_direct_action,
                movable_object
                >(id)
            ), 0u);
        } else {
            HPX_TEST_EQ((
                return_object<
                    action_move_semantics::return_test_movable_direct_action,
                movable_object
                >(id)
            ), 0u);
        }

        // test non_movable_object()
        if (is_local)
        {
            HPX_TEST_RANGE((
                return_object<
                    action_move_semantics::return_test_non_movable_direct_action,
                non_movable_object
                >(id)
            ), 1u, 3u); // ?call + set_value + ?return
        } else {
            //FIXME: bumped number for intel compiler
            HPX_TEST_RANGE((
                return_object<
                    action_move_semantics::return_test_non_movable_direct_action,
                non_movable_object
                >(id)
            ), 4u, 8u); // transfer_action + bind + function + ?call +
                    // set_value + ?return
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map&)
{
    test_actions();
    test_direct_actions();

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
    cfg += "hpx.components.action_move_semantics.enabled! = 1";

    // Initialize and run HPX.
    return hpx::init(desc_commandline, argc, argv, cfg);
}
