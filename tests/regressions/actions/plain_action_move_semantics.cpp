////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2011-2013 Hartmut Kaiser
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/async.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/config/compiler_specific.hpp>

#include <cstddef>
#include <utility>
#include <vector>

#include <tests/regressions/actions/components/movable_objects.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;

using hpx::actions::action;
using hpx::actions::direct_action;

using hpx::async;

using hpx::init;
using hpx::finalize;
using hpx::find_here;

using hpx::test::movable_object;
using hpx::test::non_movable_object;

///////////////////////////////////////////////////////////////////////////////
void pass_movable_object_void(movable_object const&) {}
std::size_t pass_movable_object(movable_object const& obj)
{
    return obj.get_count();
}

// 'normal' actions (execution is scheduled on a new thread)
HPX_PLAIN_ACTION(pass_movable_object,
    pass_movable_object_action)
HPX_PLAIN_ACTION(pass_movable_object_void,
    pass_movable_object_void_action)

// direct actions (execution happens in the calling thread)
HPX_PLAIN_DIRECT_ACTION(pass_movable_object,
    pass_movable_object_direct_action)
HPX_PLAIN_DIRECT_ACTION(pass_movable_object_void,
    pass_movable_object_void_direct_action)

///////////////////////////////////////////////////////////////////////////////
void pass_movable_object_value_void(movable_object) {}
std::size_t pass_movable_object_value(movable_object obj)
{
    return obj.get_count();
}

// 'normal' actions (execution is scheduled on a new thread)
HPX_PLAIN_ACTION(pass_movable_object_value,
    pass_movable_object_value_action)
HPX_PLAIN_ACTION(pass_movable_object_value_void,
    pass_movable_object_value_void_action)

// direct actions (execution happens in the calling thread)
HPX_PLAIN_DIRECT_ACTION(pass_movable_object_value,
    pass_movable_object_value_direct_action)
HPX_PLAIN_DIRECT_ACTION(pass_movable_object_value_void,
    pass_movable_object_value_void_direct_action)

///////////////////////////////////////////////////////////////////////////////
void pass_non_movable_object_void(non_movable_object const&) {}
std::size_t pass_non_movable_object(non_movable_object const& obj)
{
    return obj.get_count();
}

// 'normal' actions (execution is scheduled on a new thread)
HPX_PLAIN_ACTION(pass_non_movable_object_void,
    pass_non_movable_object_void_action)
HPX_PLAIN_ACTION(pass_non_movable_object, pass_non_movable_object_action)

// direct actions (execution happens in the calling thread)
HPX_PLAIN_DIRECT_ACTION(pass_non_movable_object_void,
    pass_non_movable_object_void_direct_action)
HPX_PLAIN_DIRECT_ACTION(pass_non_movable_object,
    pass_non_movable_object_direct_action)

///////////////////////////////////////////////////////////////////////////////
void pass_non_movable_object_value_void(non_movable_object) {}
std::size_t pass_non_movable_object_value(non_movable_object obj)
{
    return obj.get_count();
}

// 'normal' actions (execution is scheduled on a new thread)
HPX_PLAIN_ACTION(pass_non_movable_object_value_void,
    pass_non_movable_object_value_void_action)
HPX_PLAIN_ACTION(pass_non_movable_object_value, pass_non_movable_object_value_action)

// direct actions (execution happens in the calling thread)
HPX_PLAIN_DIRECT_ACTION(pass_non_movable_object_value_void,
    pass_non_movable_object_value_void_direct_action)
HPX_PLAIN_DIRECT_ACTION(pass_non_movable_object_value,
    pass_non_movable_object_value_direct_action)

///////////////////////////////////////////////////////////////////////////////
non_movable_object return_non_movable_object()
{
    return non_movable_object();
}
movable_object return_movable_object()
{
    return movable_object();
}

// 'normal' actions (execution is scheduled on a new thread)
HPX_PLAIN_ACTION(return_movable_object,
    return_movable_object_action)
HPX_PLAIN_ACTION(return_non_movable_object,
    return_non_movable_object_action)

// direct actions (execution happens in the calling thread)
HPX_PLAIN_DIRECT_ACTION(return_movable_object,
    return_movable_object_direct_action)
HPX_PLAIN_DIRECT_ACTION(return_non_movable_object,
    return_non_movable_object_direct_action)

///////////////////////////////////////////////////////////////////////////////
template <typename Action, typename Object>
std::size_t pass_object_void()
{
    Object obj;
    async<Action>(find_here(), obj).get();

    return obj.get_count();
}

template <typename Action, typename Object>
std::size_t pass_object(id_type id)
{
    Object obj;
    return async<Action>(id, obj).get();
}

///////////////////////////////////////////////////////////////////////////////
template <typename Action, typename Object>
std::size_t move_object_void()
{
    Object obj;
    async<Action>(find_here(), std::move(obj)).get();

    return obj.get_count();
}

template <typename Action, typename Object>
std::size_t move_object(id_type id)
{
    Object obj;
    return async<Action>(id, std::move(obj)).get();
}

///////////////////////////////////////////////////////////////////////////////
template <typename Action, typename Object>
std::size_t return_object(id_type id)
{
    Object obj(async<Action>(id).get());
    return obj.get_count();
}

///////////////////////////////////////////////////////////////////////////////
void test_void_actions()
{
    // test the void actions locally only (there is no way to get the
    // overall copy count back)

    // test void(movable_object const&)
    {
        HPX_TEST_EQ((
            pass_object_void<
                pass_movable_object_void_action, movable_object
            >()
        ), 1u); // bind

        HPX_TEST_EQ((
            move_object_void<
                pass_movable_object_void_action, movable_object
            >()
        ), 0u);
    }

    // test void(non_movable_object const&)
    {
        HPX_TEST_EQ((
            pass_object_void<
                pass_non_movable_object_void_action, non_movable_object
            >()
        ), 2u); // bind + function

        HPX_TEST_EQ((
            move_object_void<
                pass_non_movable_object_void_action, non_movable_object
            >()
        ), 2u); // bind + function
    }

    // test void(movable_object)
    {
        HPX_TEST_EQ((
            pass_object_void<
                pass_movable_object_value_void_action, movable_object
            >()
        ), 1u); // call
        //! should be: -

        HPX_TEST_EQ((
            move_object_void<
                pass_movable_object_value_void_action, movable_object
            >()
        ), 0u);
    }

    // test void(non_movable_object)
    {
        HPX_TEST_EQ((
            pass_object_void<
                pass_non_movable_object_value_void_action, non_movable_object
            >()
        ), 3u); // bind + function + call

        HPX_TEST_EQ((
            move_object_void<
                pass_non_movable_object_value_void_action, non_movable_object
            >()
        ), 3u); // bind + function + call
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_void_direct_actions()
{
    // test the void actions locally only (there is no way to get the
    // overall copy count back)

    // test void(movable_object const&)
    {
        HPX_TEST_EQ((
            pass_object_void<
                pass_movable_object_void_direct_action, movable_object
            >()
        ), 0u);

        HPX_TEST_EQ((
            move_object_void<
                pass_movable_object_void_direct_action, movable_object
            >()
        ), 0u);
    }

    // test void(non_movable_object const&)
    {
        HPX_TEST_EQ((
            pass_object_void<
                pass_non_movable_object_void_direct_action, non_movable_object
            >()
        ), 0u);

        HPX_TEST_EQ((
            move_object_void<
                pass_non_movable_object_void_direct_action, non_movable_object
            >()
        ), 0u);
    }

    // test void(movable_object)
    {
        HPX_TEST_EQ((
            pass_object_void<
                pass_movable_object_value_void_direct_action, movable_object
            >()
        ), 1u); // call
        //! should be: -

        HPX_TEST_EQ((
            move_object_void<
                pass_movable_object_value_void_direct_action, movable_object
            >()
        ), 0u);
    }

    // test void(non_movable_object)
    {
        HPX_TEST_EQ((
            pass_object_void<
                pass_non_movable_object_value_void_direct_action, non_movable_object
            >()
        ), 1u); // call

        HPX_TEST_EQ((
            move_object_void<
                pass_non_movable_object_value_void_direct_action, non_movable_object
            >()
        ), 1u); // call
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_object_actions()
{
    std::vector<id_type> localities = hpx::find_all_localities();

    for (id_type const& id : localities)
    {
        bool is_local = id == find_here();

        // test size_t(movable_object const&)
        if (is_local)
        {
            HPX_TEST_EQ((
                pass_object<pass_movable_object_action, movable_object>(id)
            ), 1u); // bind

            HPX_TEST_EQ((
                move_object<pass_movable_object_action, movable_object>(id)
            ), 0U);
        } else {
            HPX_TEST_EQ((
                pass_object<pass_movable_object_action, movable_object>(id)
            ), 1u); // transfer_action

            HPX_TEST_EQ((
                move_object<pass_movable_object_action, movable_object>(id)
            ), 0u);
        }

        // test size_t(non_movable_object const&)
        if (is_local)
        {
            HPX_TEST_EQ((
                pass_object<pass_non_movable_object_action, non_movable_object>(id)
            ), 2u); // bind + function

            HPX_TEST_EQ((
                move_object<pass_non_movable_object_action, non_movable_object>(id)
            ), 2u); // bind + function
        } else {
            HPX_TEST_EQ((
                pass_object<pass_non_movable_object_action, non_movable_object>(id)
            ), 3u); // transfer_action + bind + function

            HPX_TEST_EQ((
                move_object<pass_non_movable_object_action, non_movable_object>(id)
            ), 3u); // transfer_action + bind + function
        }

        // test size_t(movable_object)
        if (is_local)
        {
            HPX_TEST_EQ((
                pass_object<pass_movable_object_value_action, movable_object>(id)
            ), 1u); // call

            HPX_TEST_EQ((
                move_object<pass_movable_object_value_action, movable_object>(id)
            ), 0u);
        } else {
            HPX_TEST_EQ((
                pass_object<pass_movable_object_value_action, movable_object>(id)
            ), 1u); // transfer_action

            HPX_TEST_EQ((
                move_object<pass_movable_object_value_action, movable_object>(id)
            ), 0u);
        }

        // test size_t(non_movable_object)
        if (is_local)
        {
            HPX_TEST_EQ((
                pass_object<pass_non_movable_object_value_action, non_movable_object>(id)
            ), 3u); // bind + function + call

            HPX_TEST_EQ((
                move_object<pass_non_movable_object_value_action, non_movable_object>(id)
            ), 3u); // bind + function + call
        } else {
            HPX_TEST_EQ((
                pass_object<pass_non_movable_object_value_action, non_movable_object>(id)
            ), 4u); // transfer_action + bind + function + call

            HPX_TEST_EQ((
                move_object<pass_non_movable_object_value_action, non_movable_object>(id)
            ), 4u); // transfer_action + bind + function + call
        }

        // test movable_object()
        if (is_local)
        {
            HPX_TEST_EQ((
                return_object<
                    return_movable_object_action, movable_object
                >(id)
            ), 0u);
        } else {
            HPX_TEST_EQ((
                return_object<
                    return_movable_object_action, movable_object
                >(id)
            ), 0u);
        }

        // test non_movable_object()
        if (is_local)
        {
            //FIXME: bumped number for intel compiler
            HPX_TEST_RANGE((
                return_object<
                    return_non_movable_object_action, non_movable_object
                >(id)
            ), 1u, 5u); // ?call + set_value + ?return
        } else {
            //FIXME: bumped number for intel compiler
            HPX_TEST_RANGE((
                return_object<
                    return_non_movable_object_action, non_movable_object
                >(id)
            ), 3u, 8u); // transfer_action + function + ?call +
                    // set_value + ?return
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_object_direct_actions()
{
    std::vector<id_type> localities = hpx::find_all_localities();

    for (id_type const& id : localities)
    {
        bool is_local = id == find_here();

        // test std::size_t(movable_object const&)
        if (is_local)
        {
            HPX_TEST_EQ((
                pass_object<pass_movable_object_direct_action, movable_object>(id)
            ), 0u);

            HPX_TEST_EQ((
                move_object<pass_movable_object_direct_action, movable_object>(id)
            ), 0u);
        } else {
            HPX_TEST_EQ((
                pass_object<pass_movable_object_direct_action, movable_object>(id)
            ), 1u); // transfer_action

            HPX_TEST_EQ((
                move_object<pass_movable_object_direct_action, movable_object>(id)
            ), 0u);
        }

        // test std::size_t(non_movable_object const&)
        if (is_local)
        {
            HPX_TEST_EQ((
                pass_object<pass_non_movable_object_direct_action,
                non_movable_object>(id)
            ), 0u);

            HPX_TEST_EQ((
                move_object<pass_non_movable_object_direct_action,
                non_movable_object>(id)
            ), 0u);
        } else {
            HPX_TEST_EQ((
                pass_object<pass_non_movable_object_direct_action,
                non_movable_object>(id)
            ), 1u); // transfer_action

            HPX_TEST_EQ((
                move_object<pass_non_movable_object_direct_action,
                non_movable_object>(id)
            ), 1u); // transfer_action
        }

        // test std::size_t(movable_object)
        if (is_local)
        {
            HPX_TEST_EQ((
                pass_object<pass_movable_object_value_direct_action, movable_object>(id)
            ), 1u); // call

            HPX_TEST_EQ((
                move_object<pass_movable_object_value_direct_action, movable_object>(id)
            ), 0u);
        } else {
            HPX_TEST_EQ((
                pass_object<pass_movable_object_value_direct_action, movable_object>(id)
            ), 1u); // transfer_action

            HPX_TEST_EQ((
                move_object<pass_movable_object_value_direct_action, movable_object>(id)
            ), 0u);
        }

        // test std::size_t(non_movable_object)
        if (is_local)
        {
            HPX_TEST_EQ((
                pass_object<pass_non_movable_object_value_direct_action,
                non_movable_object>(id)
            ), 1u); // call

            HPX_TEST_EQ((
                move_object<pass_non_movable_object_value_direct_action,
                non_movable_object>(id)
            ), 1u); // call
        } else {
            HPX_TEST_EQ((
                pass_object<pass_non_movable_object_value_direct_action,
                non_movable_object>(id)
            ), 2u); // transfer_action + call

            HPX_TEST_EQ((
                move_object<pass_non_movable_object_value_direct_action,
                non_movable_object>(id)
            ), 2u); // transfer_action + call
        }

        // test movable_object()
        if (is_local)
        {
            HPX_TEST_EQ((
                return_object<
                    return_movable_object_direct_action, movable_object
                >(id)
            ), 0u);
        } else {
            HPX_TEST_EQ((
                return_object<
                    return_movable_object_direct_action, movable_object
                >(id)
            ), 0u);
        }

        // test non_movable_object()
        if (is_local)
        {
            HPX_TEST_RANGE((
                return_object<
                    return_non_movable_object_direct_action, non_movable_object
                >(id)
            ), 1u, 3u); // ?call + set_value + ?return
        } else {
            //FIXME: bumped number for intel compiler
            HPX_TEST_RANGE((
                return_object<
                    return_non_movable_object_direct_action, non_movable_object
                >(id)
            ), 3u, 8u); // transfer_action + function + ?call +
                    // set_value + ?return
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map&)
{
    test_void_actions();
    test_void_direct_actions();
    test_object_actions();
    test_object_direct_actions();

    finalize();

    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    return init(desc_commandline, argc, argv);
}
