////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2011-2012 Hartmut Kaiser
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#define HPX_LIMIT 6

#include <hpx/hpx_init.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/components/dataflow/dataflow.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/foreach.hpp>

#include <tests/regressions/actions/components/movable_objects.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;

using hpx::actions::plain_action1;
using hpx::actions::plain_result_action0;
using hpx::actions::plain_result_action1;
using hpx::actions::plain_direct_action1;
using hpx::actions::plain_direct_result_action0;
using hpx::actions::plain_direct_result_action1;

using hpx::lcos::dataflow;

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
typedef plain_action1<
    movable_object const&, pass_movable_object_void
> pass_movable_object_void_action;
typedef plain_result_action1<
    std::size_t, movable_object const&, pass_movable_object
> pass_movable_object_action;

HPX_REGISTER_PLAIN_ACTION(pass_movable_object_void_action)
HPX_REGISTER_PLAIN_ACTION(pass_movable_object_action)

// direct actions (execution happens in the calling thread)
typedef plain_direct_action1<
    movable_object const&, pass_movable_object_void
> pass_movable_object_void_direct_action;
typedef plain_direct_result_action1<
    std::size_t, movable_object const&, pass_movable_object
> pass_movable_object_direct_action;

HPX_REGISTER_PLAIN_ACTION(pass_movable_object_void_direct_action)
HPX_REGISTER_PLAIN_ACTION(pass_movable_object_direct_action)

///////////////////////////////////////////////////////////////////////////////
void pass_non_movable_object_void(non_movable_object const&) {}
std::size_t pass_non_movable_object(non_movable_object const& obj)
{
    return obj.get_count();
}

// 'normal' actions (execution is scheduled on a new thread
typedef plain_action1<
    non_movable_object const&, pass_non_movable_object_void
> pass_non_movable_object_void_action;
typedef plain_result_action1<
    std::size_t, non_movable_object const&, pass_non_movable_object
> pass_non_movable_object_action;

HPX_REGISTER_PLAIN_ACTION(pass_non_movable_object_void_action)
HPX_REGISTER_PLAIN_ACTION(pass_non_movable_object_action)

// direct actions (execution happens in the calling thread)
typedef plain_direct_action1<
    non_movable_object const&, pass_non_movable_object_void
> pass_non_movable_object_void_direct_action;
typedef plain_direct_result_action1<
    std::size_t, non_movable_object const&, pass_non_movable_object
> pass_non_movable_object_direct_action;

HPX_REGISTER_PLAIN_ACTION(pass_non_movable_object_void_direct_action)
HPX_REGISTER_PLAIN_ACTION(pass_non_movable_object_direct_action)

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
typedef plain_result_action0<
    movable_object
  , return_movable_object
> return_movable_object_action;

typedef plain_result_action0<
    non_movable_object
  , return_non_movable_object
> return_non_movable_object_action;

HPX_REGISTER_PLAIN_ACTION(return_movable_object_action)
HPX_REGISTER_PLAIN_ACTION(return_non_movable_object_action)

// direct actions (execution happens in the calling thread)
typedef plain_direct_result_action0<
    movable_object
  , return_movable_object
> return_movable_object_direct_action;

typedef plain_direct_result_action0<
    non_movable_object
  , return_non_movable_object
> return_non_movable_object_direct_action;

HPX_REGISTER_PLAIN_ACTION(return_movable_object_direct_action)
HPX_REGISTER_PLAIN_ACTION(return_non_movable_object_direct_action)

///////////////////////////////////////////////////////////////////////////////
template <typename Action, typename Object>
std::size_t pass_object_void()
{
    Object obj;
    dataflow<Action>(find_here(), obj).get_future().get();

    return obj.get_count();
}

template <typename Action, typename Object>
std::size_t pass_object(id_type id)
{
    Object obj;
    return dataflow<Action>(id, obj).get_future().get();
}

///////////////////////////////////////////////////////////////////////////////
template <typename Action, typename Object>
std::size_t move_object_void()
{
    Object obj;
    dataflow<Action>(find_here(), boost::move(obj)).get_future().get();

    return obj.get_count();
}

template <typename Action, typename Object>
std::size_t move_object(id_type id)
{
    Object obj;
    return dataflow<Action>(id, boost::move(obj)).get_future().get();
}

///////////////////////////////////////////////////////////////////////////////
template <typename Action, typename Object>
std::size_t return_object(id_type id)
{
    Object obj(dataflow<Action>(id).get_future().get());
    return obj.get_count();
}

///////////////////////////////////////////////////////////////////////////////
template <typename Action, typename Object>
std::size_t return_move_object(id_type id)
{
    Object obj(dataflow<Action>(id).get_future().move());
    return obj.get_count();
}


///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map&)
{
    std::vector<id_type> localities = hpx::find_all_localities();

    BOOST_FOREACH(id_type id, localities)
    {
        bool is_local = (id == find_here()) ? true : false;

        if (is_local) {
            // test the void actions locally only (there is no way to get the
            // overall copy count back)

            HPX_TEST_EQ((
                pass_object_void<
                    pass_movable_object_void_action, movable_object
                >()
            ), 1u);

            /* TODO: Make this compile
            HPX_TEST_EQ((
                pass_object_void<
                    pass_movable_object_void_direct_action, movable_object
                >()
            ), 0u);
            */

            HPX_TEST_EQ((
                pass_object_void<
                    pass_non_movable_object_void_action, non_movable_object
                >()
            ), 5u);

            /* TODO: Make this compile
            HPX_TEST_EQ((
                pass_object_void<
                    pass_non_movable_object_void_direct_action, non_movable_object
                >()
            ), 0u);
            */

            HPX_TEST_EQ((
                move_object_void<
                    pass_movable_object_void_action, movable_object
                >()
            ), 1u);

            /* TODO: Make this compile
            HPX_TEST_EQ((
                move_object_void<
                    pass_movable_object_void_direct_action, movable_object
                >()
            ), 0u);
            */

            HPX_TEST_EQ((
                move_object_void<
                    pass_non_movable_object_void_action, non_movable_object
                >()
            ), 5u);
            
            /* TODO: Make this compile
            HPX_TEST_EQ((
                move_object_void<
                    pass_non_movable_object_void_direct_action, non_movable_object
                >()
            ), 0u);
            */
        }

        // test for movable object ('normal' actions)
        HPX_TEST_EQ((
            pass_object<pass_movable_object_action, movable_object>(id)
        ), is_local ? 1u : 1u);

        /* TODO: Make this compile
        // test for movable object (direct actions)
        HPX_TEST_EQ((
            pass_object<pass_movable_object_direct_action, movable_object>(id)
        ), is_local ? 0u : 0u);
        */

        // test for a non-movable object ('normal' actions)
        HPX_TEST_EQ((
            pass_object<pass_non_movable_object_action, non_movable_object>(id)
        ), is_local ? 5u : 5u);

        /* TODO: Make this compile
        // test for a non-movable object (direct actions)
        HPX_TEST_EQ((
            pass_object<pass_non_movable_object_direct_action, non_movable_object>(id)
        ), is_local ? 0u : 0u);
        */

        // test for movable object ('normal' actions)
        HPX_TEST_EQ((
            move_object<pass_movable_object_action, movable_object>(id)
        ), is_local ? 1u : 1u);

        /* TODO: Make this compile
        // test for movable object (direct actions)
        HPX_TEST_EQ((
            move_object<pass_movable_object_direct_action, movable_object>(id)
        ), is_local ? 0u : 0u);
        */

        // test for a non-movable object ('normal' actions)
        HPX_TEST_EQ((
            move_object<pass_non_movable_object_action, non_movable_object>(id)
        ), is_local ? 5u : 5u);

        /* TODO: Make this compile
        // test for a non-movable object (direct actions)
        HPX_TEST_EQ((
            move_object<pass_non_movable_object_direct_action, non_movable_object>(id)
        ), is_local ? 0u : 0u);
        */

        HPX_TEST_EQ((
            return_object<
                return_movable_object_action, movable_object
            >(id)
        ), is_local ? 1u : 3u);

        /* TODO: Make this compile
        HPX_TEST_EQ((
            return_object<
                return_movable_object_direct_action, movable_object
            >(id)
        ), is_local ? 1u : 1u);
        */

        HPX_TEST_EQ((
            return_object<
                return_non_movable_object_action, non_movable_object
            >(id)
        ), is_local ? 2u : 9u);

        /* TODO: Make this compile
        HPX_TEST_EQ((
            return_object<
                return_non_movable_object_direct_action, non_movable_object
            >(id)
        ), is_local ? 2u : 7u);
        */
            
#if defined(__GNUC__) && (HPX_GCC_VERSION < 40500)
        HPX_TEST_EQ((
            return_move_object<
                return_movable_object_action, movable_object
            >(id)
        ), is_local ? 2u : 2u);
#else
        HPX_TEST_EQ((
            return_move_object<
                return_movable_object_action, movable_object
            >(id)
        ), is_local ? 0u : 2u);
#endif

        /* TODO: Make this compile
        HPX_TEST_EQ((
            return_move_object<
                return_movable_object_direct_action, movable_object
            >(id)
        ), is_local ? 0u : 0u);
        */

#if defined(__GNUC__) && (HPX_GCC_VERSION < 40500)
        HPX_TEST_EQ((
            return_move_object<
                return_non_movable_object_action, non_movable_object
            >(id)
        ), 
        is_local ? 3u : 10u);      // gcc V4.4 is special
#else
        HPX_TEST_EQ((
            return_move_object<
                return_non_movable_object_action, non_movable_object
            >(id)
        ), 
        is_local ? 2u : 9u);
#endif

        /* TODO: Make this compile
        HPX_TEST_EQ((
            return_move_object<
                return_non_movable_object_direct_action, non_movable_object
            >(id)
        ), is_local ? 2u : 7u);
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

    // Initialize and run HPX.
    return init(desc_commandline, argc, argv);
}

