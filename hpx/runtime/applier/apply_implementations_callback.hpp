//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_APPLIER_APPLY_IMPLEMENTATIONS_CALLBACK_DEC_17_2012_0240PM)
#define HPX_APPLIER_APPLY_IMPLEMENTATIONS_CALLBACK_DEC_17_2012_0240PM

#include <hpx/util/move.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/runtime/applier/preprocessed/apply_implementations_callback.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/apply_implementations_callback_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/applier/apply_implementations_callback.hpp"))                \
    /**/

#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_r_p_cb(naming::address& addr, naming::id_type const& id,
            threads::thread_priority priority, BOOST_FWD_REF(Callback) cb,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p(id, complement_addr<action_type>(addr),
                new hpx::actions::transfer_action<action_type>(
                    priority, HPX_ENUM_FORWARD_ARGS(N, Arg, arg)));

            // Send the parcel through the parcel handler
            hpx::applier::get_applier().get_parcel_handler()
                .put_parcel(p, boost::forward<Callback>(cb));
            return false;     // destinations are remote
        }

        template <typename Action, typename Callback,
            BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_r_cb(naming::address& addr, naming::id_type const& gid,
            BOOST_FWD_REF(Callback) cb, HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            return apply_r_p_cb<Action>(addr, gid,
                actions::action_priority<Action>(), boost::forward<Callback>(cb),
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply_p_cb(naming::id_type const& gid, threads::thread_priority priority,
        BOOST_FWD_REF(Callback) cb, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        if (!Action::is_target_valid(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb", 
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }

        // Determine whether the gid is local or remote
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            // apply locally
            bool result = applier::detail::apply_l_p<Action>(gid, addr, priority,
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
            cb(boost::system::error_code(), 0);     // invoke callback
            return result;
        }

        // apply remotely
        return applier::detail::apply_r_p_cb<Action>(addr, gid, priority,
            boost::forward<Callback>(cb), HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <typename Action, typename Callback,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply_cb(naming::id_type const& gid, BOOST_FWD_REF(Callback) cb,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return apply_p_cb<Action>(gid, actions::action_priority<Action>(),
            boost::forward<Callback>(cb), HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /*act*/,
        naming::id_type const& gid, BOOST_FWD_REF(Callback) cb,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return apply_p_cb<Derived>(gid, actions::action_priority<Derived>(),
            boost::forward<Callback>(cb), HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_r_p_cb(naming::address& addr, actions::continuation* c,
            naming::id_type const& id, threads::thread_priority priority,
            BOOST_FWD_REF(Callback) cb, HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            actions::continuation_type cont(c);

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p(id, complement_addr<action_type>(addr),
                new hpx::actions::transfer_action<action_type>(
                    priority, HPX_ENUM_FORWARD_ARGS(N, Arg, arg)), cont);

            // Send the parcel through the parcel handler
            hpx::applier::get_applier().get_parcel_handler()
              .put_parcel(p, boost::forward<Callback>(cb));
            return false;     // destination is remote
        }

        template <typename Action, typename Callback,
            BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_r_cb(naming::address& addr, actions::continuation* c,
            naming::id_type const& gid, BOOST_FWD_REF(Callback) cb,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            return apply_r_p_cb<Action>(addr, c, gid,
                actions::action_priority<Action>(), boost::forward<Callback>(cb),
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply_p_cb(actions::continuation* c, naming::address& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        BOOST_FWD_REF(Callback) cb, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        if (!Action::is_target_valid(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb", 
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }

        // Determine whether the gid is local or remote
        if (addr.locality_ == hpx::get_locality()) {
            // apply locally
            bool result = applier::detail::apply_l_p<Action>(c, gid, addr, priority,
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
            cb(boost::system::error_code(), 0);     // invoke callback
            return result;
        }

        // apply remotely
        return applier::detail::apply_r_p_cb<Action>(addr, c, gid, priority,
            boost::forward<Callback>(cb), HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <typename Action, typename Callback,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply_p_cb(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, BOOST_FWD_REF(Callback) cb,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        if (!Action::is_target_valid(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb", 
                "the target (destination) does not match the action type");
            return false;
        }

        // Determine whether the gid is local or remote
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            // apply locally
            bool result = applier::detail::apply_l_p<Action>(c, gid, addr, priority,
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
            cb(boost::system::error_code(), 0);     // invoke callback
            return result;
        }

        // apply remotely
        return applier::detail::apply_r_p_cb<Action>(addr, c, gid, priority,
            boost::forward<Callback>(cb), HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <typename Action, typename Callback,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply_cb(actions::continuation* c, naming::id_type const& gid,
        BOOST_FWD_REF(Callback) cb, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return apply_p_cb<Action>(c, gid, actions::action_priority<Action>(),
            boost::forward<Callback>(cb), HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply_cb(actions::continuation* c,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /*act*/,
        naming::id_type const& gid, BOOST_FWD_REF(Callback) cb,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>(),
            boost::forward<Callback>(cb), HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_c_p_cb(naming::address& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            BOOST_FWD_REF(Callback) cb, HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_p_cb<Action>(addr,
                new actions::typed_continuation<result_type>(contgid),
                gid, priority, boost::forward<Callback>(cb),
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }

        template <typename Action, typename Callback,
            BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_c_cb(naming::address& addr, naming::id_type const& contgid,
            naming::id_type const& gid, BOOST_FWD_REF(Callback) cb,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_p_cb<Action>(addr,
                new actions::typed_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                boost::forward<Callback>(cb),
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }
    }}

    template <typename Action, typename Callback,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, BOOST_FWD_REF(Callback) cb,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, priority, boost::forward<Callback>(cb),
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <typename Action, typename Callback,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::id_type const& gid,
        BOOST_FWD_REF(Callback) cb, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Action>(),
            boost::forward<Callback>(cb), HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <typename Action, typename Callback,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::address& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        BOOST_FWD_REF(Callback) cb, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            addr, gid, priority, boost::forward<Callback>(cb),
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <typename Action, typename Callback,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::address& addr,
        naming::id_type const& gid, BOOST_FWD_REF(Callback) cb,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            addr, gid, actions::action_priority<Action>(),
            boost::forward<Callback>(cb), HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }
}

///////////////////////////////////////////////////////////////////////////////
#undef N

#endif
