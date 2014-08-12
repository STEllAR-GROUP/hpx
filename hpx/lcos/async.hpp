//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_ASYNC_SEP_28_2011_0840AM)
#define HPX_LCOS_ASYNC_SEP_28_2011_0840AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits/component_type_is_compatible.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/lcos/packaged_action.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/async_fwd.hpp>

#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    namespace detail
    {
        BOOST_FORCEINLINE bool has_async_policy(BOOST_SCOPED_ENUM(launch) policy)
        {
            return (static_cast<int>(policy) &
                static_cast<int>(launch::async_policies)) ? true : false;
        }

        template <typename Action, typename Result>
        struct sync_local_invoke_0
        {
            BOOST_FORCEINLINE static lcos::future<Result> call(
                naming::id_type const& gid, naming::address const& /*addr*/)
            {
                lcos::packaged_action<Action, Result> p;
                p.apply(launch::sync, gid);
                return p.get_future();
            }
        };

        template <typename Action, typename R>
        struct sync_local_invoke_0<Action, lcos::future<R> >
        {
            BOOST_FORCEINLINE static lcos::future<R> call(
                naming::id_type const& /*gid*/, naming::address const& addr)
            {
                HPX_ASSERT(traits::component_type_is_compatible<
                    typename Action::component_type>::call(addr));
                return Action::execute_function(addr.address_,
                    util::forward_as_tuple());
            }
        };

        ///////////////////////////////////////////////////////////////////////
        struct keep_id_alive
        {
            explicit keep_id_alive(naming::id_type const& gid)
              : gid_(gid)
            {}

            void operator()() const {}

            naming::id_type gid_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async(BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::remote_result_type
        >::type result_type;

        naming::address addr;
        if (agas::is_local_address_cached(gid, addr) && policy == launch::sync)
        {
            return detail::sync_local_invoke_0<action_type, result_type>::
                call(gid, addr);
        }

        lcos::packaged_action<action_type, result_type> p;

        bool target_is_managed = false;
        if (policy == launch::sync || detail::has_async_policy(policy))
        {
            if (addr) {
                p.apply(policy, std::move(addr), gid);
            }
            else if (gid.get_management_type() == naming::id_type::managed) {
                p.apply(policy,
                    naming::id_type(gid.get_gid(), naming::id_type::unmanaged));
                target_is_managed = true;
            }
            else {
                p.apply(policy, gid);
            }
        }

        // keep id alive, if needed - this allows to send the destination as an
        // unmanaged id
        future<result_type> f = p.get_future();

        if (target_is_managed)
        {
            typedef typename lcos::detail::shared_state_ptr_for<
                future<result_type>
            >::type shared_state_ptr;

            shared_state_ptr const& state = lcos::detail::get_shared_state(f);
            state->set_on_completed(detail::keep_id_alive(gid));
        }

        return std::move(f);
    }

    template <typename Action>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async(naming::id_type const& gid)
    {
        return async<Action>(launch::all, gid);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Result,
        typename Arguments, typename Derived>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async(BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /*act*/, naming::id_type const& gid)
    {
        return async<Derived>(policy, gid);
    }

    template <typename Component, typename Result,
        typename Arguments, typename Derived>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const & /*act*/, naming::id_type const& gid)
    {
        return async<Derived>(launch::all, gid);
    }
}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/async.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/async_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/lcos/async.hpp"))                                                    \
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
    namespace detail
    {
        template <typename Action, typename Result>
        struct BOOST_PP_CAT(sync_local_invoke_, N)
        {
            template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
            BOOST_FORCEINLINE static lcos::future<Result> call(
                naming::id_type const& gid, naming::address const&,
                HPX_ENUM_FWD_ARGS(N, Arg, arg))
            {
                lcos::packaged_action<Action, Result> p;
                p.apply(launch::sync, gid, HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
                return p.get_future();
            }
        };

        template <typename Action, typename R>
        struct BOOST_PP_CAT(sync_local_invoke_, N)<Action, lcos::future<R> >
        {
            template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
            BOOST_FORCEINLINE static lcos::future<R> call(
                boost::mpl::true_, naming::id_type const&,
                naming::address const& addr, HPX_ENUM_FWD_ARGS(N, Arg, arg))
            {
                HPX_ASSERT(traits::component_type_is_compatible<
                    typename Action::component_type>::call(addr));
                return Action::execute_function(addr.address_,
                    util::forward_as_tuple(HPX_ENUM_FORWARD_ARGS(N, Arg, arg)));
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async(BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::remote_result_type
        >::type result_type;

        naming::address addr;
        if (agas::is_local_address_cached(gid, addr) && policy == launch::sync)
        {
            return detail::BOOST_PP_CAT(sync_local_invoke_, N)<action_type, result_type>::
                call(gid, addr, HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }

        lcos::packaged_action<action_type, result_type> p;

        bool target_is_managed = false;
        if (policy == launch::sync || detail::has_async_policy(policy))
        {
            if (addr) {
                p.apply(policy, std::move(addr), gid,
                    HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
            }
            else if (gid.get_management_type() == naming::id_type::managed) {
                p.apply(policy,
                    naming::id_type(gid.get_gid(), naming::id_type::unmanaged),
                    HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
                target_is_managed = true;
            }
            else {
                p.apply(policy, gid, HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
            }
        }

        // keep id alive, if needed - this allows to send the destination as an
        // unmanaged id
        future<result_type> f = p.get_future();

        if (target_is_managed)
        {
            typedef typename lcos::detail::shared_state_ptr_for<
                future<result_type>
            >::type shared_state_ptr;

            shared_state_ptr const& state = lcos::detail::get_shared_state(f);
            state->set_on_completed(detail::keep_id_alive(gid));
        }

        return std::move(f);
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async(naming::id_type const& gid, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return async<Action>(launch::all, gid,
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Result,
        typename Arguments, typename Derived, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async(BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const & /*act*/, naming::id_type const& gid, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return async<Derived>(policy, gid,
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <typename Component, typename Result,
        typename Arguments, typename Derived, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /*act*/ const &, naming::id_type const& gid, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return async<Derived>(launch::all, gid,
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }
}

#undef N

#endif
