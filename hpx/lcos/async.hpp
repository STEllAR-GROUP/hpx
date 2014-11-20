//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_LCOS_ASYNC_SEP_28_2011_0840AM)
#define HPX_LCOS_ASYNC_SEP_28_2011_0840AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits/component_type_is_compatible.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/lcos/packaged_action.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/async_fwd.hpp>

namespace hpx
{
    namespace detail
    {
        BOOST_FORCEINLINE bool has_async_policy(BOOST_SCOPED_ENUM(launch) policy)
        {
            return (static_cast<int>(policy) &
                static_cast<int>(launch::async_policies)) ? true : false;
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Action, typename Result>
        struct sync_local_invoke
        {
            template <typename ...Ts>
            BOOST_FORCEINLINE static lcos::future<Result> call(
                naming::id_type const& gid, naming::address const&,
                Ts&&... vs)
            {
                lcos::packaged_action<Action, Result> p;
                p.apply(launch::sync, gid, std::forward<Ts>(vs)...);
                return p.get_future();
            }
        };

        template <typename Action, typename R>
        struct sync_local_invoke<Action, lcos::future<R> >
        {
            template <typename ...Ts>
            BOOST_FORCEINLINE static lcos::future<R> call(
                boost::mpl::true_, naming::id_type const&,
                naming::address const& addr, Ts&&... vs)
            {
                HPX_ASSERT(traits::component_type_is_compatible<
                    typename Action::component_type>::call(addr));
                return Action::execute_function(addr.address_,
                    util::forward_as_tuple(std::forward<Ts>(vs)...));
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
    template <typename Action, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async(BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid,
        Ts&&... vs)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::remote_result_type
        >::type result_type;

        naming::address addr;
        if (agas::is_local_address_cached(gid, addr) && policy == launch::sync)
        {
            return detail::sync_local_invoke<action_type, result_type>::
                call(gid, addr, std::forward<Ts>(vs)...);
        }

        lcos::packaged_action<action_type, result_type> p;

        bool target_is_managed = false;
        if (policy == launch::sync || detail::has_async_policy(policy))
        {
            if (addr) {
                p.apply(policy, std::move(addr), gid,
                    std::forward<Ts>(vs)...);
            }
            else if (gid.get_management_type() == naming::id_type::managed) {
                p.apply(policy,
                    naming::id_type(gid.get_gid(), naming::id_type::unmanaged),
                    std::forward<Ts>(vs)...);
                target_is_managed = true;
            }
            else {
                p.apply(policy, gid, std::forward<Ts>(vs)...);
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

    template <typename Action, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async(naming::id_type const& gid, Ts&&... vs)
    {
        return async<Action>(launch::all, gid, std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async(BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const & /*act*/, naming::id_type const& gid, Ts&&... vs)
    {
        return async<Derived>(policy, gid, std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /*act*/ const &, naming::id_type const& gid, Ts&&... vs)
    {
        return async<Derived>(launch::all, gid, std::forward<Ts>(vs)...);
    }
}

#endif
