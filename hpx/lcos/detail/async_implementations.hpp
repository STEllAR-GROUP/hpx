//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ASYNC_IMPLEMENTATIONS_APR_13_2015_0829AM)
#define HPX_LCOS_ASYNC_IMPLEMENTATIONS_APR_13_2015_0829AM

#include <hpx/config.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/lcos/detail/async_implementations_fwd.hpp>
#include <hpx/lcos/packaged_action.hpp>
#include <hpx/util/move.hpp>

namespace hpx { namespace detail
{
    /// \cond NOINTERNAL
    BOOST_FORCEINLINE bool has_async_policy(BOOST_SCOPED_ENUM(launch) policy)
    {
        return (static_cast<int>(policy) &
            static_cast<int>(launch::async_policies)) ? true : false;
    }

    ///////////////////////////////////////////////////////////////////////////
    struct keep_id_alive
    {
        explicit keep_id_alive(naming::id_type const& gid)
            : gid_(gid)
        {}

        void operator()() const {}

        naming::id_type gid_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Result>
    struct sync_local_invoke
    {
        template <typename ...Ts>
        static lcos::future<Result> call(
            naming::id_type const& id, naming::address && addr,
            Ts &&... vs)
        {
            lcos::packaged_action<Action, Result> p;
            bool target_is_managed = false;

            if (id.get_management_type() == naming::id_type::managed)
            {
                naming::id_type id1(id.get_gid(), naming::id_type::unmanaged);
                if (addr)
                {
                    p.apply(launch::sync, std::move(addr), id1,
                        std::forward<Ts>(vs)...);
                }
                else
                {
                    p.apply(launch::sync, id1, std::forward<Ts>(vs)...);
                }
                target_is_managed = true;
            }
            else
            {
                p.apply(launch::sync, id, std::forward<Ts>(vs)...);
            }

            // keep id alive, if needed - this allows to send the destination
            // as an unmanaged id
            future<Result> f = p.get_future();

            if (target_is_managed)
            {
                typedef typename traits::detail::shared_state_ptr_for<
                    future<Result>
                >::type shared_state_ptr;

                shared_state_ptr const& state = traits::detail::get_shared_state(f);
                state->set_on_completed(hpx::detail::keep_id_alive(id));
            }

            return f;
        }
    };

    template <typename Action, typename R>
    struct sync_local_invoke<Action, lcos::future<R> >
    {
        template <typename ...Ts>
        BOOST_FORCEINLINE static lcos::future<R> call(
            boost::mpl::true_, naming::id_type const&,
            naming::address && addr, Ts &&... vs)
        {
            HPX_ASSERT(traits::component_type_is_compatible<
                typename Action::component_type>::call(addr));

            return Action::execute_function(addr.address_,
                std::forward<Ts>(vs)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename Action, typename Result>
    struct sync_local_invoke_cb
    {
        template <typename Callback, typename ...Ts>
        static lcos::future<Result> call(
            naming::id_type const& id, naming::address && addr,
            Callback && cb, Ts &&... vs)
        {
            lcos::packaged_action<Action, Result> p;
            bool target_is_managed = false;

            if (id.get_management_type() == naming::id_type::managed)
            {
                naming::id_type id1(id.get_gid(), naming::id_type::unmanaged);
                if (addr)
                {
                    p.apply_cb(launch::sync, std::move(addr), id1,
                        std::forward<Callback>(cb), std::forward<Ts>(vs)...);
                }
                else
                {
                    p.apply_cb(launch::sync, id1, std::forward<Callback>(cb),
                        std::forward<Ts>(vs)...);
                }
                target_is_managed = true;
            }
            else
            {
                p.apply_cb(launch::sync, id, std::forward<Callback>(cb),
                    std::forward<Ts>(vs)...);
            }

            // keep id alive, if needed - this allows to send the destination
            // as an unmanaged id
            future<Result> f = p.get_future();

            if (target_is_managed)
            {
                typedef typename traits::detail::shared_state_ptr_for<
                    future<Result>
                >::type shared_state_ptr;

                shared_state_ptr const& state = traits::detail::get_shared_state(f);
                state->set_on_completed(hpx::detail::keep_id_alive(id));
            }

            return f;
        }
    };

    template <typename Action, typename R>
    struct sync_local_invoke_cb<Action, lcos::future<R> >
    {
        template <typename Callback, typename ...Ts>
        BOOST_FORCEINLINE static lcos::future<R> call(
            boost::mpl::true_, naming::id_type const&,
            naming::address && addr, Callback && cb, Ts&&... vs)
        {
            HPX_ASSERT(traits::component_type_is_compatible<
                typename Action::component_type>::call(addr));

            lcos::future<R> f = Action::execute_function(addr.address_,
                std::forward<Ts>(vs)...);

            // invoke callback
            cb(boost::system::error_code(), parcelset::parcel());

            return f;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename ...Ts>
    hpx::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async_impl(BOOST_SCOPED_ENUM(launch) policy, hpx::id_type const& id,
        Ts&&... vs)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::remote_result_type
        >::type result_type;

        naming::address addr;
        if (agas::is_local_address_cached(id, addr) && policy == launch::sync)
        {
            return hpx::detail::sync_local_invoke<action_type, result_type>::
                call(id, std::move(addr), std::forward<Ts>(vs)...);
        }

        lcos::packaged_action<action_type, result_type> p;

        bool target_is_managed = false;
        if (policy == launch::sync || hpx::detail::has_async_policy(policy))
        {
            if (id.get_management_type() == naming::id_type::managed)
            {
                naming::id_type id1(id.get_gid(), naming::id_type::unmanaged);
                if (addr)
                {
                    p.apply(policy, std::move(addr), id1,
                        std::forward<Ts>(vs)...);
                }
                else
                {
                    p.apply(policy, id1, std::forward<Ts>(vs)...);
                }
                target_is_managed = true;
            }
            else
            {
                p.apply(policy, id, std::forward<Ts>(vs)...);
            }
        }

        // keep id alive, if needed - this allows to send the destination as an
        // unmanaged id
        future<result_type> f = p.get_future();

        if (target_is_managed)
        {
            typedef typename traits::detail::shared_state_ptr_for<
                future<result_type>
            >::type shared_state_ptr;

            shared_state_ptr const& state = traits::detail::get_shared_state(f);
            state->set_on_completed(hpx::detail::keep_id_alive(id));
        }

        return f;
    }

    /// \note This function is part of the invocation policy implemented by
    ///       this class
    ///
    template <typename Action, typename Callback, typename ...Ts>
    hpx::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async_cb_impl(BOOST_SCOPED_ENUM(launch) policy, hpx::id_type const& id,
        Callback&& cb, Ts&&... vs)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::remote_result_type
        >::type result_type;

        naming::address addr;
        if (agas::is_local_address_cached(id, addr) && policy == launch::sync)
        {
            return hpx::detail::sync_local_invoke_cb<action_type, result_type>::
                call(id, std::move(addr), std::forward<Callback>(cb),
                    std::forward<Ts>(vs)...);
        }

        lcos::packaged_action<action_type, result_type> p;

        bool target_is_managed = false;
        if (policy == launch::sync || hpx::detail::has_async_policy(policy))
        {
            if (id.get_management_type() == naming::id_type::managed)
            {
                naming::id_type id1(id.get_gid(), naming::id_type::unmanaged);
                if (addr)
                {
                    p.apply_cb(policy, std::move(addr), id1,
                        std::forward<Callback>(cb), std::forward<Ts>(vs)...);
                }
                else
                {
                    p.apply_cb(policy, id1, std::forward<Callback>(cb),
                        std::forward<Ts>(vs)...);
                }
                target_is_managed = true;
            }
            else
            {
                p.apply_cb(policy, id, std::forward<Callback>(cb),
                    std::forward<Ts>(vs)...);
            }
        }

        // keep id alive, if needed - this allows to send the destination
        // as an unmanaged id
        future<result_type> f = p.get_future();

        if (target_is_managed)
        {
            typedef typename traits::detail::shared_state_ptr_for<
                future<result_type>
            >::type shared_state_ptr;

            shared_state_ptr const& state = traits::detail::get_shared_state(f);
            state->set_on_completed(detail::keep_id_alive(id));
        }

        return f;
    }
    /// \endcond
}}


#endif
