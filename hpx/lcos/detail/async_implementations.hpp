//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ASYNC_IMPLEMENTATIONS_APR_13_2015_0829AM)
#define HPX_LCOS_ASYNC_IMPLEMENTATIONS_APR_13_2015_0829AM

#include <hpx/config.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/component_supports_migration.hpp>
#include <hpx/traits/action_was_object_migrated.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/lcos/detail/async_implementations_fwd.hpp>
#include <hpx/lcos/packaged_action.hpp>

#include <boost/mpl/bool.hpp>

namespace hpx { namespace detail
{
    /// \cond NOINTERNAL
    HPX_FORCEINLINE bool has_async_policy(launch policy)
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
        static lcos::future<Result>
        call(naming::id_type const& id, naming::address && addr, Ts &&... vs)
        {
            bool target_is_managed = false;
            naming::id_type id1;

            if (id.get_management_type() == naming::id_type::managed)
            {
                id1 = naming::id_type(id.get_gid(), naming::id_type::unmanaged);
                target_is_managed = true;
            }

            lcos::packaged_action<Action, Result> p;
            p.apply(std::move(addr), target_is_managed ? id1 : id,
                std::forward<Ts>(vs)...);

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

    template <typename Action, typename Result>
    struct sync_local_invoke<Action, lcos::future<Result> >
    {
        template <typename ...Ts>
        HPX_FORCEINLINE static lcos::future<Result>
        call(naming::id_type const&, naming::address && addr, Ts &&... vs)
        {
            HPX_ASSERT(!!addr);
            HPX_ASSERT(traits::component_type_is_compatible<
                    typename Action::component_type
                >::call(addr));

            return Action::execute_function(addr.address_,
                std::forward<Ts>(vs)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Result>
    struct sync_local_invoke_cb
    {
        template <typename Callback, typename ...Ts>
        static lcos::future<Result>
        call(naming::id_type const& id, naming::address && addr, Callback && cb,
            Ts &&... vs)
        {
            bool target_is_managed = false;
            naming::id_type id1;

            if (id.get_management_type() == naming::id_type::managed)
            {
                id1 = naming::id_type(id.get_gid(), naming::id_type::unmanaged);
                target_is_managed = true;
            }

            lcos::packaged_action<Action, Result> p;
            p.apply_cb(std::move(addr), target_is_managed ? id1 : id,
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);

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

    template <typename Action, typename Result>
    struct sync_local_invoke_cb<Action, lcos::future<Result> >
    {
        template <typename Callback, typename ...Ts>
        HPX_FORCEINLINE static lcos::future<Result>
        call(naming::id_type const&, naming::address && addr, Callback && cb,
            Ts &&... vs)
        {
            HPX_ASSERT(!!addr);
            HPX_ASSERT(traits::component_type_is_compatible<
                    typename Action::component_type
                >::call(addr));

            lcos::future<Result> f = Action::execute_function(
                addr.address_, std::forward<Ts>(vs)...);

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
    async_impl(launch policy, hpx::id_type const& id,
        Ts&&... vs)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::remote_result_type
        >::type result_type;
        typedef typename action_type::component_type component_type;

        std::pair<bool, components::pinned_ptr> r;

        naming::address addr;
        if (agas::is_local_address_cached(id, addr))
        {
            if (traits::component_supports_migration<component_type>::call())
            {
                r = traits::action_was_object_migrated<Action>::call(
                        id, addr.address_);
                if (policy == launch::sync && !r.first)
                {
                    return hpx::detail::sync_local_invoke<action_type, result_type>::
                        call(id, std::move(addr), std::forward<Ts>(vs)...);
                }
            }
            else if (policy == launch::sync)
            {
                return hpx::detail::sync_local_invoke<action_type, result_type>::
                    call(id, std::move(addr), std::forward<Ts>(vs)...);
            }
        }

        bool target_is_managed = false;
        naming::id_type id1;
        future<result_type> f;

        if (id.get_management_type() == naming::id_type::managed)
        {
            id1 = naming::id_type(id.get_gid(), naming::id_type::unmanaged);
            target_is_managed = true;
        }

        if (policy == launch::sync || hpx::detail::has_async_policy(policy))
        {
            lcos::packaged_action<action_type, result_type> p;
            p.apply(std::move(addr), target_is_managed ? id1 : id,
                std::forward<Ts>(vs)...);
            f = p.get_future();
        }
        else if (policy == launch::deferred)
        {
            lcos::packaged_action<action_type, result_type> p;
            p.apply_deferred(std::move(addr), target_is_managed ? id1 : id,
                std::forward<Ts>(vs)...);
            f = p.get_future();
        }
        else
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "async_impl", "unknown launch policy");
            return f;
        }

        // keep id alive, if needed - this allows to send the destination as an
        // unmanaged id
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
    async_cb_impl(launch policy, hpx::id_type const& id,
        Callback&& cb, Ts&&... vs)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::remote_result_type
        >::type result_type;
        typedef typename action_type::component_type component_type;

        std::pair<bool, components::pinned_ptr> r;

        naming::address addr;
        if (agas::is_local_address_cached(id, addr))
        {
            if (traits::component_supports_migration<component_type>::call())
            {
                r = traits::action_was_object_migrated<Action>::call(
                        id, addr.address_);
                if (policy == launch::sync && !r.first)
                {
                    return hpx::detail::sync_local_invoke_cb<
                            action_type, result_type
                        >::call(id, std::move(addr), std::forward<Callback>(cb),
                            std::forward<Ts>(vs)...);
                }
            }
            else if (policy == launch::sync)
            {
                return hpx::detail::sync_local_invoke_cb<
                        action_type, result_type
                    >::call(id, std::move(addr), std::forward<Callback>(cb),
                        std::forward<Ts>(vs)...);
            }
        }

        future<result_type> f;
        bool target_is_managed = false;
        naming::id_type id1;

        if (id.get_management_type() == naming::id_type::managed)
        {
            id1 = naming::id_type(id.get_gid(), naming::id_type::unmanaged);
            target_is_managed = true;
        }

        if (policy == launch::sync || hpx::detail::has_async_policy(policy))
        {
            lcos::packaged_action<action_type, result_type> p;
            p.apply_cb(std::move(addr), target_is_managed ? id1 : id,
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
            f = p.get_future();
        }
        else if (policy == launch::deferred)
        {
            lcos::packaged_action<action_type, result_type> p;
            p.apply_deferred_cb(std::move(addr), target_is_managed ? id1 : id,
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
            f = p.get_future();
        }
        else
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "async_cb_impl", "unknown launch policy");
            return f;
        }

        // keep id alive, if needed - this allows to send the destination
        // as an unmanaged id
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
