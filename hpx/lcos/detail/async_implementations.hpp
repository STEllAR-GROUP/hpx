//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ASYNC_IMPLEMENTATIONS_APR_13_2015_0829AM)
#define HPX_LCOS_ASYNC_IMPLEMENTATIONS_APR_13_2015_0829AM

#include <hpx/config.hpp>
#include <hpx/lcos/detail/async_implementations_fwd.hpp>
#include <hpx/lcos/packaged_action.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/traits/action_was_object_migrated.hpp>
#include <hpx/traits/component_supports_migration.hpp>
#include <hpx/traits/component_type_is_compatible.hpp>
#include <hpx/traits/extract_action.hpp>
#include <hpx/traits/future_access.hpp>

#include <utility>

namespace hpx { namespace detail
{
    /// \cond NOINTERNAL
    ///////////////////////////////////////////////////////////////////////////
    struct keep_id_alive
    {
        explicit keep_id_alive(naming::id_type const& id)
          : id_(id)
        {}

        void operator()() const {}

        naming::id_type id_;
    };

    template <typename Result>
    class handle_managed_target
    {
    public:
        HPX_NON_COPYABLE(handle_managed_target);

    public:
        handle_managed_target(hpx::id_type const& id, future<Result>& f)
          : target_is_managed_(false), id_(id), f_(f)
        {
            if (id.get_management_type() == naming::id_type::managed)
            {
                unmanaged_id_ = naming::id_type(
                    id.get_gid(), naming::id_type::unmanaged);
                target_is_managed_ = true;
            }
        }

        ~handle_managed_target()
        {
            // keep id alive, if needed - this allows to send the destination
            // as an unmanaged id
            if (target_is_managed_)
            {
                typedef typename traits::detail::shared_state_ptr_for<
                    future<Result>
                >::type shared_state_ptr;

                shared_state_ptr const& state =
                    traits::detail::get_shared_state(f_);
                if (state)
                {
                    HPX_ASSERT(id_);
                    HPX_ASSERT(unmanaged_id_);
                    state->set_on_completed(
                        hpx::detail::keep_id_alive(id_));
                }
            }
        }

        hpx::id_type const& get_id() const
        {
            return target_is_managed_ ? unmanaged_id_ : id_;
        }

        bool target_is_managed_;
        naming::id_type const& id_;
        naming::id_type unmanaged_id_;
        future<Result>& f_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Result>
    struct sync_local_invoke
    {
        template <typename ...Ts>
        static lcos::future<Result>
        call(naming::id_type const& id, naming::address && addr, Ts &&... vs)
        {
            future<Result> f;
            {
                handle_managed_target<Result> hmt(id, f);
                lcos::packaged_action<Action, Result> p;

                f = p.get_future();
                p.apply(std::move(addr), hmt.get_id(), std::forward<Ts>(vs)...);
                f.wait();
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

            return Action::execute_function(addr.address_, addr.type_,
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
            future<Result> f;
            {
                handle_managed_target<Result> hmt(id, f);
                lcos::packaged_action<Action, Result> p;

                f = p.get_future();
                p.apply_cb(std::move(addr), hmt.get_id(),
                    std::forward<Callback>(cb), std::forward<Ts>(vs)...);
                f.wait();
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
                addr.address_, addr.type_, std::forward<Ts>(vs)...);

            // invoke callback
            cb(boost::system::error_code(), parcelset::parcel());

            return f;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename ...Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::local_result_type
    >
    async_impl(launch policy, hpx::id_type const& id, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::local_result_type result_type;
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

        future<result_type> f;
        {
            handle_managed_target<result_type> hmt(id, f);

            if (policy == launch::sync || hpx::detail::has_async_policy(policy))
            {
                lcos::packaged_action<action_type, result_type> p;

                f = p.get_future();
                p.apply(std::move(addr), hmt.get_id(), std::forward<Ts>(vs)...);
                if (policy == launch::sync)
                    f.wait();
            }
            else if (policy == launch::deferred)
            {
                lcos::packaged_action<action_type, result_type> p;

                f = p.get_future();
                p.apply_deferred(std::move(addr), hmt.get_id(),
                    std::forward<Ts>(vs)...);
            }
            else
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "async_impl", "unknown launch policy");
                return f;
            }
        }
        return f;
    }

    template <typename Action, typename ...Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::local_result_type
    >
    async_impl(hpx::detail::sync_policy, hpx::id_type const& id, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::local_result_type result_type;
        typedef typename action_type::component_type component_type;

        std::pair<bool, components::pinned_ptr> r;

        naming::address addr;
        if (agas::is_local_address_cached(id, addr))
        {
            if (traits::component_supports_migration<component_type>::call())
            {
                r = traits::action_was_object_migrated<Action>::call(
                        id, addr.address_);
                if (!r.first)
                {
                    return hpx::detail::sync_local_invoke<action_type, result_type>::
                        call(id, std::move(addr), std::forward<Ts>(vs)...);
                }
            }
            else
            {
                return hpx::detail::sync_local_invoke<action_type, result_type>::
                    call(id, std::move(addr), std::forward<Ts>(vs)...);
            }
        }

        future<result_type> f;
        {
            handle_managed_target<result_type> hmt(id, f);
            lcos::packaged_action<action_type, result_type> p;

            f = p.get_future();
            p.apply(std::move(addr), hmt.get_id(), std::forward<Ts>(vs)...);
            f.wait();
        }
        return f;
    }

    template <typename Action, typename ...Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::local_result_type
    >
    async_impl(hpx::detail::async_policy, hpx::id_type const& id, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::local_result_type result_type;

        naming::address addr;
        agas::is_local_address_cached(id, addr);

        future<result_type> f;
        {
            handle_managed_target<result_type> hmt(id, f);
            lcos::packaged_action<action_type, result_type> p;

            f = p.get_future();
            p.apply(std::move(addr), hmt.get_id(), std::forward<Ts>(vs)...);
        }
        return f;
    }

    template <typename Action, typename ...Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::local_result_type
    >
    async_impl(hpx::detail::deferred_policy, hpx::id_type const& id, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::local_result_type result_type;

        naming::address addr;
        agas::is_local_address_cached(id, addr);

        future<result_type> f;
        {
            handle_managed_target<result_type> hmt(id, f);
            lcos::packaged_action<action_type, result_type> p;

            f = p.get_future();
            p.apply_deferred(std::move(addr), hmt.get_id(),
                std::forward<Ts>(vs)...);
        }
        return f;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \note This function is part of the invocation policy implemented by
    ///       this class
    ///
    template <typename Action, typename Callback, typename ...Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::local_result_type
    >
    async_cb_impl(launch policy, hpx::id_type const& id, Callback&& cb, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::local_result_type result_type;
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
        {
            handle_managed_target<result_type> hmt(id, f);

            if (policy == launch::sync || hpx::detail::has_async_policy(policy))
            {
                lcos::packaged_action<action_type, result_type> p;

                f = p.get_future();
                p.apply_cb(std::move(addr), hmt.get_id(),
                    std::forward<Callback>(cb), std::forward<Ts>(vs)...);
                if (policy == launch::sync)
                    f.wait();
            }
            else if (policy == launch::deferred)
            {
                lcos::packaged_action<action_type, result_type> p;

                f = p.get_future();
                p.apply_deferred_cb(std::move(addr), hmt.get_id(),
                    std::forward<Callback>(cb), std::forward<Ts>(vs)...);
            }
            else
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "async_cb_impl", "unknown launch policy");
                return f;
            }
        }
        return f;
    }

    template <typename Action, typename Callback, typename ...Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::local_result_type
    >
    async_cb_impl(hpx::detail::sync_policy, hpx::id_type const& id, Callback&& cb,
        Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::local_result_type result_type;
        typedef typename action_type::component_type component_type;

        std::pair<bool, components::pinned_ptr> r;

        naming::address addr;
        if (agas::is_local_address_cached(id, addr))
        {
            if (traits::component_supports_migration<component_type>::call())
            {
                r = traits::action_was_object_migrated<Action>::call(
                        id, addr.address_);
                if (!r.first)
                {
                    return hpx::detail::sync_local_invoke_cb<
                            action_type, result_type
                        >::call(id, std::move(addr), std::forward<Callback>(cb),
                            std::forward<Ts>(vs)...);
                }
            }
            else
            {
                return hpx::detail::sync_local_invoke_cb<
                        action_type, result_type
                    >::call(id, std::move(addr), std::forward<Callback>(cb),
                        std::forward<Ts>(vs)...);
            }
        }

        future<result_type> f;
        {
            handle_managed_target<result_type> hmt(id, f);
            lcos::packaged_action<action_type, result_type> p;

            f = p.get_future();
            p.apply_cb(std::move(addr), hmt.get_id(), std::forward<Callback>(cb),
                std::forward<Ts>(vs)...);
            f.wait();
        }
        return f;
    }

    template <typename Action, typename Callback, typename ...Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::local_result_type
    >
    async_cb_impl(hpx::detail::async_policy, hpx::id_type const& id, Callback&& cb,
        Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::local_result_type result_type;

        naming::address addr;
        agas::is_local_address_cached(id, addr);

        future<result_type> f;
        {
            handle_managed_target<result_type> hmt(id, f);
            lcos::packaged_action<action_type, result_type> p;

            f = p.get_future();
            p.apply_cb(std::move(addr), hmt.get_id(), std::forward<Callback>(cb),
                std::forward<Ts>(vs)...);
        }
        return f;
    }

    template <typename Action, typename Callback, typename ...Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::local_result_type
    >
    async_cb_impl(hpx::detail::deferred_policy, hpx::id_type const& id,
        Callback&& cb, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::local_result_type result_type;

        naming::address addr;
        agas::is_local_address_cached(id, addr);

        future<result_type> f;
        {
            handle_managed_target<result_type> hmt(id, f);
            lcos::packaged_action<action_type, result_type> p;

            f = p.get_future();
            p.apply_deferred_cb(std::move(addr), hmt.get_id(),
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
        }
        return f;
    }
    /// \endcond
}}

#endif
