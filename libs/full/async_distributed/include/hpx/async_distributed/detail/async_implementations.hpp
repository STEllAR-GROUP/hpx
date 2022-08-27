//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/traits/action_select_direct_execution.hpp>
#include <hpx/actions_base/traits/action_was_object_migrated.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_distributed/detail/async_implementations_fwd.hpp>
#include <hpx/async_distributed/packaged_action.hpp>
#include <hpx/components_base/traits/action_decorate_function.hpp>
#include <hpx/components_base/traits/component_supports_migration.hpp>
#include <hpx/components_base/traits/component_type_is_compatible.hpp>
#include <hpx/functional/traits/get_function_address.hpp>
#include <hpx/functional/traits/get_function_annotation.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/runtime_local.hpp>
#include <hpx/modules/threading.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/threading_base/thread_init_data.hpp>

#include <cstddef>
#include <utility>

namespace hpx { namespace detail {
    /// \cond NOINTERNAL
    ///////////////////////////////////////////////////////////////////////////
    struct keep_id_alive
    {
        explicit keep_id_alive(hpx::id_type const& id)
          : id_(id)
        {
        }

        void operator()() const {}

        hpx::id_type id_;
    };

    template <typename T>
    future<T> keep_alive(future<T>&& f, id_type const& id)
    {
        if (id.get_management_type() == hpx::id_type::management_type::managed)
        {
            traits::detail::get_shared_state(f)->set_on_completed(
                hpx::detail::keep_id_alive(id));
        }
        return HPX_MOVE(f);
    }

    struct keep_id_and_ptr_alive
    {
        explicit keep_id_and_ptr_alive(
            hpx::id_type const& id, components::pinned_ptr&& p)
          : id_(id)
          , p_(HPX_MOVE(p))
        {
        }

        void operator()() const {}

        hpx::id_type id_;
        components::pinned_ptr p_;
    };

    template <typename T>
    future<T> keep_alive(
        future<T>&& f, id_type const& id, components::pinned_ptr&& p)
    {
        if (id.get_management_type() == hpx::id_type::management_type::managed)
        {
            traits::detail::get_shared_state(f)->set_on_completed(
                hpx::detail::keep_id_and_ptr_alive(id, HPX_MOVE(p)));
        }
        return HPX_MOVE(f);
    }

    template <typename Result>
    class handle_managed_target
    {
    public:
        HPX_NON_COPYABLE(handle_managed_target);

    public:
        handle_managed_target(hpx::id_type const& id, future<Result>& f)
          : target_is_managed_(false)
          , id_(id)
          , f_(f)
        {
            if (id.get_management_type() ==
                hpx::id_type::management_type::managed)
            {
                unmanaged_id_ = hpx::id_type(
                    id.get_gid(), hpx::id_type::management_type::unmanaged);
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
                    future<Result>>::type shared_state_ptr;

                shared_state_ptr const& state =
                    traits::detail::get_shared_state(f_);
                if (state)
                {
                    HPX_ASSERT(id_);
                    HPX_ASSERT(unmanaged_id_);
                    state->set_on_completed(hpx::detail::keep_id_alive(id_));
                }
            }
        }

        hpx::id_type const& get_id() const
        {
            return target_is_managed_ ? unmanaged_id_ : id_;
        }

        bool target_is_managed_;
        hpx::id_type const& id_;
        hpx::id_type unmanaged_id_;
        future<Result>& f_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Result>
    struct sync_local_invoke
    {
        template <typename... Ts>
        static hpx::future<Result> call(
            hpx::id_type const& /*id*/, naming::address&& addr, Ts&&... vs)
        {
            try
            {
                typedef typename Action::remote_result_type remote_result_type;

                typedef traits::get_remote_result<Result, remote_result_type>
                    get_remote_result_type;

                return make_ready_future(
                    get_remote_result_type::call(Action::execute_function(
                        addr.address_, addr.type_, HPX_FORWARD(Ts, vs)...)));
            }
            catch (...)
            {
                return make_exceptional_future<Result>(
                    std::current_exception());
            }
        }
    };

    template <typename Action>
    bool can_invoke_locally()
    {
        std::ptrdiff_t requested_stack_size =
            threads::get_stack_size(static_cast<threads::thread_stacksize>(
                traits::action_stacksize<Action>::value));
        constexpr bool df = traits::action_decorate_function<Action>::value;
        return !df && this_thread::get_stack_size() >= requested_stack_size &&
            this_thread::has_sufficient_stack_space(requested_stack_size);
    }

    template <typename Action>
    struct sync_local_invoke<Action, void>
    {
        template <typename... Ts>
        static hpx::future<void> call(
            hpx::id_type const& /*id*/, naming::address&& addr, Ts&&... vs)
        {
            try
            {
                Action::execute_function(
                    addr.address_, addr.type_, HPX_FORWARD(Ts, vs)...);

                return make_ready_future();
            }
            catch (...)
            {
                return make_exceptional_future<void>(std::current_exception());
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Result>
    struct sync_local_invoke_cb
    {
        template <typename Callback, typename... Ts>
        static hpx::future<Result> call(hpx::id_type const& id,
            naming::address&& addr, Callback&& cb, Ts&&... vs)
        {
            future<Result> f;
            {
                handle_managed_target<Result> hmt(id, f);
                lcos::packaged_action<Action, Result> p;

                f = p.get_future();
                p.apply_cb(HPX_MOVE(addr), hmt.get_id(),
                    HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
                f.wait();
            }
            return f;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename... Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type>
    async_remote_impl(launch::sync_policy, hpx::id_type const& id,
        naming::address&& addr, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::local_result_type result_type;

        future<result_type> f;
        {
            handle_managed_target<result_type> hmt(id, f);
            lcos::packaged_action<action_type, result_type> p;
            f = p.get_future();
            p.apply(HPX_MOVE(addr), hmt.get_id(), HPX_FORWARD(Ts, vs)...);
            f.wait();
        }
        return f;
    }

    template <typename Action, typename... Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type>
    async_remote_impl(launch::async_policy, hpx::id_type const& id,
        naming::address&& addr, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::local_result_type result_type;

        hpx::future<result_type> f;
        {
            handle_managed_target<result_type> hmt(id, f);
            lcos::packaged_action<action_type, result_type> p;
            f = p.get_future();
            p.apply(HPX_MOVE(addr), hmt.get_id(), HPX_FORWARD(Ts, vs)...);
        }
        return f;
    }

    template <typename Action, typename... Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type>
    async_remote_impl(launch::deferred_policy, hpx::id_type const& id,
        naming::address&& addr, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::local_result_type result_type;

        hpx::future<result_type> f;
        {
            handle_managed_target<result_type> hmt(id, f);
            lcos::packaged_action<action_type, result_type> p;
            f = p.get_future();
            p.apply_deferred(
                HPX_MOVE(addr), hmt.get_id(), HPX_FORWARD(Ts, vs)...);
        }
        return f;
    }

    // generic function for dynamic launch policy
    template <typename Action, typename... Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type>
    async_remote_impl(launch policy, hpx::id_type const& id,
        naming::address&& addr, Ts&&... vs)
    {
        if (policy == launch::sync)
        {
            return async_remote_impl<Action>(
                launch::sync, id, HPX_MOVE(addr), HPX_FORWARD(Ts, vs)...);
        }
        if (hpx::detail::has_async_policy(policy))
        {
            return async_remote_impl<Action>(
                launch::async, id, HPX_MOVE(addr), HPX_FORWARD(Ts, vs)...);
        }
        if (policy == launch::deferred)
        {
            return async_remote_impl<Action>(
                launch::deferred, id, HPX_MOVE(addr), HPX_FORWARD(Ts, vs)...);
        }

        HPX_THROW_EXCEPTION(
            bad_parameter, "async_remote_impl", "unknown launch policy");
    }

    ///////////////////////////////////////////////////////////////////////////
    // do local invocation, if possible
    template <typename Action>
    struct action_invoker
    {
        typedef typename Action::remote_result_type remote_result_type;
        typedef typename Action::local_result_type result_type;
        typedef traits::get_remote_result<result_type, remote_result_type>
            get_remote_result_type;

        template <typename... Ts>
        HPX_FORCEINLINE result_type operator()(
            naming::address::address_type lva,
            naming::address::component_type comptype, Ts&&... vs) const
        {
            return get_remote_result_type::call(
                Action::invoker(lva, comptype, HPX_FORWARD(Ts, vs)...));
        }
    };
}}    // namespace hpx::detail

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits {

    template <typename Action>
    struct get_function_address<hpx::detail::action_invoker<Action>>
    {
        static constexpr std::size_t call(
            hpx::detail::action_invoker<Action> const&) noexcept
        {
            return reinterpret_cast<std::size_t>(&Action::invoker);
        }
    };

    template <typename Action>
    struct get_function_annotation<hpx::detail::action_invoker<Action>>
    {
        static constexpr char const* call(
            hpx::detail::action_invoker<Action> const&) noexcept
        {
            return hpx::actions::detail::get_action_name<Action>();
        }
    };

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename Action>
    struct get_function_annotation_itt<hpx::detail::action_invoker<Action>>
    {
        static util::itt::string_handle call(
            hpx::detail::action_invoker<Action> const&) noexcept
        {
            return hpx::actions::detail::get_action_name_itt<Action>();
        }
    };
#endif
}}    // namespace hpx::traits
#endif

namespace hpx { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename... Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type>
    async_local_impl(launch policy, hpx::id_type const& id,
        naming::address& addr, std::pair<bool, components::pinned_ptr>& r,
        Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::local_result_type result_type;

        if (policy == launch::sync || action_type::direct_execution::value)
        {
            return hpx::detail::sync_local_invoke<action_type,
                result_type>::call(id, HPX_MOVE(addr), HPX_FORWARD(Ts, vs)...);
        }
        if (hpx::detail::has_async_policy(policy))
        {
            return keep_alive(
                hpx::async(policy, action_invoker<action_type>(), addr.address_,
                    addr.type_, HPX_FORWARD(Ts, vs)...),
                id, HPX_MOVE(r.second));
        }

        HPX_ASSERT(policy == launch::deferred);

        return keep_alive(
            hpx::async(launch::deferred, action_invoker<action_type>(),
                addr.address_, addr.type_, HPX_FORWARD(Ts, vs)...),
            id, HPX_MOVE(r.second));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Result, typename... Ts>
    bool async_local_impl_all(launch policy, hpx::id_type const& id,
        naming::address& addr, std::pair<bool, components::pinned_ptr>& r,
        hpx::future<Result>& f, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        //typedef typename action_type::local_result_type result_type;
        typedef typename action_type::component_type component_type;

        // route launch policy through component
        policy = traits::action_select_direct_execution<Action>::call(
            policy, addr.address_);

        if (traits::component_supports_migration<component_type>::call())
        {
            r = traits::action_was_object_migrated<Action>::call(
                id, addr.address_);

            if (!r.first)
            {
                f = async_local_impl<Action>(
                    policy, id, addr, r, HPX_FORWARD(Ts, vs)...);
                return true;
            }

            // can't locally handle object if it is currently being migrated
            return false;
        }

        f = async_local_impl<Action>(
            policy, id, addr, r, HPX_FORWARD(Ts, vs)...);

        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Launch, typename... Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type>
    async_impl(Launch&& policy, hpx::id_type const& id, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::local_result_type result_type;

        std::pair<bool, components::pinned_ptr> r;

        naming::address addr;
        if (agas::is_local_address_cached(id, addr) &&
            can_invoke_locally<action_type>())
        {
            hpx::future<result_type> f;
            if (async_local_impl_all<Action>(
                    policy, id, addr, r, f, HPX_FORWARD(Ts, vs)...))
            {
                return f;
            }
        }

        return async_remote_impl<Action>(HPX_FORWARD(Launch, policy), id,
            HPX_MOVE(addr), HPX_FORWARD(Ts, vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \note This function is part of the invocation policy implemented by
    ///       this class
    template <typename Action, typename Callback, typename... Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type>
    async_cb_impl(
        launch policy, hpx::id_type const& id, Callback&& cb, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::local_result_type result_type;
        typedef typename action_type::component_type component_type;

        std::pair<bool, components::pinned_ptr> r;

        naming::address addr;
        if (agas::is_local_address_cached(id, addr) &&
            can_invoke_locally<action_type>())
        {
            // route launch policy through component
            policy = traits::action_select_direct_execution<Action>::call(
                policy, addr.address_);

            if (traits::component_supports_migration<component_type>::call())
            {
                r = traits::action_was_object_migrated<Action>::call(
                    id, addr.address_);
                if (!r.first)
                {
                    if (policy == launch::sync ||
                        action_type::direct_execution::value)
                    {
                        return hpx::detail::sync_local_invoke_cb<action_type,
                            result_type>::call(id, HPX_MOVE(addr),
                            HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
                    }
                }
            }
            else if (policy == launch::sync ||
                action_type::direct_execution::value)
            {
                return hpx::detail::sync_local_invoke_cb<action_type,
                    result_type>::call(id, HPX_MOVE(addr),
                    HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
            }
        }

        future<result_type> f;
        {
            handle_managed_target<result_type> hmt(id, f);

            if (policy == launch::sync || hpx::detail::has_async_policy(policy))
            {
                lcos::packaged_action<action_type, result_type> p;

                f = p.get_future();
                p.apply_cb(HPX_MOVE(addr), hmt.get_id(),
                    HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
                if (policy == launch::sync)
                    f.wait();
            }
            else if (policy == launch::deferred)
            {
                lcos::packaged_action<action_type, result_type> p;

                f = p.get_future();
                p.apply_deferred_cb(HPX_MOVE(addr), hmt.get_id(),
                    HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
            }
            else
            {
                HPX_THROW_EXCEPTION(
                    bad_parameter, "async_cb_impl", "unknown launch policy");
                return f;
            }
        }
        return f;
    }

    template <typename Action, typename Callback, typename... Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type>
    async_cb_impl(hpx::detail::sync_policy, hpx::id_type const& id,
        Callback&& cb, Ts&&... vs)
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
                    return hpx::detail::sync_local_invoke_cb<action_type,
                        result_type>::call(id, HPX_MOVE(addr),
                        HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
                }
            }
            else
            {
                return hpx::detail::sync_local_invoke_cb<action_type,
                    result_type>::call(id, HPX_MOVE(addr),
                    HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
            }
        }

        future<result_type> f;
        {
            handle_managed_target<result_type> hmt(id, f);
            lcos::packaged_action<action_type, result_type> p;

            f = p.get_future();
            p.apply_cb(HPX_MOVE(addr), hmt.get_id(), HPX_FORWARD(Callback, cb),
                HPX_FORWARD(Ts, vs)...);
            f.wait();
        }
        return f;
    }

    template <typename Action, typename Callback, typename... Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type>
    async_cb_impl(hpx::detail::async_policy async_policy,
        hpx::id_type const& id, Callback&& cb, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::local_result_type result_type;
        typedef typename action_type::component_type component_type;

        std::pair<bool, components::pinned_ptr> r;

        naming::address addr;
        if (agas::is_local_address_cached(id, addr))
        {
            // route launch policy through component
            launch policy =
                traits::action_select_direct_execution<Action>::call(
                    async_policy, addr.address_);

            if (traits::component_supports_migration<component_type>::call())
            {
                r = traits::action_was_object_migrated<Action>::call(
                    id, addr.address_);
                if (!r.first)
                {
                    if (policy == launch::sync ||
                        action_type::direct_execution::value)
                    {
                        return hpx::detail::sync_local_invoke_cb<action_type,
                            result_type>::call(id, HPX_MOVE(addr),
                            HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
                    }
                }
            }
            else if (policy == launch::sync ||
                action_type::direct_execution::value)
            {
                return hpx::detail::sync_local_invoke_cb<action_type,
                    result_type>::call(id, HPX_MOVE(addr),
                    HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
            }
        }

        future<result_type> f;
        {
            handle_managed_target<result_type> hmt(id, f);
            lcos::packaged_action<action_type, result_type> p;

            f = p.get_future();
            p.apply_cb(HPX_MOVE(addr), hmt.get_id(), HPX_FORWARD(Callback, cb),
                HPX_FORWARD(Ts, vs)...);
        }
        return f;
    }

    template <typename Action, typename Callback, typename... Ts>
    hpx::future<
        typename hpx::traits::extract_action<Action>::type::local_result_type>
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
            p.apply_deferred_cb(HPX_MOVE(addr), hmt.get_id(),
                HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
        }
        return f;
    }
    /// \endcond
}}    // namespace hpx::detail
