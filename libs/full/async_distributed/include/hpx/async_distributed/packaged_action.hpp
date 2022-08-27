//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/traits/action_priority.hpp>
#include <hpx/actions_base/traits/action_was_object_migrated.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_distributed/applier/apply.hpp>
#include <hpx/async_distributed/applier/apply_callback.hpp>
#include <hpx/async_distributed/promise.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/components_base/traits/component_supports_migration.hpp>
#include <hpx/components_base/traits/component_type_is_compatible.hpp>
#include <hpx/memory/intrusive_ptr.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/memory.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <asio/error.hpp>
#endif

#include <exception>
#include <memory>
#include <system_error>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos {

#if defined(HPX_HAVE_NETWORKING)
    namespace detail {

        template <typename Result>
        struct parcel_write_handler
        {
            hpx::intrusive_ptr<detail::promise_data<Result>> shared_state;

            void operator()(
                std::error_code const& ec, parcelset::parcel const& p)
            {
                // any error in the parcel layer will be stored in the future
                // object
                if (ec)
                {
                    if (hpx::tolerate_node_faults())
                    {
                        if (ec ==
                            asio::error::make_error_code(
                                asio::error::connection_reset))
                        {
                            return;
                        }
                    }
                    std::exception_ptr exception = HPX_GET_EXCEPTION(ec,
                        "packaged_action::parcel_write_handler",
                        parcelset::dump_parcel(p));
                    shared_state->set_exception(exception);
                }
            }
        };

        template <typename Result, typename Callback>
        struct parcel_write_handler_cb
        {
            hpx::intrusive_ptr<detail::promise_data<Result>> shared_state;
            Callback cb;

            void operator()(
                std::error_code const& ec, parcelset::parcel const& p)
            {
                // any error in the parcel layer will be stored in the future
                // object
                if (ec)
                {
                    std::exception_ptr exception = HPX_GET_EXCEPTION(ec,
                        "packaged_action::parcel_write_handler_cb",
                        parcelset::dump_parcel(p));
                    shared_state->set_exception(exception);
                }

                // invoke user supplied callback
                cb(ec, p);
            }
        };
    }    // namespace detail
#endif

    ///////////////////////////////////////////////////////////////////////////
    /// A packaged_action can be used by a single \a thread to invoke a
    /// (remote) action and wait for the result. The result is expected to be
    /// sent back to the packaged_action using the LCO's set_event action
    ///
    /// A packaged_action is one of the simplest synchronization primitives
    /// provided by HPX. It allows to synchronize on a eager evaluated remote
    /// operation returning a result of the type \a Result.
    ///
    /// \tparam Action   The template parameter \a Action defines the action
    ///                  to be executed by this packaged_action instance. The
    ///                  arguments \a arg0,... \a argN are used as parameters
    ///                  for this action.
    /// \tparam Result   The template parameter \a Result defines the type this
    ///                  packaged_action is expected to return from its
    ///                  associated
    ///                  future \a packaged_action#get_future.
    /// \tparam DirectExecute The template parameter \a DirectExecute is an
    ///                  optimization aid allowing to execute the action
    ///                  directly if the target is local (without spawning a
    ///                  new thread for this). This template does not have to be
    ///                  supplied explicitly as it is derived from the template
    ///                  parameter \a Action.
    ///
    /// \note            The action executed using the packaged_action as a
    ///                  continuation must return a value of a type convertible
    ///                  to the type as specified by the template parameter
    ///                  \a Result.
    template <typename Action, typename Result, bool DirectExecute>
    class packaged_action;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Result>
    class packaged_action<Action, Result, /*DirectExecute=*/false>
      : public hpx::distributed::promise<Result,
            typename hpx::traits::extract_action<Action>::remote_result_type>
    {
    protected:
        using action_type = typename hpx::traits::extract_action<Action>::type;
        using remote_result_type = typename action_type::remote_result_type;
        using base_type = hpx::distributed::promise<Result, remote_result_type>;

        ///////////////////////////////////////////////////////////////////////
        template <typename... Ts>
        void do_apply(naming::address&& addr, hpx::id_type const& id,
            threads::thread_priority priority, Ts&&... vs)
        {
            LLCO_(info).format("packaged_action::do_apply({}, {}) args({})",
                hpx::actions::detail::get_action_name<action_type>(), id,
                sizeof...(Ts));

#if defined(HPX_HAVE_NETWORKING)
            auto&& f =
                detail::parcel_write_handler<Result>{this->shared_state_};
#else
            auto shared_state = this->shared_state_;
            auto&& f = [shared_state = HPX_MOVE(shared_state)]() {};
#endif
            naming::address resolved_addr(this->resolve());
            hpx::id_type cont_id(this->get_id(false));
            naming::detail::set_dont_store_in_cache(cont_id);

            if (addr)
            {
                hpx::apply_p_cb<action_type>(
                    actions::typed_continuation<Result, remote_result_type>(
                        HPX_MOVE(cont_id), HPX_MOVE(resolved_addr)),
                    HPX_MOVE(addr), id, priority, HPX_MOVE(f),
                    HPX_FORWARD(Ts, vs)...);
            }
            else
            {
                hpx::apply_p_cb<action_type>(
                    actions::typed_continuation<Result, remote_result_type>(
                        HPX_MOVE(cont_id), HPX_MOVE(resolved_addr)),
                    id, priority, HPX_MOVE(f), HPX_FORWARD(Ts, vs)...);
            }

            this->shared_state_->mark_as_started();
        }

        template <typename... Ts>
        void do_apply(hpx::id_type const& id, threads::thread_priority priority,
            Ts&&... vs)
        {
            LLCO_(info).format("packaged_action::do_apply({}, {}) args({})",
                hpx::actions::detail::get_action_name<action_type>(), id,
                sizeof...(Ts));

#if defined(HPX_HAVE_NETWORKING)
            auto&& f =
                detail::parcel_write_handler<Result>{this->shared_state_};
#else
            auto shared_state = this->shared_state_;
            auto&& f = [shared_state = HPX_MOVE(shared_state)]() {};
#endif

            naming::address resolved_addr(this->resolve());
            hpx::id_type cont_id(this->get_id(false));
            naming::detail::set_dont_store_in_cache(cont_id);

            hpx::apply_p_cb<action_type>(
                actions::typed_continuation<Result, remote_result_type>(
                    HPX_MOVE(cont_id), HPX_MOVE(resolved_addr)),
                id, priority, HPX_MOVE(f), HPX_FORWARD(Ts, vs)...);

            this->shared_state_->mark_as_started();
        }

        template <typename Callback, typename... Ts>
        void do_apply_cb(naming::address&& addr, hpx::id_type const& id,
            threads::thread_priority priority, Callback&& cb, Ts&&... vs)
        {
            LLCO_(info).format("packaged_action::do_apply_cb({}, {}) args({})",
                hpx::actions::detail::get_action_name<action_type>(), id,
                sizeof...(Ts));

#if defined(HPX_HAVE_NETWORKING)
            using callback_type = typename std::decay<Callback>::type;
            auto&& f = detail::parcel_write_handler_cb<Result, callback_type>{
                this->shared_state_, HPX_FORWARD(Callback, cb)};
#else
            auto shared_state = this->shared_state_;
            auto&& f = [shared_state = HPX_MOVE(shared_state),
                           cb = HPX_FORWARD(Callback, cb)]() { cb(); };
#endif

            naming::address resolved_addr(this->resolve());
            hpx::id_type cont_id(this->get_id(false));
            naming::detail::set_dont_store_in_cache(cont_id);

            if (addr)
            {
                hpx::apply_p_cb<action_type>(
                    actions::typed_continuation<Result, remote_result_type>(
                        HPX_MOVE(cont_id), HPX_MOVE(resolved_addr)),
                    HPX_MOVE(addr), id, priority, HPX_MOVE(f),
                    HPX_FORWARD(Ts, vs)...);
            }
            else
            {
                hpx::apply_p_cb<action_type>(
                    actions::typed_continuation<Result, remote_result_type>(
                        HPX_MOVE(cont_id), HPX_MOVE(resolved_addr)),
                    id, priority, HPX_MOVE(f), HPX_FORWARD(Ts, vs)...);
            }

            this->shared_state_->mark_as_started();
        }

        template <typename Callback, typename... Ts>
        void do_apply_cb(hpx::id_type const& id,
            threads::thread_priority priority, Callback&& cb, Ts&&... vs)
        {
            LLCO_(info).format("packaged_action::do_apply_cb({}, {}) args({})",
                hpx::actions::detail::get_action_name<action_type>(), id,
                sizeof...(Ts));

#if defined(HPX_HAVE_NETWORKING)
            using callback_type = typename std::decay<Callback>::type;
            auto&& f = detail::parcel_write_handler_cb<Result, callback_type>{
                this->shared_state_, HPX_FORWARD(Callback, cb)};
#else
            auto shared_state = this->shared_state_;
            auto&& f = [shared_state = HPX_MOVE(shared_state),
                           cb = HPX_FORWARD(Callback, cb)]() { cb(); };
#endif

            naming::address resolved_addr(this->resolve());
            hpx::id_type cont_id(this->get_id(false));
            naming::detail::set_dont_store_in_cache(cont_id);

            hpx::apply_p_cb<action_type>(
                actions::typed_continuation<Result, remote_result_type>(
                    HPX_MOVE(cont_id), HPX_MOVE(resolved_addr)),
                id, priority, HPX_MOVE(f), HPX_FORWARD(Ts, vs)...);

            this->shared_state_->mark_as_started();
        }

    public:
        // Construct a (non-functional) instance of an \a packaged_action. To
        // use this instance its member function \a apply needs to be directly
        // called.
        packaged_action()
          : base_type(std::allocator_arg, hpx::util::internal_allocator<>{})
        {
        }

        template <typename Allocator>
        packaged_action(std::allocator_arg_t, Allocator const& alloc)
          : base_type(std::allocator_arg, alloc)
        {
        }

        template <typename... Ts>
        void apply(hpx::id_type const& id, Ts&&... vs)
        {
            do_apply(id, actions::action_priority<action_type>(),
                HPX_FORWARD(Ts, vs)...);
        }

        template <typename... Ts>
        void apply(naming::address&& addr, hpx::id_type const& id, Ts&&... vs)
        {
            do_apply(HPX_MOVE(addr), id,
                actions::action_priority<action_type>(),
                HPX_FORWARD(Ts, vs)...);
        }

        template <typename Callback, typename... Ts>
        void apply_cb(hpx::id_type const& id, Callback&& cb, Ts&&... vs)
        {
            do_apply_cb(id, actions::action_priority<action_type>(),
                HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
        }

        template <typename Callback, typename... Ts>
        void apply_cb(naming::address&& addr, hpx::id_type const& id,
            Callback&& cb, Ts&&... vs)
        {
            do_apply_cb(HPX_MOVE(addr), id,
                actions::action_priority<action_type>(),
                HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
        }

        template <typename... Ts>
        void apply_p(hpx::id_type const& id, threads::thread_priority priority,
            Ts&&... vs)
        {
            do_apply(id, priority, HPX_FORWARD(Ts, vs)...);
        }

        template <typename... Ts>
        void apply_p(naming::address&& addr, hpx::id_type const& id,
            threads::thread_priority priority, Ts&&... vs)
        {
            do_apply(HPX_MOVE(addr), id, priority, HPX_FORWARD(Ts, vs)...);
        }

        template <typename Callback, typename... Ts>
        void apply_p_cb(hpx::id_type const& id,
            threads::thread_priority priority, Callback&& cb, Ts&&... vs)
        {
            do_apply_cb(id, priority, HPX_FORWARD(Callback, cb),
                HPX_FORWARD(Ts, vs)...);
        }

        template <typename Callback, typename... Ts>
        void apply_p_cb(naming::address&& addr, hpx::id_type const& id,
            threads::thread_priority priority, Callback&& cb, Ts&&... vs)
        {
            do_apply_cb(HPX_MOVE(addr), id, priority, HPX_FORWARD(Callback, cb),
                HPX_FORWARD(Ts, vs)...);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename... Ts>
        void apply_deferred(
            naming::address&& addr, hpx::id_type const& id, Ts&&... vs)
        {
            LLCO_(info).format(
                "packaged_action::apply_deferred({}, {}) args({})",
                hpx::actions::detail::get_action_name<action_type>(), id,
                sizeof...(Ts));

#if defined(HPX_HAVE_NETWORKING)
            auto&& f =
                detail::parcel_write_handler<Result>{this->shared_state_};
#else
            auto shared_state = this->shared_state_;
            auto&& f = [shared_state = HPX_MOVE(shared_state)]() {};
#endif

            hpx::id_type cont_id(this->get_id(false));
            naming::detail::set_dont_store_in_cache(cont_id);

            auto fut = hpx::functional::apply_c_p_cb<action_type>(cont_id,
                HPX_MOVE(addr), id, actions::action_priority<action_type>(),
                HPX_MOVE(f), HPX_FORWARD(Ts, vs)...);

            this->shared_state_->set_task(HPX_MOVE(fut));
        }

        template <typename Callback, typename... Ts>
        void apply_deferred_cb(naming::address&& addr, hpx::id_type const& id,
            Callback&& cb, Ts&&... vs)
        {
            LLCO_(info).format(
                "packaged_action::apply_deferred({}, {}) args({})",
                hpx::actions::detail::get_action_name<action_type>(), id,
                sizeof...(Ts));

#if defined(HPX_HAVE_NETWORKING)
            using callback_type = typename std::decay<Callback>::type;
            auto&& f = detail::parcel_write_handler_cb<Result, callback_type>{
                this->shared_state_, HPX_FORWARD(Callback, cb)};
#else
            auto shared_state = this->shared_state_;
            auto&& f = [shared_state = HPX_MOVE(shared_state),
                           cb = HPX_FORWARD(Callback, cb)]() { cb(); };
#endif

            hpx::id_type cont_id(this->get_id(false));
            naming::detail::set_dont_store_in_cache(cont_id);

            auto fut = hpx::functional::apply_c_p_cb<action_type>(cont_id,
                HPX_MOVE(addr), id, actions::action_priority<action_type>(),
                HPX_MOVE(f), HPX_FORWARD(Ts, vs)...);

            this->shared_state_->set_task(HPX_MOVE(fut));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Result>
    class packaged_action<Action, Result, /*DirectExecute=*/true>
      : public packaged_action<Action, Result, /*DirectExecute=*/false>
    {
        using action_type = typename packaged_action<Action, Result,
            /*DirectExecute=*/false>::action_type;

    public:
        /// Construct a (non-functional) instance of an \a packaged_action. To
        /// use this instance its member function \a apply needs to be directly
        /// called.
        packaged_action()
          : packaged_action<Action, Result, false>(
                std::allocator_arg, hpx::util::internal_allocator<>{})
        {
        }

        template <typename Allocator>
        packaged_action(std::allocator_arg_t, Allocator const& alloc)
          : packaged_action<Action, Result, false>(std::allocator_arg, alloc)
        {
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename... Ts>
        void apply(hpx::id_type const& id, Ts&&... vs)
        {
            std::pair<bool, components::pinned_ptr> r;

            naming::address addr;
            if (agas::is_local_address_cached(id, addr))
            {
                using component_type = typename Action::component_type;
                HPX_ASSERT(
                    traits::component_type_is_compatible<component_type>::call(
                        addr));

                if (traits::component_supports_migration<
                        component_type>::call())
                {
                    r = traits::action_was_object_migrated<Action>::call(
                        id, addr.address_);
                    if (!r.first)
                    {
                        // local, direct execution
                        auto&& result = action_type::execute_function(
                            addr.address_, addr.type_, HPX_FORWARD(Ts, vs)...);
                        this->shared_state_->mark_as_started();
                        this->shared_state_->set_remote_data(HPX_MOVE(result));
                        return;
                    }
                }
                else
                {
                    // local, direct execution
                    auto&& result = action_type::execute_function(
                        addr.address_, addr.type_, HPX_FORWARD(Ts, vs)...);
                    this->shared_state_->mark_as_started();
                    this->shared_state_->set_remote_data(HPX_MOVE(result));
                    return;
                }
            }

            // remote execution
            this->do_apply(id, actions::action_priority<action_type>(),
                HPX_FORWARD(Ts, vs)...);
        }

        template <typename... Ts>
        void apply(naming::address&& addr, hpx::id_type const& id, Ts&&... vs)
        {
            std::pair<bool, components::pinned_ptr> r;

            if (naming::get_locality_id_from_gid(addr.locality_) ==
                agas::get_locality_id())
            {
                using component_type = typename Action::component_type;
                HPX_ASSERT(
                    traits::component_type_is_compatible<component_type>::call(
                        addr));

                if (traits::component_supports_migration<
                        component_type>::call())
                {
                    r = traits::action_was_object_migrated<Action>::call(
                        id, addr.address_);
                    if (!r.first)
                    {
                        // local, direct execution
                        auto&& result = action_type::execute_function(
                            addr.address_, addr.type_, HPX_FORWARD(Ts, vs)...);
                        this->shared_state_->mark_as_started();
                        this->shared_state_->set_remote_data(HPX_MOVE(result));
                        return;
                    }
                }
                else
                {
                    // local, direct execution
                    auto&& result = action_type::execute_function(
                        addr.address_, addr.type_, HPX_FORWARD(Ts, vs)...);
                    this->shared_state_->mark_as_started();
                    this->shared_state_->set_remote_data(HPX_MOVE(result));
                    return;
                }
            }

            // remote execution
            this->do_apply(HPX_MOVE(addr), id,
                actions::action_priority<action_type>(),
                HPX_FORWARD(Ts, vs)...);
        }

        template <typename Callback, typename... Ts>
        void apply_cb(hpx::id_type const& id, Callback&& cb, Ts&&... vs)
        {
            std::pair<bool, components::pinned_ptr> r;

            naming::address addr;
            if (agas::is_local_address_cached(id, addr))
            {
                using component_type = typename Action::component_type;
                HPX_ASSERT(
                    traits::component_type_is_compatible<component_type>::call(
                        addr));

                if (traits::component_supports_migration<
                        component_type>::call())
                {
                    r = traits::action_was_object_migrated<Action>::call(
                        id, addr.address_);
                    if (!r.first)
                    {
                        // local, direct execution
                        auto&& result = action_type::execute_function(
                            addr.address_, addr.type_, HPX_FORWARD(Ts, vs)...);
                        this->shared_state_->mark_as_started();
                        this->shared_state_->set_remote_data(HPX_MOVE(result));

                        // invoke callback
#if defined(HPX_HAVE_NETWORKING)
                        cb(std::error_code(), parcelset::parcel());
#else
                        cb();
#endif

                        return;
                    }
                }
                else
                {
                    // local, direct execution
                    auto&& result = action_type::execute_function(
                        addr.address_, addr.type_, HPX_FORWARD(Ts, vs)...);
                    this->shared_state_->mark_as_started();
                    this->shared_state_->set_remote_data(HPX_MOVE(result));

                    // invoke callback
#if defined(HPX_HAVE_NETWORKING)
                    cb(std::error_code(), parcelset::parcel());
#else
                    cb();
#endif
                    return;
                }
            }

            // remote execution
            this->do_apply_cb(id, actions::action_priority<action_type>(),
                HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
        }

        template <typename Callback, typename... Ts>
        void apply_cb(naming::address&& addr, hpx::id_type const& id,
            Callback&& cb, Ts&&... vs)
        {
            std::pair<bool, components::pinned_ptr> r;

            if (naming::get_locality_id_from_gid(addr.locality_) ==
                agas::get_locality_id())
            {
                using component_type = typename Action::component_type;
                HPX_ASSERT(
                    traits::component_type_is_compatible<component_type>::call(
                        addr));

                if (traits::component_supports_migration<
                        component_type>::call())
                {
                    r = traits::action_was_object_migrated<Action>::call(
                        id, addr.address_);
                    if (!r.first)
                    {
                        // local, direct execution
                        auto&& result = action_type::execute_function(
                            addr.address_, addr.type_, HPX_FORWARD(Ts, vs)...);
                        this->shared_state_->mark_as_started();
                        this->shared_state_->set_remote_data(HPX_MOVE(result));

                        // invoke callback
#if defined(HPX_HAVE_NETWORKING)
                        cb(std::error_code(), parcelset::parcel());
#else
                        cb();
#endif
                        return;
                    }
                }
                else
                {
                    // local, direct execution
                    auto&& result = action_type::execute_function(
                        addr.address_, addr.type_, HPX_FORWARD(Ts, vs)...);
                    this->shared_state_->mark_as_started();
                    this->shared_state_->set_remote_data(HPX_MOVE(result));

                    // invoke callback
#if defined(HPX_HAVE_NETWORKING)
                    cb(std::error_code(), parcelset::parcel());
#else
                    cb();
#endif
                    return;
                }
            }

            // remote execution
            this->do_apply_cb(HPX_MOVE(addr), id,
                actions::action_priority<action_type>(),
                HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
        }
    };
}}    // namespace hpx::lcos
