//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/config/detail/compat_error_code.hpp>
#include <hpx/actions_base/traits/action_priority.hpp>
#include <hpx/actions_base/traits/action_was_object_migrated.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_distributed/applier/apply.hpp>
#include <hpx/async_distributed/applier/apply_callback.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/components_base/traits/component_supports_migration.hpp>
#include <hpx/components_base/traits/component_type_is_compatible.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/memory/intrusive_ptr.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/memory.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <boost/asio/error.hpp>
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
                // any error in the parcel layer will be stored in the future object
                if (ec)
                {
                    if (hpx::tolerate_node_faults())
                    {
                        if (compat_error_code::equal(
                                ec, boost::asio::error::connection_reset))
                            return;
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
                // any error in the parcel layer will be stored in the future object
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
    }
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
        : public promise<Result,
              typename hpx::traits::extract_action<Action>::remote_result_type>
    {
    protected:
        using action_type = typename hpx::traits::extract_action<Action>::type;
        using remote_result_type = typename action_type::remote_result_type;
        using base_type = promise<Result, remote_result_type>;

        ///////////////////////////////////////////////////////////////////////
        template <typename... Ts>
        void do_apply(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Ts&&... vs)
        {
            LLCO_(info) << "packaged_action::do_apply("    //-V128
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", " << id << ") args(" << sizeof...(Ts) << ")";

#if defined(HPX_HAVE_NETWORKING)
            auto&& f = detail::parcel_write_handler<Result>{this->shared_state_};
#else
            auto shared_state = this->shared_state_;
            auto&& f = [shared_state = std::move(shared_state)]() {};
#endif
            naming::address resolved_addr(this->resolve());
            naming::id_type cont_id(this->get_id(false));
            naming::detail::set_dont_store_in_cache(cont_id);

            if (addr)
            {
                hpx::apply_p_cb<action_type>(
                    actions::typed_continuation<Result, remote_result_type>(
                        std::move(cont_id), std::move(resolved_addr)),
                    std::move(addr), id, priority, std::move(f),
                    std::forward<Ts>(vs)...);
            }
            else
            {
                hpx::apply_p_cb<action_type>(
                    actions::typed_continuation<Result, remote_result_type>(
                        std::move(cont_id), std::move(resolved_addr)),
                    id, priority, std::move(f), std::forward<Ts>(vs)...);
            }

            this->shared_state_->mark_as_started();
        }

        template <typename... Ts>
        void do_apply(naming::id_type const& id,
            threads::thread_priority priority, Ts&&... vs)
        {
            LLCO_(info) << "packaged_action::do_apply("    //-V128
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", " << id << ") args(" << sizeof...(Ts) << ")";

#if defined(HPX_HAVE_NETWORKING)
            auto&& f = detail::parcel_write_handler<Result>{this->shared_state_};
#else
            auto shared_state = this->shared_state_;
            auto&& f = [shared_state = std::move(shared_state)]() {};
#endif

            naming::address resolved_addr(this->resolve());
            naming::id_type cont_id(this->get_id(false));
            naming::detail::set_dont_store_in_cache(cont_id);

            hpx::apply_p_cb<action_type>(
                actions::typed_continuation<Result, remote_result_type>(
                    std::move(cont_id), std::move(resolved_addr)),
                id, priority, std::move(f), std::forward<Ts>(vs)...);

            this->shared_state_->mark_as_started();
        }

        template <typename Callback, typename... Ts>
        void do_apply_cb(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Callback&& cb, Ts&&... vs)
        {
            LLCO_(info) << "packaged_action::do_apply_cb("    //-V128
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", " << id << ") args(" << sizeof...(Ts) << ")";

#if defined(HPX_HAVE_NETWORKING)
            using callback_type = typename std::decay<Callback>::type;
            auto&& f = detail::parcel_write_handler_cb<Result, callback_type>{
                this->shared_state_, std::forward<Callback>(cb)};
#else
            auto shared_state = this->shared_state_;
            auto&& f = [shared_state = std::move(shared_state),
                           cb = std::forward<Callback>(cb)]() { cb(); };
#endif

            naming::address resolved_addr(this->resolve());
            naming::id_type cont_id(this->get_id(false));
            naming::detail::set_dont_store_in_cache(cont_id);

            if (addr)
            {
                hpx::apply_p_cb<action_type>(
                    actions::typed_continuation<Result, remote_result_type>(
                        std::move(cont_id), std::move(resolved_addr)),
                    std::move(addr), id, priority, std::move(f),
                    std::forward<Ts>(vs)...);
            }
            else
            {
                hpx::apply_p_cb<action_type>(
                    actions::typed_continuation<Result, remote_result_type>(
                        std::move(cont_id), std::move(resolved_addr)),
                    id, priority, std::move(f), std::forward<Ts>(vs)...);
            }

            this->shared_state_->mark_as_started();
        }

        template <typename Callback, typename... Ts>
        void do_apply_cb(naming::id_type const& id,
            threads::thread_priority priority, Callback&& cb, Ts&&... vs)
        {
            LLCO_(info) << "packaged_action::do_apply_cb("    //-V128
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", " << id << ") args(" << sizeof...(Ts) << ")";


#if defined(HPX_HAVE_NETWORKING)
            using callback_type = typename std::decay<Callback>::type;
            auto&& f = detail::parcel_write_handler_cb<Result, callback_type>{
                this->shared_state_, std::forward<Callback>(cb)};
#else
            auto shared_state = this->shared_state_;
            auto&& f = [shared_state = std::move(shared_state),
                           cb = std::forward<Callback>(cb)]() { cb(); };
#endif

            naming::address resolved_addr(this->resolve());
            naming::id_type cont_id(this->get_id(false));
            naming::detail::set_dont_store_in_cache(cont_id);

            hpx::apply_p_cb<action_type>(
                actions::typed_continuation<Result, remote_result_type>(
                    std::move(cont_id), std::move(resolved_addr)),
                id, priority, std::move(f), std::forward<Ts>(vs)...);

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
        void apply(naming::id_type const& id, Ts&&... vs)
        {
            do_apply(id, actions::action_priority<action_type>(),
                std::forward<Ts>(vs)...);
        }

        template <typename... Ts>
        void apply(
            naming::address&& addr, naming::id_type const& id, Ts&&... vs)
        {
            do_apply(std::move(addr), id,
                actions::action_priority<action_type>(),
                std::forward<Ts>(vs)...);
        }

        template <typename Callback, typename... Ts>
        void apply_cb(naming::id_type const& id, Callback&& cb, Ts&&... vs)
        {
            do_apply_cb(id, actions::action_priority<action_type>(),
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
        }

        template <typename Callback, typename... Ts>
        void apply_cb(naming::address&& addr, naming::id_type const& id,
            Callback&& cb, Ts&&... vs)
        {
            do_apply_cb(std::move(addr), id,
                actions::action_priority<action_type>(),
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
        }

        template <typename... Ts>
        void apply_p(naming::id_type const& id,
            threads::thread_priority priority, Ts&&... vs)
        {
            do_apply(id, priority, std::forward<Ts>(vs)...);
        }

        template <typename... Ts>
        void apply_p(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Ts&&... vs)
        {
            do_apply(std::move(addr), id, priority, std::forward<Ts>(vs)...);
        }

        template <typename Callback, typename... Ts>
        void apply_p_cb(naming::id_type const& id,
            threads::thread_priority priority, Callback&& cb, Ts&&... vs)
        {
            do_apply_cb(id, priority, std::forward<Callback>(cb),
                std::forward<Ts>(vs)...);
        }

        template <typename Callback, typename... Ts>
        void apply_p_cb(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Callback&& cb, Ts&&... vs)
        {
            do_apply_cb(std::move(addr), id, priority,
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename... Ts>
        void apply_deferred(
            naming::address&& addr, naming::id_type const& id, Ts&&... vs)
        {
            LLCO_(info) << "packaged_action::apply_deferred("    //-V128
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", " << id << ") args(" << sizeof...(Ts) << ")";

#if defined(HPX_HAVE_NETWORKING)
            auto&& f = detail::parcel_write_handler<Result>{this->shared_state_};
#else
            auto shared_state = this->shared_state_;
            auto&& f = [shared_state = std::move(shared_state)]() {};
#endif

            naming::id_type cont_id(this->get_id(false));
            naming::detail::set_dont_store_in_cache(cont_id);

            auto fut = hpx::functional::apply_c_p_cb<action_type>(cont_id,
                std::move(addr), id, actions::action_priority<action_type>(),
                std::move(f), std::forward<Ts>(vs)...);

            this->shared_state_->set_task(std::move(fut));
        }

        template <typename Callback, typename... Ts>
        void apply_deferred_cb(naming::address&& addr,
            naming::id_type const& id, Callback&& cb, Ts&&... vs)
        {
            LLCO_(info) << "packaged_action::apply_deferred("    //-V128
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", " << id << ") args(" << sizeof...(Ts) << ")";

#if defined(HPX_HAVE_NETWORKING)
            using callback_type = typename std::decay<Callback>::type;
            auto&& f = detail::parcel_write_handler_cb<Result, callback_type>{
                this->shared_state_, std::forward<Callback>(cb)};
#else
            auto shared_state = this->shared_state_;
            auto&& f = [shared_state = std::move(shared_state),
                           cb = std::forward<Callback>(cb)]() { cb(); };
#endif

            naming::id_type cont_id(this->get_id(false));
            naming::detail::set_dont_store_in_cache(cont_id);

            auto fut = hpx::functional::apply_c_p_cb<action_type>(cont_id,
                std::move(addr), id, actions::action_priority<action_type>(),
                std::move(f), std::forward<Ts>(vs)...);

            this->shared_state_->set_task(std::move(fut));
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
        void apply(naming::id_type const& id, Ts&&... vs)
        {
            std::pair<bool, components::pinned_ptr> r;

            naming::address addr;
            if (agas::is_local_address_cached(id, addr))
            {
                using component_type = typename Action::component_type;
                HPX_ASSERT(
                    traits::component_type_is_compatible<component_type>::call(
                        addr));

                if (traits::component_supports_migration<component_type>::call())
                {
                    r = traits::action_was_object_migrated<Action>::call(
                        id, addr.address_);
                    if (!r.first)
                    {
                        // local, direct execution
                        auto && result = action_type::execute_function(
                            addr.address_, addr.type_, std::forward<Ts>(vs)...);
                        this->shared_state_->mark_as_started();
                        this->shared_state_->set_remote_data(std::move(result));
                        return;
                    }
                }
                else
                {
                    // local, direct execution
                    auto && result = action_type::execute_function(
                        addr.address_, addr.type_, std::forward<Ts>(vs)...);
                    this->shared_state_->mark_as_started();
                    this->shared_state_->set_remote_data(std::move(result));
                    return;
                }
            }

            // remote execution
            this->do_apply(id, actions::action_priority<action_type>(),
                std::forward<Ts>(vs)...);
        }

        template <typename... Ts>
        void apply(
            naming::address&& addr, naming::id_type const& id, Ts&&... vs)
        {
            std::pair<bool, components::pinned_ptr> r;

            if (addr.locality_ == hpx::get_locality())
            {
                using component_type = typename Action::component_type;
                HPX_ASSERT(
                    traits::component_type_is_compatible<component_type>::call(
                        addr));

                if (traits::component_supports_migration<component_type>::call())
                {
                    r = traits::action_was_object_migrated<Action>::call(
                        id, addr.address_);
                    if (!r.first)
                    {
                        // local, direct execution
                        auto && result = action_type::execute_function(
                            addr.address_, addr.type_, std::forward<Ts>(vs)...);
                        this->shared_state_->mark_as_started();
                        this->shared_state_->set_remote_data(std::move(result));
                        return;
                    }
                }
                else
                {
                    // local, direct execution
                    auto && result = action_type::execute_function(
                        addr.address_, addr.type_, std::forward<Ts>(vs)...);
                    this->shared_state_->mark_as_started();
                    this->shared_state_->set_remote_data(std::move(result));
                    return;
                }
            }

            // remote execution
            this->do_apply(std::move(addr), id,
                actions::action_priority<action_type>(),
                std::forward<Ts>(vs)...);
        }

        template <typename Callback, typename... Ts>
        void apply_cb(naming::id_type const& id, Callback&& cb, Ts&&... vs)
        {
            std::pair<bool, components::pinned_ptr> r;

            naming::address addr;
            if (agas::is_local_address_cached(id, addr))
            {
                using component_type = typename Action::component_type;
                HPX_ASSERT(
                    traits::component_type_is_compatible<component_type>::call(
                        addr));

                if (traits::component_supports_migration<component_type>::call())
                {
                    r = traits::action_was_object_migrated<Action>::call(
                        id, addr.address_);
                    if (!r.first)
                    {
                        // local, direct execution
                        auto && result = action_type::execute_function(
                            addr.address_, addr.type_, std::forward<Ts>(vs)...);
                        this->shared_state_->mark_as_started();
                        this->shared_state_->set_remote_data(std::move(result));

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
                        addr.address_, addr.type_, std::forward<Ts>(vs)...);
                    this->shared_state_->mark_as_started();
                    this->shared_state_->set_remote_data(std::move(result));

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
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
        }

        template <typename Callback, typename... Ts>
        void apply_cb(naming::address&& addr, naming::id_type const& id,
            Callback&& cb, Ts&&... vs)
        {
            std::pair<bool, components::pinned_ptr> r;

            if (addr.locality_ == hpx::get_locality())
            {
                using component_type = typename Action::component_type;
                HPX_ASSERT(
                    traits::component_type_is_compatible<component_type>::call(
                        addr));

                if (traits::component_supports_migration<component_type>::call())
                {
                    r = traits::action_was_object_migrated<Action>::call(
                        id, addr.address_);
                    if (!r.first)
                    {
                        // local, direct execution
                        auto && result = action_type::execute_function(
                            addr.address_, addr.type_, std::forward<Ts>(vs)...);
                        this->shared_state_->mark_as_started();
                        this->shared_state_->set_remote_data(std::move(result));

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
                    auto && result = action_type::execute_function(
                        addr.address_, addr.type_, std::forward<Ts>(vs)...);
                    this->shared_state_->mark_as_started();
                    this->shared_state_->set_remote_data(std::move(result));

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
            this->do_apply_cb(std::move(addr), id,
                actions::action_priority<action_type>(),
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
        }
    };
}}

