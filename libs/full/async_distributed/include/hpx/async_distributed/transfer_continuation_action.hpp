//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file transfer_continuation_action.hpp

#pragma once

#include <hpx/config.hpp>

#include <hpx/actions/base_action.hpp>
#include <hpx/actions/post_helper.hpp>
#include <hpx/actions/register_action.hpp>
#include <hpx/actions/transfer_base_action.hpp>
#include <hpx/actions_base/actions_base_support.hpp>
#include <hpx/async_distributed/continuation.hpp>
#include <hpx/async_distributed/traits/action_trigger_continuation.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/traits/needs_automatic_registration.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/threading_base/thread_init_data.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace actions {

    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct transfer_continuation_action : transfer_base_action<Action>
    {
    public:
        transfer_continuation_action(
            transfer_continuation_action const&) = delete;
        transfer_continuation_action(transfer_continuation_action&&) = delete;
        transfer_continuation_action& operator=(
            transfer_continuation_action const&) = delete;
        transfer_continuation_action& operator=(
            transfer_continuation_action&&) = delete;

    public:
        using base_type = transfer_base_action<Action>;
        using continuation_type = typename base_type::continuation_type;

        // construct an empty transfer_continuation_action to avoid serialization
        // overhead
        transfer_continuation_action() = default;

        ~transfer_continuation_action() noexcept override;

        // construct an action from its arguments
        template <typename... Ts>
        explicit transfer_continuation_action(
            continuation_type&& cont, Ts&&... vs);

        template <typename... Ts>
        transfer_continuation_action(threads::thread_priority priority,
            continuation_type&& cont, Ts&&... vs);

        bool has_continuation() const override;

        /// The \a get_thread_function constructs a proper thread function for
        /// a \a thread, encapsulating the functionality and the arguments
        /// of the action it is called for.
        ///
        /// \param lva    [in] This is the local virtual address of the
        ///               component the action has to be invoked on.
        ///
        /// \returns      This function returns a proper thread function usable
        ///               for a \a thread.
        ///
        /// \note This \a get_thread_function will be invoked to retrieve the
        ///       thread function for an action which has to be invoked without
        ///       continuations.
        template <std::size_t... Is>
        threads::thread_function_type get_thread_function(
            util::index_pack<Is...>, hpx::id_type&& target,
            naming::address::address_type lva,
            naming::address::component_type comptype);

        threads::thread_function_type get_thread_function(hpx::id_type&& target,
            naming::address::address_type lva,
            naming::address::component_type comptype) override;

        template <std::size_t... Is>
        void schedule_thread(util::index_pack<Is...>,
            naming::gid_type const& target_gid,
            naming::address::address_type lva,
            naming::address::component_type comptype, std::size_t num_thread);

        // schedule a new thread
        void schedule_thread(naming::gid_type const& target_gid,
            naming::address::address_type lva,
            naming::address::component_type comptype,
            std::size_t num_thread) override;

        // serialization support
        // loading ...
        void load(hpx::serialization::input_archive& ar) override;

        // saving ...
        void save(hpx::serialization::output_archive& ar) override;

        void load_schedule(serialization::input_archive& ar,
            naming::gid_type&& target, naming::address_type lva,
            naming::component_type comptype, std::size_t num_thread,
            bool& deferred_schedule) override;

    private:
        continuation_type cont_;
    };
    /// \endcond

    template <typename Action>
    template <typename... Ts>
    transfer_continuation_action<Action>::transfer_continuation_action(
        continuation_type&& cont, Ts&&... vs)
      : base_type(HPX_FORWARD(Ts, vs)...)
      , cont_(HPX_MOVE(cont))
    {
    }

    template <typename Action>
    template <typename... Ts>
    transfer_continuation_action<Action>::transfer_continuation_action(
        threads::thread_priority priority, continuation_type&& cont, Ts&&... vs)
      : base_type(priority, HPX_FORWARD(Ts, vs)...)
      , cont_(HPX_MOVE(cont))
    {
    }

    template <typename Action>
    bool transfer_continuation_action<Action>::has_continuation() const
    {
        return true;
    }

    template <typename Action>
    template <std::size_t... Is>
    threads::thread_function_type
    transfer_continuation_action<Action>::get_thread_function(
        util::index_pack<Is...>, hpx::id_type&& target,
        naming::address::address_type lva,
        naming::address::component_type comptype)
    {
        return base_type::derived_type::construct_thread_function(
            HPX_MOVE(target), HPX_MOVE(cont_), lva, comptype,
            hpx::get<Is>(HPX_MOVE(this->arguments_))...);
    }

    template <typename Action>
    threads::thread_function_type
    transfer_continuation_action<Action>::get_thread_function(
        hpx::id_type&& target, naming::address::address_type lva,
        naming::address::component_type comptype)
    {
        return get_thread_function(
            typename util::make_index_pack<Action::arity>::type(),
            HPX_MOVE(target), lva, comptype);
    }

    template <typename Action>
    template <std::size_t... Is>
    void transfer_continuation_action<Action>::schedule_thread(
        util::index_pack<Is...>, naming::gid_type const& target_gid,
        naming::address::address_type lva,
        naming::address::component_type comptype, std::size_t /*num_thread*/)
    {
        hpx::id_type target;
        if (naming::detail::has_credits(target_gid))
        {
            target = hpx::id_type(
                target_gid, hpx::id_type::management_type::managed);
        }

        threads::thread_init_data data;
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
        data.description = actions::detail::get_action_name<Action>();
#endif
#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
        data.parent_id = this->parent_id_;
        data.parent_locality_id = this->parent_locality_;
#endif
#if defined(HPX_HAVE_APEX)
        data.timer_data = hpx::util::external_timer::new_task(
            data.description, data.parent_locality_id, data.parent_id);
#endif
        hpx::detail::post_helper<typename base_type::derived_type>::call(
            HPX_MOVE(data), HPX_MOVE(cont_), target, lva, comptype,
            this->priority_, HPX_MOVE(hpx::get<Is>(this->arguments_))...);
    }

    template <typename Action>
    void transfer_continuation_action<Action>::schedule_thread(
        naming::gid_type const& target_gid, naming::address::address_type lva,
        naming::address::component_type comptype, std::size_t num_thread)
    {
        schedule_thread(typename util::make_index_pack<Action::arity>::type(),
            target_gid, lva, comptype, num_thread);

        // keep track of number of invocations
        this->increment_invocation_count();
    }

    template <typename Action>
    void transfer_continuation_action<Action>::load(
        hpx::serialization::input_archive& ar)
    {
        this->load_base(ar);
        ar >> cont_;
    }

    template <typename Action>
    void transfer_continuation_action<Action>::save(
        hpx::serialization::output_archive& ar)
    {
        this->save_base(ar);
        ar << cont_;
    }

    template <typename Action>
    void transfer_continuation_action<Action>::load_schedule(
        serialization::input_archive& ar, naming::gid_type&& target,
        naming::address_type lva, naming::component_type comptype,
        std::size_t num_thread, bool& deferred_schedule)
    {
        // First, serialize, then schedule
        load(ar);

        if (deferred_schedule)
        {
            // If this is a direct action and deferred schedule was requested,
            // that is we are not the last parcel, return immediately
            if (base_type::direct_execution::value)
            {
                return;
            }
            else
            {
                // If this is not a direct action, we can safely set deferred_schedule
                // to false
                deferred_schedule = false;
            }
        }

        schedule_thread(HPX_MOVE(target), lva, comptype, num_thread);
    }

    template <typename Action>
    base_action* detail::register_action<Action>::create_cont()
    {
        return new transfer_continuation_action<Action>{};
    }

    template <typename Action>
    transfer_continuation_action<
        Action>::~transfer_continuation_action() noexcept
    {
        // make sure proper register action function is instantiated
        auto* ptr = &detail::register_action<Action>::create_cont;
        (void) ptr;
    }
}}    // namespace hpx::actions

namespace hpx { namespace traits {
    /// \cond NOINTERNAL
    template <typename Action>
    struct needs_automatic_registration<
        hpx::actions::transfer_continuation_action<Action>>
      : needs_automatic_registration<Action>
    {
    };
    /// \endcond
}}    // namespace hpx::traits

#include <hpx/config/warnings_suffix.hpp>

#endif
