//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file transfer_action.hpp

#ifndef HPX_RUNTIME_ACTIONS_TRANSFER_ACTION_HPP
#define HPX_RUNTIME_ACTIONS_TRANSFER_ACTION_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/actions/transfer_base_action.hpp>
#include <hpx/runtime/applier/apply_helper.hpp>
#include <hpx/runtime/parcelset/detail/per_action_data_counter_registry.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/util/detail/pack.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace hpx { namespace actions
{
    /// \cond NOINTERNAL

// MSVC12 reports: warning C4520: 'hpx::actions::transfer_action<Action>' :
// multiple default constructors specified
#if defined(HPX_MSVC_WARNING_PRAGMA) && HPX_MSVC < 1900
#pragma warning(push)
#pragma warning (disable: 4520)
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct transfer_action : transfer_base_action<Action>
    {
        HPX_MOVABLE_ONLY(transfer_action);

        typedef transfer_base_action<Action> base_type;

    public:
        // construct an empty transfer_action to avoid serialization overhead
        transfer_action();

        // construct an action from its arguments
        template <typename ...Ts>
        explicit transfer_action(Ts&&... vs);

        template <typename ...Ts>
        transfer_action(threads::thread_priority priority, Ts&&... vs);

        bool has_continuation() const;

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
        template <std::size_t ...Is>
        threads::thread_function_type
        get_thread_function(util::detail::pack_c<std::size_t, Is...>,
            naming::id_type&& target, naming::address::address_type lva);

        threads::thread_function_type
        get_thread_function(naming::id_type&& target,
            naming::address::address_type lva);

        template <std::size_t ...Is>
        void
        schedule_thread(util::detail::pack_c<std::size_t, Is...>,
            naming::gid_type const& target_gid,
            naming::address::address_type lva,
            std::size_t num_thread);

        // schedule a new thread
        void schedule_thread(naming::gid_type const& target_gid,
            naming::address::address_type lva,
            std::size_t num_thread);

        // serialization support
        // loading ...
        void load(hpx::serialization::input_archive & ar);

        // saving ...
        void save(hpx::serialization::output_archive & ar);

        void load_schedule(serialization::input_archive& ar,
            naming::gid_type&& target, naming::address_type lva,
            std::size_t num_thread, bool& deferred_schedule);
    };
    /// \endcond

    template <typename Action>
    transfer_action<Action>::transfer_action()
    {}

    template <typename Action>
    template <typename ...Ts>
    transfer_action<Action>::transfer_action(Ts&&... vs)
      : base_type(std::forward<Ts>(vs)...)
    {}

    template <typename Action>
    template <typename ...Ts>
    transfer_action<Action>::transfer_action(
            threads::thread_priority priority, Ts&&... vs)
      : base_type(priority, std::forward<Ts>(vs)...)
    {}

    template <typename Action>
    bool transfer_action<Action>::has_continuation() const
    {
        return false;
    }

    template <typename Action>
    template <std::size_t ...Is>
    threads::thread_function_type
    transfer_action<Action>::get_thread_function(
        util::detail::pack_c<std::size_t, Is...>,
        naming::id_type&& target, naming::address::address_type lva)
    {
        return base_type::derived_type::construct_thread_function(
            std::move(target), lva,
            util::get<Is>(std::move(this->arguments_))...);
    }

    template <typename Action>
    threads::thread_function_type
    transfer_action<Action>::get_thread_function(
        naming::id_type&& target, naming::address::address_type lva)
    {
        return get_thread_function(
            typename util::detail::make_index_pack<Action::arity>::type(),
            std::move(target), lva);
    }

    template <typename Action>
    template <std::size_t ...Is>
    void
    transfer_action<Action>::schedule_thread(
            util::detail::pack_c<std::size_t, Is...>,
        naming::gid_type const& target_gid,
        naming::address::address_type lva,
        std::size_t num_thread)
    {
        naming::id_type target;
        if (naming::detail::has_credits(target_gid))
        {
            target = naming::id_type(target_gid, naming::id_type::managed);
        }

        threads::thread_init_data data;
#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
        data.parent_id =
            reinterpret_cast<threads::thread_id_repr_type>(this->parent_id_);
        data.parent_locality_id = this->parent_locality_;
#endif
        applier::detail::apply_helper<typename base_type::derived_type>::call(
            std::move(data), target, lva, this->priority_,
            std::move(util::get<Is>(this->arguments_))...);
    }

    template <typename Action>
    void transfer_action<Action>::schedule_thread(
        naming::gid_type const& target_gid,
        naming::address::address_type lva,
        std::size_t num_thread)
    {
        schedule_thread(
            typename util::detail::make_index_pack<Action::arity>::type(),
            target_gid, lva, num_thread);

        // keep track of number of invocations
        this->increment_invocation_count();
    }

#if defined(HPX_MSVC_WARNING_PRAGMA) && HPX_MSVC < 1900
#pragma warning(pop)
#endif

    template <typename Action>
    void transfer_action<Action>::load(hpx::serialization::input_archive & ar)
    {
        this->load_base(ar);
    }

    template <typename Action>
    void transfer_action<Action>::save(hpx::serialization::output_archive & ar)
    {
        this->save_base(ar);
    }

    template <typename Action>
    void transfer_action<Action>::load_schedule(serialization::input_archive& ar,
        naming::gid_type&& target, naming::address_type lva,
        std::size_t num_thread, bool& deferred_schedule)
    {
        // First, serialize, then schedule
        load(ar);

        if (deferred_schedule)
        {
            // If this is a direct action and deferred schedule was requested, that
            // is we are not the last parcel, return immediately
            if (base_type::direct_execution::value)
                return;

            // If this is not a direct action, we can safely set deferred_schedule
            // to false
            deferred_schedule = false;
        }

        schedule_thread(std::move(target), lva, num_thread);
    }
}}

namespace hpx { namespace traits
{
    /// \cond NOINTERNAL
    template <typename Action>
    struct needs_automatic_registration<hpx::actions::transfer_action<Action> >
      : needs_automatic_registration<Action>
    {};
    /// \endcond
}}

#endif /*HPX_RUNTIME_ACTIONS_TRANSFER_ACTION_HPP*/
