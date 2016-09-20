//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file transfer_continuation_action.hpp

#ifndef HPX_RUNTIME_ACTIONS_TRANSFER_CONTINUATION_ACTION_HPP
#define HPX_RUNTIME_ACTIONS_TRANSFER_CONTINUATION_ACTION_HPP

#include <hpx/config.hpp>
#if defined(HPX_HAVE_SECURITY)
#include <hpx/traits/action_capability_provider.hpp>
#endif
#include <hpx/runtime/actions/continuation.hpp>
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

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct transfer_continuation_action : transfer_base_action<Action>
    {
        HPX_MOVABLE_ONLY(transfer_continuation_action);

        typedef transfer_base_action<Action> base_type;
        typedef typename base_type::continuation_type continuation_type;
    public:
        // construct an empty transfer_continuation_action to avoid serialization overhead
        transfer_continuation_action();

        // construct an action from its arguments
        template <typename ...Ts>
        explicit transfer_continuation_action(continuation_type&& cont, Ts&&... vs);

        template <typename ...Ts>
        transfer_continuation_action(
            threads::thread_priority priority, continuation_type&& cont,
            Ts&&... vs);

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
        get_thread_function(naming::id_type&& target, naming::address::address_type lva);

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
            std::size_t num_thread);

    private:
        continuation_type cont_;
    };
    /// \endcond

    template <typename Action>
    transfer_continuation_action<Action>::transfer_continuation_action()
    {}

    template <typename Action>
    template <typename ...Ts>
    transfer_continuation_action<Action>::transfer_continuation_action(continuation_type&& cont, Ts&&... vs)
      : base_type(std::forward<Ts>(vs)...)
      , cont_(std::move(cont))
    {}

    template <typename Action>
    template <typename ...Ts>
    transfer_continuation_action<Action>::transfer_continuation_action(threads::thread_priority priority, continuation_type&& cont, Ts&&... vs)
      : base_type(priority, std::forward<Ts>(vs)...)
      , cont_(std::move(cont))
    {}

    template <typename Action>
    bool transfer_continuation_action<Action>::has_continuation() const
    {
        return true;
    }

    template <typename Action>
    template <std::size_t ...Is>
    threads::thread_function_type
    transfer_continuation_action<Action>::get_thread_function(util::detail::pack_c<std::size_t, Is...>,
    naming::id_type&& target, naming::address::address_type lva)
    {
        return base_type::derived_type::construct_thread_function(std::move(target),
            std::move(cont_), lva, util::get<Is>(std::move(this->arguments_))...);
    }

    template <typename Action>
    threads::thread_function_type
    transfer_continuation_action<Action>::get_thread_function(naming::id_type&& target, naming::address::address_type lva)
    {
        return get_thread_function(
            typename util::detail::make_index_pack<Action::arity>::type(),
            std::move(target), lva);
    }

    template <typename Action>
    template <std::size_t ...Is>
    void
    transfer_continuation_action<Action>::schedule_thread(util::detail::pack_c<std::size_t, Is...>,
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
            std::move(data), std::move(cont_), target, lva, this->priority_,
            util::get<Is>(std::move(this->arguments_))...);
    }

    template <typename Action>
    void transfer_continuation_action<Action>::schedule_thread(naming::gid_type const& target_gid,
        naming::address::address_type lva,
        std::size_t num_thread)
    {
        schedule_thread(
            typename util::detail::make_index_pack<Action::arity>::type(),
            target_gid, lva, num_thread);

        // keep track of number of invocations
        this->increment_invocation_count();
    }

    template <typename Action>
    void transfer_continuation_action<Action>::load(hpx::serialization::input_archive & ar)
    {
        this->load_base(ar);
        ar >> cont_;
    }

    template <typename Action>
    void transfer_continuation_action<Action>::save(hpx::serialization::output_archive & ar)
    {
        this->save_base(ar);
        ar << cont_;
    }

    template <typename Action>
    void transfer_continuation_action<Action>::load_schedule(serialization::input_archive& ar,
        naming::gid_type&& target, naming::address_type lva,
        std::size_t num_thread)
    {
        // First, serialize, then schedule
        load(ar);
        schedule_thread(std::move(target), lva, num_thread);
    }
}}

namespace hpx { namespace traits
{
    /// \cond NOINTERNAL
    template <typename Action>
    struct needs_automatic_registration<
        hpx::actions::transfer_continuation_action<Action> >
      : needs_automatic_registration<Action>
    {};
    /// \endcond
}}

#endif /*HPX_RUNTIME_ACTIONS_TRANSFER_ACTION_HPP*/
