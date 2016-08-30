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
#if defined(HPX_HAVE_SECURITY)
#include <hpx/traits/action_capability_provider.hpp>
#endif
#include <hpx/runtime/actions_fwd.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/actions/detail/invocation_count_registry.hpp>
#include <hpx/runtime/components/pinned_ptr.hpp>
#include <hpx/runtime/get_locality_id.hpp>
#include <hpx/runtime/parcelset/detail/per_action_data_counter_registry.hpp>
#include <hpx/runtime/serialization/base_object.hpp>
#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/traits/action_decorate_continuation.hpp>
#include <hpx/traits/action_does_termination_detection.hpp>
#include <hpx/traits/action_message_handler.hpp>
#include <hpx/traits/action_priority.hpp>
#include <hpx/traits/action_schedule_thread.hpp>
#include <hpx/traits/action_serialization_filter.hpp>
#include <hpx/traits/action_stacksize.hpp>
#include <hpx/traits/action_was_object_migrated.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/get_and_reset_value.hpp>
#include <hpx/util/serialize_exception.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/atomic.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace hpx { namespace actions
{
    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct transfer_action : base_action
    {
        HPX_MOVABLE_ONLY(transfer_action);

    public:
        typedef typename Action::component_type component_type;
        typedef typename Action::derived_type derived_type;
        typedef typename Action::result_type result_type;
        typedef typename Action::arguments_type arguments_type;

        // This is the priority value this action has been instantiated with
        // (statically). This value might be different from the priority member
        // holding the runtime value an action has been created with
        enum { priority_value = traits::action_priority<Action>::value };

        // This is the stacksize value this action has been instantiated with
        // (statically). This value might be different from the stacksize member
        // holding the runtime value an action has been created with
        enum { stacksize_value = traits::action_stacksize<Action>::value };

        typedef typename Action::direct_execution direct_execution;
        typedef void serialized_with_id;

        // construct an empty transfer_action to avoid serialization overhead
        transfer_action()
        {}

        // construct an action from its arguments
        template <typename ...Ts>
        explicit transfer_action(Ts&&... vs)
          : arguments_(std::forward<Ts>(vs)...),
#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
            parent_locality_(transfer_action::get_locality_id()),
            parent_id_(reinterpret_cast<std::uint64_t>(threads::get_parent_id())),
            parent_phase_(threads::get_parent_phase()),
#endif
            priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(threads::thread_priority_default)),
            stacksize_(
                detail::thread_stacksize<
                    static_cast<threads::thread_stacksize>(stacksize_value)
                >::call(threads::thread_stacksize_default))
        {}

        template <typename ...Ts>
        transfer_action(threads::thread_priority priority, Ts&&... vs)
          : arguments_(std::forward<Ts>(vs)...),
#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
            parent_locality_(transfer_action::get_locality_id()),
            parent_id_(reinterpret_cast<std::uint64_t>(threads::get_parent_id())),
            parent_phase_(threads::get_parent_phase()),
#endif
            priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority)),
            stacksize_(
                detail::thread_stacksize<
                    static_cast<threads::thread_stacksize>(stacksize_value)
                >::call(threads::thread_stacksize_default))
        {}

        //
        ~transfer_action()
        {
        }

    public:
        /// retrieve component type
        static int get_static_component_type()
        {
            return derived_type::get_component_type();
        }

    private:
        /// The function \a get_component_type returns the \a component_type
        /// of the component this action belongs to.
        int get_component_type() const
        {
            return derived_type::get_component_type();
        }

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* get_action_name() const
        {
            return detail::get_action_name<derived_type>();
        }

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        action_type get_action_type() const
        {
            return derived_type::get_action_type();
        }

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
            naming::address::address_type lva)
        {
            return derived_type::construct_thread_function(lva,
                util::get<Is>(std::move(arguments_))...);
        }

        threads::thread_function_type
        get_thread_function(naming::address::address_type lva)
        {
            return get_thread_function(
                typename util::detail::make_index_pack<Action::arity>::type(),
                lva);
        }

        /// The \a get_thread_function constructs a proper thread function for
        /// a \a thread, encapsulating the functionality, the arguments, and
        /// the continuations of the action it is called for.
        ///
        /// \param cont   [in] This is the list of continuations to be
        ///               triggered after the execution of the action
        /// \param lva    [in] This is the local virtual address of the
        ///               component the action has to be invoked on.
        ///
        /// \returns      This function returns a proper thread function usable
        ///               for a \a thread.
        ///
        /// \note This \a get_thread_function will be invoked to retrieve the
        ///       thread function for an action which has to be invoked with
        ///       continuations.
        template <std::size_t ...Is>
        threads::thread_function_type
        get_thread_function(util::detail::pack_c<std::size_t, Is...>,
            std::unique_ptr<continuation> cont, naming::address::address_type lva)
        {
            return derived_type::construct_thread_function(std::move(cont), lva,
                util::get<Is>(std::move(arguments_))...);
        }

        threads::thread_function_type
        get_thread_function(std::unique_ptr<continuation> cont,
            naming::address::address_type lva)
        {
            return get_thread_function(
                typename util::detail::make_index_pack<Action::arity>::type(),
                std::move(cont), lva);
        }

#if !defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
        /// Return the locality of the parent thread
        std::uint32_t get_parent_locality_id() const
        {
            return naming::invalid_locality_id;
        }

        /// Return the thread id of the parent thread
        threads::thread_id_repr_type get_parent_thread_id() const
        {
            return threads::invalid_thread_id_repr;
        }

        /// Return the phase of the parent thread
        std::uint64_t get_parent_thread_phase() const
        {
            return 0;
        }
#else
        /// Return the locality of the parent thread
        std::uint32_t get_parent_locality_id() const
        {
            return parent_locality_;
        }

        /// Return the thread id of the parent thread
        threads::thread_id_repr_type get_parent_thread_id() const
        {
            return reinterpret_cast<threads::thread_id_repr_type>(parent_id_);
        }

        /// Return the phase of the parent thread
        std::uint64_t get_parent_thread_phase() const
        {
            return parent_phase_;
        }
#endif

        /// Return the thread priority this action has to be executed with
        threads::thread_priority get_thread_priority() const
        {
            return priority_;
        }

        /// Return the thread stacksize this action has to be executed with
        threads::thread_stacksize get_thread_stacksize() const
        {
            return stacksize_;
        }

        /// Return whether the embedded action is part of termination detection
        bool does_termination_detection() const
        {
            return traits::action_does_termination_detection<derived_type>::call();
        }

        /// Return all data needed for thread initialization
        threads::thread_init_data&
        get_thread_init_data(naming::id_type&& target,
            naming::address::address_type lva, threads::thread_init_data& data)
        {
            data.func = get_thread_function(lva);
#if defined(HPX_HAVE_THREAD_TARGET_ADDRESS)
            data.lva = lva;
#endif
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            data.description = detail::get_action_name<derived_type>();
#endif
#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
            data.parent_id =
                reinterpret_cast<threads::thread_id_repr_type>(parent_id_);
            data.parent_locality_id = parent_locality_;
#endif
            data.priority = priority_;
            data.stacksize = threads::get_stack_size(stacksize_);

            data.target = std::move(target);
            return data;
        }

        threads::thread_init_data&
        get_thread_init_data(std::unique_ptr<continuation> cont,
            naming::id_type&& target,
            naming::address::address_type lva, threads::thread_init_data& data)
        {
            data.func = get_thread_function(std::move(cont), lva);
#if defined(HPX_HAVE_THREAD_TARGET_ADDRESS)
            data.lva = lva;
#endif
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            data.description = detail::get_action_name<derived_type>();
#endif
#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
            data.parent_id =
                reinterpret_cast<threads::thread_id_repr_type>(parent_id_);
            data.parent_locality_id = parent_locality_;
#endif
            data.priority = priority_;
            data.stacksize = threads::get_stack_size(stacksize_);

            data.target = std::move(target);
            return data;
        }

        // schedule a new thread
        void schedule_thread(naming::gid_type const& target_gid,
            naming::address::address_type lva,
            threads::thread_state_enum initial_state,
            std::size_t num_thread)
        {
            std::unique_ptr<continuation> cont;
            threads::thread_init_data data;
            data.num_os_thread = num_thread;

            naming::id_type target;
            if (naming::detail::has_credits(target_gid))
            {
                target = naming::id_type(target_gid, naming::id_type::managed);
            }
            else
            {
                target = naming::id_type(target_gid, naming::id_type::unmanaged);
            }

            if (traits::action_decorate_continuation<derived_type>::call(cont))
            {
                traits::action_schedule_thread<derived_type>::call(lva,
                    get_thread_init_data(std::move(cont), std::move(target), lva, data),
                    initial_state);

            }
            else
            {
                traits::action_schedule_thread<derived_type>::call(lva,
                    get_thread_init_data(std::move(target), lva, data), initial_state);
            }

            // keep track of number of invocations
            increment_invocation_count();
        }

        void schedule_thread(std::unique_ptr<continuation> cont,
            naming::gid_type const& target_gid, naming::address::address_type lva,
            threads::thread_state_enum initial_state,
            std::size_t num_thread)
        {
            // first decorate the continuation
            traits::action_decorate_continuation<derived_type>::call(cont);

            // now, schedule the thread
            threads::thread_init_data data;
            data.num_os_thread = num_thread;

            naming::id_type target;
            if (naming::detail::has_credits(target_gid))
            {
                target = naming::id_type(target_gid, naming::id_type::managed);
            }
            else
            {
                target = naming::id_type(target_gid, naming::id_type::unmanaged);
            }

            traits::action_schedule_thread<derived_type>::call(lva,
                get_thread_init_data(std::move(cont), std::move(target), lva, data),
                initial_state);


            // keep track of number of invocations
            increment_invocation_count();
        }

        /// Return whether the given object was migrated
        std::pair<bool, components::pinned_ptr>
            was_object_migrated(hpx::naming::gid_type const& id,
                naming::address::address_type lva)
        {
            return traits::action_was_object_migrated<derived_type>::call(id, lva);
        }

        /// Return a pointer to the filter to be used while serializing an
        /// instance of this action type.
        serialization::binary_filter* get_serialization_filter(
            parcelset::parcel const& p) const
        {
            return traits::action_serialization_filter<derived_type>::call(p);
        }

        /// Return a pointer to the message handler to be used for this action.
        parcelset::policies::message_handler* get_message_handler(
            parcelset::parcelhandler* ph, parcelset::locality const& loc,
            parcelset::parcel const& p) const
        {
            return traits::action_message_handler<derived_type>::
                call(ph, loc, p);
        }

#if defined(HPX_HAVE_SECURITY)
        /// Return the set of capabilities required to invoke this action
        components::security::capability get_required_capabilities(
            naming::address::address_type lva) const
        {
            return traits::action_capability_provider<derived_type>::call(lva);
        }
#endif
    public:
        /// retrieve the N's argument
        template <std::size_t N>
        inline typename util::tuple_element<N, arguments_type>::type const&
        get() const
        {
            return util::get<N>(arguments_);
        }

        // serialization support
        // loading ...
        void serialize(hpx::serialization::input_archive & ar)
        {
            ar >> arguments_;

            // Always serialize the parent information to maintain binary
            // compatibility on the wire.

            detail::action_serialization_data data;
            ar >> data;

#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
            parent_locality_ = data.parent_locality_;
            parent_id_ = data.parent_id_;
            parent_phase_ = data.parent_phase_;
#endif
            priority_ = data.priority_;
            stacksize_ = data.stacksize_;
        }

        // saving ...
        void serialize(hpx::serialization::output_archive & ar)
        {
            ar << arguments_;

            // Always serialize the parent information to maintain binary
            // compatibility on the wire.

#if !defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
            std::uint32_t parent_locality_ = naming::invalid_locality_id;
            std::uint64_t parent_id_ = std::uint64_t(-1);
            std::uint64_t parent_phase_ = 0;
#endif
            detail::action_serialization_data data(parent_locality_,
                parent_id_, parent_phase_, priority_, stacksize_);
            ar << data;
        }

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & hpx::serialization::base_object<base_action>(*this);
            serialize(ar);
        }
        HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME(
            transfer_action, detail::get_action_name<derived_type>())

        /// Extract the current invocation count for this action
        static std::int64_t get_invocation_count(bool reset)
        {
            return util::get_and_reset_value(invocation_count_, reset);
        }

    private:
        static std::uint32_t get_locality_id()
        {
            error_code ec(lightweight);      // ignore any errors
            return hpx::get_locality_id(ec);
        }

    protected:
        arguments_type arguments_;

#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
        std::uint32_t parent_locality_;
        std::uint64_t parent_id_;
        std::uint64_t parent_phase_;
#endif
        threads::thread_priority priority_;
        threads::thread_stacksize stacksize_;

    private:
        static boost::atomic<std::int64_t> invocation_count_;

    protected:
        static void increment_invocation_count()
        {
            ++invocation_count_;
        }
    };

    template <typename Action>
    boost::atomic<std::int64_t>
        transfer_action<Action>::invocation_count_(0);

    namespace detail
    {
        template <typename Action>
        void register_remote_action_invocation_count(
            invocation_count_registry& registry)
        {
            registry.register_class(
                hpx::actions::detail::get_action_name<Action>(),
                &transfer_action<Action>::get_invocation_count
            );
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <std::size_t N, typename Action>
    inline typename util::tuple_element<
        N, typename transfer_action<Action>::arguments_type
    >::type const& get(transfer_action<Action> const& args)
    {
        return args.template get<N>();
    }

    /// \endcond
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

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
namespace hpx { namespace parcelset { namespace detail
{
    /// \cond NOINTERNAL
    template <typename Action>
    void register_per_action_data_counter_types(
        per_action_data_counter_registry& registry)
    {
        registry.register_class(
            hpx::actions::detail::get_action_name<Action>()
        );
    }
    /// \endcond
}}}
#endif

#endif /*HPX_RUNTIME_ACTIONS_TRANSFER_ACTION_HPP*/
