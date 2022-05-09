//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2011-2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file transfer_action.hpp

#pragma once

#include <hpx/config/defines.hpp>

#include <hpx/actions/actions_fwd.hpp>
#include <hpx/actions/base_action.hpp>
#include <hpx/actions/register_action.hpp>
#include <hpx/actions_base/actions_base_support.hpp>
#include <hpx/actions_base/detail/invocation_count_registry.hpp>
#include <hpx/actions_base/traits/action_continuation.hpp>
#include <hpx/actions_base/traits/action_does_termination_detection.hpp>
#include <hpx/actions_base/traits/action_priority.hpp>
#include <hpx/actions_base/traits/action_schedule_thread.hpp>
#include <hpx/actions_base/traits/action_stacksize.hpp>
#include <hpx/actions_base/traits/action_trigger_continuation_fwd.hpp>
#include <hpx/actions_base/traits/action_was_object_migrated.hpp>
#include <hpx/components_base/pinned_ptr.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/assert.hpp>
#include <hpx/datastructures/serialization/tuple.hpp>
#include <hpx/modules/datastructures.hpp>
#include <hpx/modules/runtime_local.hpp>
#include <hpx/modules/serialization.hpp>
#include <hpx/modules/util.hpp>
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
#include <hpx/modules/itt_notify.hpp>
#endif

#include <hpx/parcelset_base/traits/action_get_embedded_parcel.hpp>
#include <hpx/parcelset_base/traits/action_message_handler.hpp>
#include <hpx/parcelset_base/traits/action_serialization_filter.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace actions {

    ///////////////////////////////////////////////////////////////////////////
    // If one or more arguments of the action are non-default-constructible,
    // the transfer_action does not store the argument tuple directly but a
    // unique_ptr to the tuple instead.
    namespace detail {

        template <typename Args>
        struct argument_holder
        {
            argument_holder() = default;

            explicit argument_holder(Args&& args)
              : data_(new Args(HPX_MOVE(args)))
            {
            }

            template <typename... Ts>
            argument_holder(Ts&&... ts)
              : data_(new Args(HPX_FORWARD(Ts, ts)...))
            {
            }

            template <typename Archive>
            void serialize(Archive& ar, unsigned int const)
            {
                // clang-format off
                ar & data_;
                // clang-format on
            }

            HPX_HOST_DEVICE HPX_FORCEINLINE Args& data()
            {
                HPX_ASSERT(!!data_);
                return *data_;
            }

            HPX_HOST_DEVICE HPX_FORCEINLINE Args const& data() const
            {
                HPX_ASSERT(!!data_);
                return *data_;
            }

        private:
            std::unique_ptr<Args> data_;
        };
    }    // namespace detail
}}       // namespace hpx::actions

namespace hpx {

    template <std::size_t I, typename Args>
    constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
        typename hpx::tuple_element<I, Args>::type&
        get(hpx::actions::detail::argument_holder<Args>& t)
    {
        return hpx::tuple_element<I, Args>::get(t.data());
    }

    template <std::size_t I, typename Args>
    constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
        typename hpx::tuple_element<I, Args>::type const&
        get(hpx::actions::detail::argument_holder<Args> const& t)
    {
        return hpx::tuple_element<I, Args>::get(t.data());
    }

    template <std::size_t I, typename Args>
    constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
        typename hpx::tuple_element<I, Args>::type&&
        get(hpx::actions::detail::argument_holder<Args>&& t)
    {
        return std::forward<typename hpx::tuple_element<I, Args>::type>(
            hpx::get<I>(t.data()));
    }

    template <std::size_t I, typename Args>
    constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
        typename hpx::tuple_element<I, Args>::type const&&
        get(hpx::actions::detail::argument_holder<Args> const&& t)
    {
        return std::forward<typename hpx::tuple_element<I, Args>::type const>(
            hpx::get<I>(t.data()));
    }
}    // namespace hpx

namespace hpx { namespace actions {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct transfer_base_action : base_action_data
    {
    public:
        transfer_base_action(transfer_base_action const&) = delete;
        transfer_base_action(transfer_base_action&&) = delete;
        transfer_base_action& operator=(transfer_base_action const&) = delete;
        transfer_base_action& operator=(transfer_base_action&&) = delete;

    public:
        using component_type = typename Action::component_type;
        using derived_type = typename Action::derived_type;
        using result_type = typename Action::result_type;
        using arguments_base_type = typename Action::arguments_type;
        using arguments_type = typename std::conditional<
            std::is_constructible<arguments_base_type>::value,
            arguments_base_type,
            detail::argument_holder<arguments_base_type>>::type;

        using continuation_type =
            typename traits::action_continuation<Action>::type;

        // This is the priority value this action has been instantiated with
        // (statically). This value might be different from the priority member
        // holding the runtime value an action has been created with
        static constexpr threads::thread_priority priority_value =
            traits::action_priority<Action>::value;

        // This is the stacksize value this action has been instantiated with
        // (statically). This value might be different from the stacksize member
        // holding the runtime value an action has been created with
        static constexpr threads::thread_stacksize stacksize_value =
            traits::action_stacksize<Action>::value;

        using direct_execution = typename Action::direct_execution;

        // construct an empty transfer_action to avoid serialization overhead
        transfer_base_action() = default;

        // construct an action from its arguments
        template <typename... Ts>
        explicit transfer_base_action(Ts&&... vs)
          : base_action_data(threads::thread_priority::default_,
                threads::thread_stacksize::default_)
          , arguments_(HPX_FORWARD(Ts, vs)...)
        {
        }

        template <typename... Ts>
        transfer_base_action(threads::thread_priority priority, Ts&&... vs)
          : base_action_data(priority, threads::thread_stacksize::default_)
          , arguments_(HPX_FORWARD(Ts, vs)...)
        {
        }

        //
        ~transfer_base_action() noexcept override
        {
            detail::register_action<derived_type>::instance.instantiate();
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
        int get_component_type() const override
        {
            return derived_type::get_component_type();
        }

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* get_action_name() const override
        {
            return detail::get_action_name<derived_type>();
        }

        /// The function \a get_serialization_id returns the id which has been
        /// associated with this action (mainly used for serialization purposes).
        std::uint32_t get_action_id() const override
        {
            return detail::get_action_id<derived_type>();
        }

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
        /// The function \a get_action_name_itt returns the name of this action
        /// as a ITT string_handle
        util::itt::string_handle const& get_action_name_itt() const override
        {
            return detail::get_action_name_itt<derived_type>();
        }
#endif

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        action_flavor get_action_type() const override
        {
            return derived_type::get_action_type();
        }

        /// Return whether the embedded action is part of termination detection
        bool does_termination_detection() const override
        {
            return traits::action_does_termination_detection<
                derived_type>::call();
        }

        /// Return whether the given object was migrated
        std::pair<bool, components::pinned_ptr> was_object_migrated(
            hpx::naming::gid_type const& id,
            naming::address::address_type lva) override
        {
            return traits::action_was_object_migrated<derived_type>::call(
                id, lva);
        }

        /// Return a pointer to the filter to be used while serializing an
        /// instance of this action type.
        serialization::binary_filter* get_serialization_filter() const override
        {
            return traits::action_serialization_filter<derived_type>::call();
        }

        /// Return an embedded parcel if available (e.g. routing action).
        hpx::optional<parcelset::parcel> get_embedded_parcel() const override
        {
            return traits::action_get_embedded_parcel<Action>::call(*this);
        }

        /// Return a pointer to the message handler to be used for this action.
        parcelset::policies::message_handler* get_message_handler(
            parcelset::locality const& loc) const override
        {
            return traits::action_message_handler<derived_type>::call(loc);
        }

    public:
        /// retrieve the N's argument
        template <std::size_t N>
        constexpr inline
            typename hpx::tuple_element<N, arguments_type>::type const&
            get() const
        {
            return hpx::get<N>(arguments_);
        }

        /// Extract the current invocation count for this action
        static std::int64_t get_invocation_count(bool reset)
        {
            return util::get_and_reset_value(invocation_count_, reset);
        }

        // serialization support
        // loading ...
        void load_base(hpx::serialization::input_archive& ar)
        {
            ar >> arguments_;
            this->base_action_data::load_base(ar);
        }

        // saving ...
        void save_base(hpx::serialization::output_archive& ar)
        {
            ar << arguments_;
            this->base_action_data::save_base(ar);
        }

    protected:
        arguments_type arguments_;

    private:
        static std::atomic<std::int64_t> invocation_count_;

    protected:
        static void increment_invocation_count()
        {
            ++invocation_count_;
        }
    };

    template <typename Action>
    std::atomic<std::int64_t> transfer_base_action<Action>::invocation_count_(
        0);

    namespace detail {
        template <typename Action>
        void register_remote_action_invocation_count(
            invocation_count_registry& registry)
        {
            registry.register_class(
                hpx::actions::detail::get_action_name<Action>(),
                &transfer_base_action<Action>::get_invocation_count);
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <std::size_t N, typename Action>
    constexpr inline typename hpx::tuple_element<N,
        typename transfer_action<Action>::arguments_type>::type const&
    get(transfer_base_action<Action> const& args)
    {
        return args.template get<N>();
    }
}}    // namespace hpx::actions

#if defined(HPX_HAVE_PARCELPORT_COUNTERS) &&                                   \
    defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
#include <hpx/actions_base/detail/per_action_data_counter_registry.hpp>

namespace hpx { namespace actions { namespace detail {
    /// \cond NOINTERNAL
    template <typename Action>
    void register_per_action_data_counter_types(
        per_action_data_counter_registry& registry)
    {
        registry.register_class(
            hpx::actions::detail::get_action_name<Action>());
    }
    /// \endcond
}}}       // namespace hpx::actions::detail
#endif    // HPX_HAVE_PARCELPORT_ACTION_COUNTERS

#endif    // HPX_HAVE_NETWORKING
