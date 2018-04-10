//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2011-2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file transfer_action.hpp

#ifndef HPX_RUNTIME_ACTIONS_TRANSFER_BASE_ACTION_HPP
#define HPX_RUNTIME_ACTIONS_TRANSFER_BASE_ACTION_HPP

#include <hpx/runtime/actions_fwd.hpp>

#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/actions/base_action.hpp>
#include <hpx/runtime/actions/detail/invocation_count_registry.hpp>
#include <hpx/runtime/components/pinned_ptr.hpp>
#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>
#include <hpx/runtime/serialization/unique_ptr.hpp>
#include <hpx/traits/action_does_termination_detection.hpp>
#include <hpx/traits/action_message_handler.hpp>
#include <hpx/traits/action_priority.hpp>
#include <hpx/traits/action_schedule_thread.hpp>
#include <hpx/traits/action_serialization_filter.hpp>
#include <hpx/traits/action_stacksize.hpp>
#include <hpx/traits/action_was_object_migrated.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/get_and_reset_value.hpp>
#include <hpx/util/serialize_exception.hpp>
#include <hpx/util/tuple.hpp>
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
#include <hpx/util/itt_notify.hpp>
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    // If one or more arguments of the action are non-default-constructible,
    // the transfer_action does not store the argument tuple directly but a
    // unique_ptr to the tuple instead.
    namespace detail
    {
        template <typename Args>
        struct argument_holder
        {
            argument_holder() = default;

            explicit argument_holder(Args && args)
              : data_(new Args(std::move(args)))
            {}

            template <typename ... Ts>
            argument_holder(Ts && ... ts)
              : data_(new Args(std::forward<Ts>(ts)...))
            {}

            template <typename Archive>
            void serialize(Archive& ar, unsigned int const)
            {
                ar & data_;
            }

            HPX_HOST_DEVICE HPX_FORCEINLINE
            Args& data()
            {
                HPX_ASSERT(!!data_);
                return *data_;
            }

#if defined(HPX_DISABLE_ASSERTS) || defined(BOOST_DISABLE_ASSERTS) || defined(NDEBUG)
            HPX_CONSTEXPR HPX_HOST_DEVICE HPX_FORCEINLINE
            Args const& data() const
            {
                return *data_;
            }
#else
            HPX_HOST_DEVICE HPX_FORCEINLINE
            Args const& data() const
            {
                HPX_ASSERT(!!data_);
                return *data_;
            }
#endif

        private:
            std::unique_ptr<Args> data_;
        };
    }
}}

namespace hpx { namespace util
{
    template <std::size_t I, typename Args>
    HPX_CONSTEXPR HPX_HOST_DEVICE HPX_FORCEINLINE
    typename util::tuple_element<I, Args>::type&
    get(hpx::actions::detail::argument_holder<Args>& t)
    {
        return util::tuple_element<I, Args>::get(t.data());
    }

    template <std::size_t I, typename Args>
    HPX_CONSTEXPR HPX_HOST_DEVICE HPX_FORCEINLINE
    typename util::tuple_element<I, Args>::type const&
    get(hpx::actions::detail::argument_holder<Args> const& t)
    {
        return util::tuple_element<I, Args>::get(t.data());
    }

    template <std::size_t I, typename Args>
    HPX_CONSTEXPR HPX_HOST_DEVICE HPX_FORCEINLINE
    typename util::tuple_element<I, Args>::type&&
    get(hpx::actions::detail::argument_holder<Args>&& t)
    {
        return std::forward<typename util::tuple_element<I, Args>::type>(
            util::get<I>(t.data()));
    }

    template <std::size_t I, typename Args>
    HPX_CONSTEXPR HPX_HOST_DEVICE HPX_FORCEINLINE
    typename util::tuple_element<I, Args>::type const&&
    get(hpx::actions::detail::argument_holder<Args> const&& t)
    {
        return std::forward<
                typename util::tuple_element<I, Args>::type const
            >(util::get<I>(t.data()));
    }
}}

namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct transfer_base_action : base_action_data
    {
    public:
        HPX_NON_COPYABLE(transfer_base_action);

    public:
        typedef typename Action::component_type component_type;
        typedef typename Action::derived_type derived_type;
        typedef typename Action::result_type result_type;
        typedef typename Action::arguments_type arguments_base_type;
        typedef typename std::conditional<
                std::is_constructible<arguments_base_type>::value,
                    arguments_base_type,
                    detail::argument_holder<arguments_base_type>
            >::type arguments_type;
        typedef typename Action::continuation_type continuation_type;

        // This is the priority value this action has been instantiated with
        // (statically). This value might be different from the priority member
        // holding the runtime value an action has been created with
        HPX_STATIC_CONSTEXPR std::uint32_t priority_value =
            traits::action_priority<Action>::value;

        // This is the stacksize value this action has been instantiated with
        // (statically). This value might be different from the stacksize member
        // holding the runtime value an action has been created with
        HPX_STATIC_CONSTEXPR std::uint32_t stacksize_value =
            traits::action_stacksize<Action>::value;

        typedef typename Action::direct_execution direct_execution;

        // construct an empty transfer_action to avoid serialization overhead
        transfer_base_action() = default;

        // construct an action from its arguments
        template <typename... Ts>
        explicit transfer_base_action(Ts&&... vs)
          : base_action_data(
                detail::thread_priority<static_cast<threads::thread_priority>(
                    priority_value)>::call(threads::thread_priority_default),
                detail::thread_stacksize<static_cast<threads::thread_stacksize>(
                    stacksize_value)>::call(threads::thread_stacksize_default))
          , arguments_(std::forward<Ts>(vs)...)
        {}

        template <typename ...Ts>
        transfer_base_action(threads::thread_priority priority, Ts&&... vs)
          : base_action_data(
                detail::thread_priority<static_cast<threads::thread_priority>(
                    priority_value)>::call(priority),
                detail::thread_stacksize<static_cast<threads::thread_stacksize>(
                    stacksize_value)>::call(threads::thread_stacksize_default))
          , arguments_(std::forward<Ts>(vs)...)
        {}

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
        action_type get_action_type() const override
        {
            return derived_type::get_action_type();
        }

        /// Return whether the embedded action is part of termination detection
        bool does_termination_detection() const override
        {
            return traits::action_does_termination_detection<derived_type>::call();
        }

        /// Return whether the given object was migrated
        std::pair<bool, components::pinned_ptr>
            was_object_migrated(hpx::naming::gid_type const& id,
                naming::address::address_type lva) override
        {
            return traits::action_was_object_migrated<derived_type>::call(id, lva);
        }

        /// Return a pointer to the filter to be used while serializing an
        /// instance of this action type.
        serialization::binary_filter* get_serialization_filter(
            parcelset::parcel const& p) const override
        {
            return traits::action_serialization_filter<derived_type>::call(p);
        }

        /// Return a pointer to the message handler to be used for this action.
        parcelset::policies::message_handler* get_message_handler(
            parcelset::parcelhandler* ph, parcelset::locality const& loc,
            parcelset::parcel const& p) const override
        {
            return traits::action_message_handler<derived_type>::
                call(ph, loc, p);
        }

    public:
        /// retrieve the N's argument
        template <std::size_t N>
        HPX_CONSTEXPR inline
        typename util::tuple_element<N, arguments_type>::type const&
        get() const
        {
            return util::get<N>(arguments_);
        }

        /// Extract the current invocation count for this action
        static std::int64_t get_invocation_count(bool reset)
        {
            return util::get_and_reset_value(invocation_count_, reset);
        }

        // serialization support
        // loading ...
        void load_base(hpx::serialization::input_archive & ar)
        {
            ar >> arguments_;
            this->base_action_data::load_base(ar);
        }

        // saving ...
        void save_base(hpx::serialization::output_archive & ar)
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
    std::atomic<std::int64_t>
        transfer_base_action<Action>::invocation_count_(0);

    namespace detail
    {
        template <typename Action>
        void register_remote_action_invocation_count(
            invocation_count_registry& registry)
        {
            registry.register_class(
                hpx::actions::detail::get_action_name<Action>(),
                &transfer_base_action<Action>::get_invocation_count
            );
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <std::size_t N, typename Action>
    HPX_CONSTEXPR inline typename util::tuple_element<
        N, typename transfer_action<Action>::arguments_type
    >::type const& get(transfer_base_action<Action> const& args)
    {
        return args.template get<N>();
    }
}}

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
#include <hpx/runtime/parcelset/detail/per_action_data_counter_registry.hpp>

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

#endif
