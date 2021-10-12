//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/constexpr.hpp>
#include <hpx/execution_base/operation_state.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/tag_dispatch.hpp>
#include <hpx/functional/traits/is_invocable.hpp>
#include <hpx/type_support/equality.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace execution { namespace experimental {
#if defined(DOXYGEN)
    /// connect is a customization point object.
    /// For some subexpression `s` and `r`, let `S` be the type such that `decltype((s))`
    /// is `S` and let `R` be the type such that `decltype((r))` is `R`. The result of
    /// the expression `hpx::execution::experimental::connect(s, r)` is then equivalent to:
    ///     * `s.connect(r)`, if that expression is valid and returns a type
    ///       satisfying the `operation_state`
    ///       (\see hpx::execution::experimental::traits::is_operation_state)
    ///       and if `S` satisfies the `sender` concept.
    ///     * `s.connect(r)`, if that expression is valid and returns a type
    ///       satisfying the `operation_state`
    ///       (\see hpx::execution::experimental::traits::is_operation_state)
    ///       and if `S` satisfies the `sender` concept.
    ///       Overload resolution is performed in a context that include the declaration
    ///       `void connect();`
    ///     * Otherwise, the expression is ill-formed.
    ///
    /// The customization is implemented in terms of
    /// `hpx::functional::tag_dispatch`.
    template <typename S, typename R>
    void connect(S&& s, R&& r);

    /// The name schedule denotes a customization point object. For some
    /// subexpression s, let S be decltype((s)). The expression schedule(s) is
    /// expression-equivalent to:
    ///
    ///     * s.schedule(), if that expression is valid and its type models
    ///       sender.
    ///     * Otherwise, schedule(s), if that expression is valid and its type
    ///       models sender with overload resolution performed in a context that
    ///       includes the declaration
    ///
    ///           void schedule();
    ///
    ///       and that does not include a declaration of schedule.
    ///
    ///      * Otherwise, schedule(s) is ill-formed.
    ///
    /// The customization is implemented in terms of
    /// `hpx::functional::tag_dispatch`.

#endif

    /// A sender is a type that is describing an asynchronous operation. The
    /// operation itself might not have started yet. In order to get the result
    /// of this asynchronous operation, a sender needs to be connected to a
    /// receiver with the corresponding value, error and done channels:
    ///     * `hpx::execution::experimental::connect`
    ///
    /// In addition, `hpx::execution::experimental::::sender_traits ` needs to
    /// be specialized in some form.
    ///
    /// A sender's destructor shall not block pending completion of submitted
    /// operations.
    template <typename Sender>
    struct is_sender;

    /// \see is_sender
    template <typename Sender, typename Receiver>
    struct is_sender_to;

    /// `sender_traits` expose the different value and error types exposed
    /// by a sender. This can be either specialized directly for user defined
    /// sender types or embedded value_types, error_types and sends_done
    /// inside the sender type can be provided.
    template <typename Sender>
    struct sender_traits;

    template <typename Sender>
    struct sender_traits<Sender volatile> : sender_traits<Sender>
    {
    };
    template <typename Sender>
    struct sender_traits<Sender const> : sender_traits<Sender>
    {
    };
    template <typename Sender>
    struct sender_traits<Sender&> : sender_traits<Sender>
    {
    };
    template <typename Sender>
    struct sender_traits<Sender&&> : sender_traits<Sender>
    {
    };

    namespace detail {
        template <typename Sender>
        constexpr bool specialized(...)
        {
            return true;
        }

        template <typename Sender>
        constexpr bool specialized(
            typename sender_traits<Sender>::__unspecialized*)
        {
            return false;
        }
    }    // namespace detail

    template <typename Sender>
    struct is_sender
      : std::integral_constant<bool,
            std::is_move_constructible<std::decay_t<Sender>>::value &&
                detail::specialized<std::decay_t<Sender>>(nullptr)>
    {
    };

    template <typename Sender>
    HPX_INLINE_CONSTEXPR_VARIABLE bool is_sender_v = is_sender<Sender>::value;

    struct invocable_archetype
    {
        void operator()() {}
    };

    namespace detail {
        template <typename Executor, typename F, typename Enable = void>
        struct is_executor_of_base_impl : std::false_type
        {
        };

        template <typename Executor, typename F>
        struct is_executor_of_base_impl<Executor, F,
            std::enable_if_t<hpx::is_invocable<std::decay_t<F>&>::value &&
                std::is_constructible<std::decay_t<F>, F>::value &&
                std::is_destructible<std::decay_t<F>>::value &&
                std::is_move_constructible<std::decay_t<F>>::value &&
                std::is_copy_constructible<Executor>::value &&
                hpx::traits::is_equality_comparable<Executor>::value>>
          : std::true_type
        {
        };

        template <typename Executor>
        struct is_executor_base
          : is_executor_of_base_impl<std::decay_t<Executor>,
                invocable_archetype>
        {
        };
    }    // namespace detail

    namespace detail {
        template <typename S, typename R, typename Enable = void>
        struct has_member_connect : std::false_type
        {
        };

        template <typename S, typename R>
        struct has_member_connect<S, R,
            typename hpx::util::always_void<decltype(std::declval<S>().connect(
                std::declval<R>()))>::type> : std::true_type
        {
        };
    }    // namespace detail

    HPX_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE
    struct connect_t : hpx::functional::tag<connect_t>
    {
    } connect{};

    namespace detail {
        template <typename S, typename R, typename Enable = void>
        struct connect_result_helper
        {
            struct dummy_operation_state
            {
            };
            using type = dummy_operation_state;
        };

        template <typename S, typename R>
        struct connect_result_helper<S, R,
            std::enable_if_t<hpx::is_invocable<connect_t, S, R>::value>>
          : hpx::util::invoke_result<connect_t, S, R>
        {
        };
    }    // namespace detail

    namespace detail {
        template <typename F, typename E>
        struct as_receiver
        {
            F f;

            void set_value() noexcept(noexcept(HPX_INVOKE(f, )))
            {
                HPX_INVOKE(f, );
            }

            template <typename E_>
            HPX_NORETURN void set_error(E_&&) noexcept
            {
                std::terminate();
            }

            void set_done() noexcept {}
        };
    }    // namespace detail

    namespace detail {
        template <typename S, typename Enable = void>
        struct has_member_schedule : std::false_type
        {
        };

        template <typename S>
        struct has_member_schedule<S,
            typename hpx::util::always_void<decltype(
                std::declval<S>().schedule())>::type> : std::true_type
        {
        };
    }    // namespace detail

    HPX_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE
    struct schedule_t : hpx::functional::tag<schedule_t>
    {
    } schedule{};

    namespace detail {
        template <bool IsSenderReceiver, typename Sender, typename Receiver>
        struct is_sender_to_impl;

        template <typename Sender, typename Receiver>
        struct is_sender_to_impl<false, Sender, Receiver> : std::false_type
        {
        };

        template <typename Sender, typename Receiver>
        struct is_sender_to_impl<true, Sender, Receiver>
          : std::integral_constant<bool,
                hpx::is_invocable_v<connect_t, Sender&&, Receiver&&> ||
                    hpx::is_invocable_v<connect_t, Sender&&, Receiver&> ||
                    hpx::is_invocable_v<connect_t, Sender&&, Receiver const&> ||
                    hpx::is_invocable_v<connect_t, Sender&, Receiver&&> ||
                    hpx::is_invocable_v<connect_t, Sender&, Receiver&> ||
                    hpx::is_invocable_v<connect_t, Sender&, Receiver const&> ||
                    hpx::is_invocable_v<connect_t, Sender const&, Receiver&&> ||
                    hpx::is_invocable_v<connect_t, Sender const&, Receiver&> ||
                    hpx::is_invocable_v<connect_t, Sender const&,
                        Receiver const&>>
        {
        };
    }    // namespace detail

    template <typename Sender, typename Receiver>
    struct is_sender_to
      : detail::is_sender_to_impl<
            is_sender_v<Sender> && is_receiver_v<Receiver>, Sender, Receiver>
    {
    };

    namespace detail {
        template <typename... As>
        struct tuple_mock;
        template <typename... As>
        struct variant_mock;

        template <typename Sender>
        constexpr bool has_value_types(
            typename Sender::template value_types<tuple_mock, variant_mock>*)
        {
            return true;
        }

        template <typename Sender>
        constexpr bool has_value_types(...)
        {
            return false;
        }

        template <typename Sender>
        constexpr bool has_error_types(
            typename Sender::template error_types<variant_mock>*)
        {
            return true;
        }

        template <typename Sender>
        constexpr bool has_error_types(...)
        {
            return false;
        }

        template <typename Sender>
        constexpr bool has_sends_done(decltype(Sender::sends_done)*)
        {
            return true;
        }

        template <typename Sender>
        constexpr bool has_sends_done(...)
        {
            return false;
        }

        template <typename Sender>
        struct has_sender_types
          : std::integral_constant<bool,
                has_value_types<Sender>(nullptr) &&
                    has_error_types<Sender>(nullptr) &&
                    has_sends_done<Sender>(nullptr)>
        {
        };

        template <bool HasSenderTraits, typename Sender>
        struct sender_traits_base;

        template <typename Sender>
        struct sender_traits_base<true /* HasSenderTraits */, Sender>
        {
            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types =
                typename Sender::template value_types<Tuple, Variant>;

            template <template <typename...> class Variant>
            using error_types = typename Sender::template error_types<Variant>;

            static constexpr bool sends_done = Sender::sends_done;
        };

        template <typename Sender>
        struct sender_traits_base<false /* HasSenderTraits */, Sender>
        {
            using __unspecialized = void;
        };

        template <typename Sender>
        struct is_typed_sender
          : std::integral_constant<bool,
                is_sender<Sender>::value &&
                    detail::has_sender_types<Sender>::value>
        {
        };
    }    // namespace detail

    template <typename Sender>
    struct sender_traits
      : detail::sender_traits_base<detail::has_sender_types<Sender>::value,
            Sender>
    {
    };

    // Explicitly specialize for void to avoid forming references to void
    // (is_invocable is in the base implementation, which forms a reference to
    // the Sender type).
    template <>
    struct sender_traits<void>
    {
        using __unspecialized = void;
    };

    namespace detail {
        template <template <typename...> class Tuple,
            template <typename...> class Variant>
        struct value_types
        {
            template <typename Sender>
            struct apply
            {
                using type =
                    typename hpx::execution::experimental::sender_traits<
                        Sender>::template value_types<Tuple, Variant>;
            };
        };

        template <template <typename...> class Variant>
        struct error_types
        {
            template <typename Sender>
            struct apply
            {
                using type =
                    typename hpx::execution::experimental::sender_traits<
                        Sender>::template error_types<Variant>;
            };
        };
    }    // namespace detail

    template <typename Scheduler, typename Enable = void>
    struct is_scheduler : std::false_type
    {
    };

    template <typename Scheduler>
    struct is_scheduler<Scheduler,
        std::enable_if_t<hpx::is_invocable<schedule_t, Scheduler>::value &&
            std::is_copy_constructible<Scheduler>::value &&
            hpx::traits::is_equality_comparable<Scheduler>::value>>
      : std::true_type
    {
    };

    template <typename Scheduler>
    HPX_INLINE_CONSTEXPR_VARIABLE bool is_scheduler_v =
        is_scheduler<Scheduler>::value;

    template <typename S, typename R>
    using connect_result_t =
        typename hpx::util::invoke_result<connect_t, S, R>::type;
}}}    // namespace hpx::execution::experimental
