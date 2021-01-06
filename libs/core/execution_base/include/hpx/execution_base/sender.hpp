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
#include <hpx/functional/tag_priority_invoke.hpp>
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
    /// From some subexpression `s` and `r`, let `S` be the type such that `decltype((s))`
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
    ///     * Otherwise: TODO once executor is in place...
    ///     * Otherwise, the expression is ill-formed.
    ///
    /// The customization is implemented in terms of `hpx::function::tag_invoke`
    template <typename S, typename R>
    void connect(S&& s, R&& r);
#endif

    namespace traits {
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
                typename hpx::execution::experimental::traits::sender_traits<
                    Sender>::__unspecialized*)
            {
                return false;
            }
        }    // namespace detail

        template <typename Sender>
        struct is_sender
          : std::integral_constant<bool,
                std::is_move_constructible<
                    typename std::decay<Sender>::type>::value &&
                    detail::specialized<Sender>(nullptr)>
        {
        };

        template <typename Sender>
        constexpr bool is_sender_v = is_sender<Sender>::value;

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
                typename std::enable_if<hpx::traits::is_invocable<typename std::
                                                decay<F>::type&>::value &&
                    std::is_constructible<typename std::decay<F>::type,
                        F>::value &&
                    std::is_destructible<typename std::decay<F>::type>::value &&
                    std::is_move_constructible<
                        typename std::decay<F>::type>::value &&
                    std::is_copy_constructible<Executor>::value &&
                    hpx::traits::is_equality_comparable<Executor>::value>::type>
              : std::true_type
            {
            };

            template <typename Executor>
            struct is_executor_base
              : is_executor_of_base_impl<typename std::decay<Executor>::type,
                    invocable_archetype>
            {
            };
        }    // namespace detail
    }        // namespace traits

    HPX_INLINE_CONSTEXPR_VARIABLE struct execute_t
      : hpx::functional::tag_priority<execute_t>
    {
        template <typename Executor, typename F>
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(execute_t,
            Executor&& executor,
            F&& f) noexcept(noexcept(std::forward<Executor>(executor)
                                         .execute(std::forward<F>(f)))) ->
            typename std::enable_if<hpx::traits::is_invocable<
                                        typename std::decay<F>::type&>::value &&
                    (traits::is_sender_v<Executor> ||
                        traits::detail::is_executor_base<Executor>::value),
                decltype(std::forward<Executor>(executor).execute(
                    std::forward<F>(f)))>::type
        {
            return std::forward<Executor>(executor).execute(std::forward<F>(f));
        }
    } execute;

    namespace detail {
        template <typename R, typename>
        struct as_invocable
        {
            R* r;

            explicit as_invocable(R& r) noexcept
              : r(std::addressof(r))
            {
            }

            as_invocable(as_invocable&& other) noexcept
              : r(std::exchange(other.r, nullptr))
            {
            }

            ~as_invocable()
            {
                if (r)
                {
                    execution::experimental::set_done(std::move(*r));
                }
            }

            void operator()() & noexcept
            try
            {
                execution::experimental::set_value(std::move(*r));
                r = nullptr;
            }
            catch (...)
            {
                execution::experimental::set_error(
                    std::move(*r), std::current_exception());
                r = nullptr;
            }
        };

        template <typename S, typename R>
        struct as_operation
        {
            typename std::decay<S>::type e;
            typename std::decay<R>::type r;

            void start() noexcept
            try
            {
                hpx::execution::experimental::execute(std::move(e),
                    as_invocable<typename std::decay<R>::type, S>{r});
            }
            catch (...)
            {
                hpx::execution::experimental::set_error(
                    std::move(r), std::current_exception());
            }
        };

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

    HPX_INLINE_CONSTEXPR_VARIABLE struct connect_t
      : hpx::functional::tag_priority<connect_t>
    {
        template <typename S, typename R>
        friend constexpr HPX_FORCEINLINE auto
        tag_override_invoke(connect_t, S&& s, R&& r) noexcept(
            noexcept(std::declval<S&&>().connect(std::forward<R>(r)))) ->
            typename std::enable_if<traits::is_sender_v<S> &&
                    traits::is_receiver_v<R>,
                decltype(std::declval<S&&>().connect(std::forward<R>(r)))>::type
        {
            static_assert(
                hpx::execution::experimental::traits::is_operation_state_v<
                    decltype(std::declval<S&&>().connect(std::forward<R>(r)))>,
                "hpx::execution::experimental::connect needs to return a "
                "type satisfying the operation_state concept");

            return std::forward<S>(s).connect(std::forward<R>(r));
        }

        template <typename S, typename R>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(connect_t,
            S&& s, R&& r) noexcept(noexcept(detail::as_operation<S, R>{
            std::forward<S>(s), std::forward<R>(r)})) ->
            typename std::enable_if<!detail::has_member_connect<S, R>::value &&
                    traits::is_receiver_of_v<R> &&
                    traits::detail::is_executor_of_base_impl<
                        typename std::decay<S>::type,
                        detail::as_invocable<typename std::decay<R>::type,
                            S>>::value,
                decltype(detail::as_operation<S, R>{
                    std::forward<S>(s), std::forward<R>(r)})>::type
        {
            return detail::as_operation<S, R>{
                std::forward<S>(s), std::forward<R>(r)};
        }
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
            typename std::enable_if<
                hpx::is_invocable<connect_t, S, R>::value>::type>
          : hpx::util::invoke_result<hpx::execution::experimental::connect_t, S,
                R>
        {
        };

        template <typename S, typename R>
        struct submit_state
        {
            struct submit_receiver
            {
                submit_state* p;

                template <typename... Ts>
                    void set_value(Ts&&... ts) &&
                    noexcept(traits::is_nothrow_receiver_of_v<R, Ts...>)
                {
                    hpx::execution::experimental::set_value(
                        std::move(p->r), std::forward<Ts>(ts)...);
                    delete p;
                }

                template <typename E>
                    void set_error(E&& e) && noexcept
                {
                    hpx::execution::experimental::set_error(
                        std::move(p->r), std::forward<E>(e));
                    delete p;
                }

                void set_done() && noexcept
                {
                    hpx::execution::experimental::set_done(std::move(p->r));
                    delete p;
                }
            };

            typename std::decay<R>::type r;
            typename connect_result_helper<S, submit_receiver>::type state;

            submit_state(S&& s, R&& r)
              : r(std::forward<R>(r))
              , state(hpx::execution::experimental::connect(
                    std::forward<S>(s), submit_receiver{this}))
            {
            }
        };

        template <typename S, typename R, typename Enable = void>
        struct has_member_submit : std::false_type
        {
        };

        template <typename S, typename R>
        struct has_member_submit<S, R,
            typename hpx::util::always_void<decltype(std::declval<S>().submit(
                std::declval<R>()))>::type> : std::true_type
        {
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct submit_t
      : hpx::functional::tag_priority<submit_t>
    {
        template <typename S, typename R>
        friend constexpr HPX_FORCEINLINE auto
        tag_override_invoke(submit_t, S&& s, R&& r) noexcept(
            noexcept(std::forward<S>(s).submit(std::forward<R>(r)))) ->
            typename std::enable_if<traits::is_sender_to<S, R>::value,
                decltype(std::forward<S>(s).submit(std::forward<R>(r)))>::type
        {
            std::forward<S>(s).submit(std::forward<R>(r));
        }

        template <typename S, typename R>
        friend constexpr HPX_FORCEINLINE auto
        tag_fallback_invoke(submit_t, S&& s, R&& r) noexcept(
            noexcept(hpx::execution::experimental::start(
                (new detail::submit_state<S, R>{
                     std::forward<S>(s), std::forward<R>(r)})
                    ->state))) ->
            typename std::enable_if<!detail::has_member_submit<S, R>::value,
                typename hpx::util::always_void<decltype(
                    hpx::execution::experimental::start(
                        (new detail::submit_state<S, R>{
                             std::forward<S>(s), std::forward<R>(r)})
                            ->state))>::type>::type
        {
            hpx::execution::experimental::start(
                (new detail::submit_state<S, R>{
                     std::forward<S>(s), std::forward<R>(r)})
                    ->state);
        }
    } submit;

    namespace detail {
        template <typename F, typename E>
        struct as_receiver
        {
            F f;

            void set_value() noexcept(noexcept(HPX_INVOKE(f)))
            {
                HPX_INVOKE(f);
            }

            template <typename E_>
            HPX_NORETURN void set_error(E_&&) noexcept
            {
                std::terminate();
            }

            void set_done() noexcept {}
        };

        template <typename S, typename R, typename Enable = void>
        struct has_member_execute : std::false_type
        {
        };

        template <typename S, typename R>
        struct has_member_execute<S, R,
            typename hpx::util::always_void<decltype(std::declval<S>().execute(
                std::declval<R>()))>::type> : std::true_type
        {
        };
    }    // namespace detail

    template <typename Executor, typename F>
    constexpr HPX_FORCEINLINE auto
    tag_fallback_invoke(execute_t, Executor&& executor, F&& f) noexcept(
        noexcept(hpx::execution::experimental::submit(
            std::forward<Executor>(executor),
            detail::as_receiver<typename std::decay<F>::type, Executor>{
                std::forward<F>(f)}))) ->
        typename std::enable_if<
            hpx::traits::is_invocable<typename std::decay<F>::type&>::value &&
                !detail::has_member_execute<Executor, F>::value,
            decltype(hpx::execution::experimental::submit(
                std::forward<Executor>(executor),
                detail::as_receiver<typename std::decay<F>::type, Executor>{
                    std::forward<F>(f)}))>::type
    {
        return hpx::execution::experimental::submit(
            std::forward<Executor>(executor),
            detail::as_receiver<typename std::decay<F>::type, Executor>{
                std::forward<F>(f)});
    }

    namespace traits {
        namespace detail {
            template <typename Executor, typename F, typename Enable = void>
            struct is_executor_of_impl : std::false_type
            {
            };

            template <typename Executor, typename F>
            struct is_executor_of_impl<Executor, F,
                typename std::enable_if<hpx::traits::is_invocable<execute_t,
                    Executor, F>::value>::type>
              : is_executor_of_base_impl<Executor, F>
            {
            };
        }    // namespace detail

        template <typename Executor>
        struct is_executor
          : detail::is_executor_of_base_impl<Executor, invocable_archetype>
        {
        };

        template <typename Executor, typename F>
        struct is_executor_of : detail::is_executor_of_base_impl<Executor, F>
        {
        };

        template <typename Executor>
        constexpr bool is_executor_v = is_executor<Executor>::value;

        template <typename Executor, typename F>
        constexpr bool is_executor_of_v = is_executor_of<Executor, F>::value;
    }    // namespace traits

    namespace detail {
        template <typename Executor, typename F, typename N,
            typename Enable = void>
        struct has_member_bulk_execute : std::false_type
        {
        };

        template <typename Executor, typename F, typename N>
        struct has_member_bulk_execute<Executor, F, N,
            typename hpx::util::always_void<decltype(
                std::declval<Executor>().bulk_execute(
                    std::declval<F>(), std::size_t{}))>::type> : std::true_type
        {
        };
    }    // namespace detail

    // TODO: P0443 is conflicting on whether this returns void or a sender.
    HPX_INLINE_CONSTEXPR_VARIABLE struct bulk_execute_t
      : hpx::functional::tag_priority<bulk_execute_t>
    {
        template <typename Executor, typename F, typename N>
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(
            bulk_execute_t, Executor&& executor, F&& f,
            N n) noexcept(noexcept(std::forward<Executor>(executor)
                                       .bulk_execute(std::forward<F>(f)),
            static_cast<std::size_t>(n))) ->
            typename std::enable_if<hpx::traits::is_invocable<F, N>::value &&
                    std::is_convertible<N, std::size_t>::value,
                decltype(std::forward<Executor>(executor).bulk_execute(
                    std::forward<F>(f), static_cast<std::size_t>(n)))>::type
        {
            return std::forward<Executor>(executor).bulk_execute(
                std::forward<F>(f), static_cast<std::size_t>(n));
        }

        template <typename Executor, typename F, typename N>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            bulk_execute_t, Executor&& executor, F&& f,
            N n) noexcept(noexcept(std::forward<Executor>(executor)
                                       .bulk_execute(std::forward<F>(f)),
            n)) ->
            typename std::enable_if<hpx::traits::is_invocable<F, N>::value &&
                    std::is_convertible<N, std::size_t>::value &&
                    !detail::has_member_bulk_execute<Executor, F, N>::value,
                decltype(std::forward<Executor>(executor).bulk_execute(
                    std::forward<F>(f), n))>::type
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                hpx::execution::experimental::execute(
                    executor, [f, n]() { HPX_INVOKE(f, n); });
            }
        }
    } bulk_execute;

    namespace detail {
        template <typename Executor>
        struct as_sender
        {
        private:
            Executor exec;

        public:
            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types = Variant<Tuple<>>;

            template <template <typename...> class Variant>
            using error_types = Variant<std::exception_ptr>;

            static constexpr bool sends_done = true;

            explicit as_sender(Executor exec) noexcept
              : exec(exec)
            {
            }

            template <typename R>
            auto connect(R&& r) && ->
                typename std::enable_if<traits::is_receiver_of_v<R>,
                    decltype(hpx::execution::experimental::connect(
                        std::move(exec), std::forward<R>(r)))>::type
            {
                return hpx::execution::experimental::connect(
                    std::move(exec), std::forward<R>(r));
            }

            template <typename R>
            auto connect(R&& r) const& ->
                typename std::enable_if<traits::is_receiver_of_v<R>,
                    decltype(hpx::execution::experimental::connect(
                        exec, std::forward<R>(r)))>::type
            {
                return hpx::execution::experimental::connect(
                    exec, std::forward<R>(r));
            }
        };

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

    HPX_INLINE_CONSTEXPR_VARIABLE struct schedule_t
      : hpx::functional::tag_priority<schedule_t>
    {
        template <typename S>
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(
            schedule_t, S&& s) noexcept(noexcept(std::forward<S>(s).schedule()))
            -> typename std::enable_if<traits::is_sender_v<S>,
                decltype(std::forward<S>(s).schedule())>::type
        {
            return std::forward<S>(s).schedule();
        }

        template <typename S>
        friend constexpr HPX_FORCEINLINE auto
        tag_fallback_invoke(schedule_t, S&& s) noexcept(
            noexcept(detail::as_sender<typename std::decay<S>::type>{
                std::forward<S>(s)})) ->
            typename std::enable_if<!detail::has_member_schedule<S>::value &&
                    traits::is_executor_v<S>,
                decltype(detail::as_sender<typename std::decay<S>::type>{
                    std::forward<S>(s)})>::type
        {
            return detail::as_sender<typename std::decay<S>::type>{
                std::forward<S>(s)};
        }
    } schedule{};

    namespace traits {
        namespace detail {
            template <bool IsSenderReceiver, typename Sender, typename Receiver>
            struct is_sender_to_impl;

            template <typename Sender, typename Receiver>
            struct is_sender_to_impl<false, Sender, Receiver> : std::false_type
            {
            };

            // clang-format off
            template <typename Sender, typename Receiver>
            struct is_sender_to_impl<true, Sender, Receiver>
              : std::integral_constant<bool,
                    hpx::is_invocable_v<
                        hpx::execution::experimental::connect_t,
                            Sender&&, Receiver&&> ||
                    hpx::is_invocable_v<
                        hpx::execution::experimental::connect_t,
                            Sender&&, Receiver&> ||
                    hpx::is_invocable_v<
                        hpx::execution::experimental::connect_t,
                            Sender&&, Receiver const&> ||
                    hpx::is_invocable_v<
                        hpx::execution::experimental::connect_t,
                            Sender&, Receiver&&> ||
                    hpx::is_invocable_v<
                        hpx::execution::experimental::connect_t,
                            Sender&, Receiver&> ||
                    hpx::is_invocable_v<
                        hpx::execution::experimental::connect_t,
                            Sender&, Receiver const&> ||
                    hpx::is_invocable_v<
                        hpx::execution::experimental::connect_t,
                            Sender const&, Receiver&&> ||
                    hpx::is_invocable_v<
                        hpx::execution::experimental::connect_t,
                            Sender const&, Receiver&> ||
                    hpx::is_invocable_v<
                        hpx::execution::experimental::connect_t,
                            Sender const&, Receiver const&>>
            {
            };
            // clang-format on
        }    // namespace detail

        template <typename Sender, typename Receiver>
        struct is_sender_to
          : detail::is_sender_to_impl<is_sender_v<Sender> &&
                    is_receiver_v<Receiver>,
                Sender, Receiver>
        {
        };

        namespace detail {
            template <typename... As>
            struct tuple_mock;
            template <typename... As>
            struct variant_mock;

            template <typename Sender>
            constexpr bool has_value_types(
                typename Sender::template value_types<tuple_mock,
                    variant_mock>*)
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
                using error_types =
                    typename Sender::template error_types<Variant>;

                static constexpr bool sends_done = Sender::sends_done;
            };

            struct void_receiver
            {
                void set_value() noexcept;
                void set_error(std::exception_ptr) noexcept;
                void set_done() noexcept;
            };

            template <typename Sender, typename Enable = void>
            struct sender_traits_executor_base : std::false_type
            {
                using __unspecialized = void;
            };

            template <typename Sender>
            struct sender_traits_executor_base<Sender,
                typename std::enable_if<is_executor_of_base_impl<Sender,
                    hpx::execution::experimental::detail::as_invocable<
                        void_receiver, Sender>>::value>::type> : std::false_type
            {
                template <template <class...> class Tuple,
                    template <class...> class Variant>
                using value_types = Variant<Tuple<>>;

                template <template <class...> class Variant>
                using error_types = Variant<std::exception_ptr>;

                static constexpr bool sends_done = true;
            };

            template <typename Sender>
            struct sender_traits_base<false /* HasSenderTraits */, Sender>
              : sender_traits_executor_base<Sender>
            {
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

        template <typename Scheduler, typename Enable = void>
        struct is_scheduler : std::false_type
        {
        };

        template <typename Scheduler>
        struct is_scheduler<Scheduler,
            typename std::enable_if<
                hpx::traits::is_invocable<schedule_t, Scheduler&&>::value &&
                std::is_copy_constructible<Scheduler>::value &&
                hpx::traits::is_equality_comparable<Scheduler>::value>::type>
          : std::true_type
        {
        };
    }    // namespace traits
}}}      // namespace hpx::execution::experimental
