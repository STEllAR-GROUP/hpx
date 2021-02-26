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
    /// execute is a customization point object.  For some subexpression `e` and
    /// `f`, let `E` be `decltype((e))` and let `F` be `decltype((F))`. The
    /// expression `execute(e, f)` is ill-formed if `F` does not model
    /// `invocable`, or if `E` does not model either `executor` or `sender`. The
    /// result of the expression `hpx::execution::experimental::execute(e, f)`
    /// is then equivalent to:
    ///     * `e.execute(f)`, if that expression is valid. If the function
    ///       selected does not execute the function object `f` on the executor
    ///       `e`, the program is ill-formed with no diagnostic required.
    ///     * Otherwise, `execute(e, f)`, if that expression is valid, with
    ///       overload resolution performed in a context that includes the
    ///       declaration `void execute();` and that does not include a
    ///       declaration of `hpx::execution::experimental::execute`. If the
    ///       function selected by overload resolution does not execute the
    ///       function object `f` on the executor `e`, the program is ill-formed
    ///       with no diagnostic required.
    ///     * Otherwise, execution::submit(e, as-receiver<remove_cvref_t<F>,
    ///       E>{forward<F>(f)}) if
    ///       * F is not an instance of as-invocable<R,E'> for some type R where
    ///         E and E' name the same type ignoring cv and reference
    ///         qualifiers, and
    ///       * invocable<remove_cvref_t<F>&> && sender_to<E,
    ///         as-receiver<remove_cvref_t<F>, E>> is true
    ///
    ///       where as-receiver is some implementation-defined class template
    ///       equivalent to:
    ///
    ///           template<class F, class>
    ///           struct as-receiver {
    ///             F f_;
    ///             void set_value() noexcept(is_nothrow_invocable_v<F&>) {
    ///               invoke(f_);
    ///             }
    ///             template<class E>
    ///             [[noreturn]] void set_error(E&&) noexcept {
    ///               terminate();
    ///             }
    ///             void set_done() noexcept {}
    ///           };
    ///
    /// The customization is implemented in terms of `hpx::functional::tag_invoke`.
    template <typename E, typename F>
    void execute(E&& e, F&& f);

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
    ///     * Otherwise, as-operation{s, r}, if
    ///       * r is not an instance of as-receiver<F, S'> for some type F where
    ///         S and S' name the same type ignoring cv and reference
    ///         qualifiers, and
    ///       * receiver_of<R> && executor-of-impl<remove_cvref_t<S>,
    ///         as-invocable<remove_cvref_t<R>, S>> is true,
    ///
    ///       where as-operation is an implementation-defined class equivalent
    ///       to:
    ///
    ///           struct as-operation {
    ///             remove_cvref_t<S> e_;
    ///             remove_cvref_t<R> r_;
    ///             void start() noexcept try {
    ///               execution::execute(std::move(e_), as-invocable<remove_cvref_t<R>, S>{r_});
    ///             } catch(...) {
    ///               execution::set_error(std::move(r_), current_exception());
    ///             }
    ///           };
    ///
    ///       and as-invocable is a class template equivalent to the following:
    ///
    ///           template<class R, class>
    ///           struct as-invocable {
    ///             R* r_;
    ///             explicit as-invocable(R& r) noexcept
    ///               : r_(std::addressof(r)) {}
    ///             as-invocable(as-invocable && other) noexcept
    ///               : r_(std::exchange(other.r_, nullptr)) {}
    ///             ~as-invocable() {
    ///               if(r_)
    ///                 execution::set_done(std::move(*r_));
    ///             }
    ///             void operator()() & noexcept try {
    ///               execution::set_value(std::move(*r_));
    ///               r_ = nullptr;
    ///             } catch(...) {
    ///               execution::set_error(std::move(*r_), current_exception());
    ///               r_ = nullptr;
    ///             }
    ///           };
    ///     * Otherwise, the expression is ill-formed.
    ///
    /// The customization is implemented in terms of
    /// `hpx::functional::tag_invoke`.
    template <typename S, typename R>
    void connect(S&& s, R&& r);

    /// The name submit denotes a customization point object.
    ///
    /// For some subexpressions s and r, let S be decltype((s)) and let R be
    /// decltype((r)). The expression submit(s, r) is ill-formed if
    /// sender_to<S, R> is not true. Otherwise, it is expression-equivalent to:
    ///
    ///     * s.submit(r), if that expression is valid and S models sender. If the
    ///       function selected does not submit the receiver object r via the
    ///       sender s, the program is ill-formed with no diagnostic required.
    ///
    ///     * Otherwise, submit(s, r), if that expression is valid and S models
    ///       sender, with overload resolution performed in a context that
    ///       includes the declaration
    ///
    ///           void submit();
    ///
    ///       and that does not include a declaration of execution::submit. If
    ///       the function selected by overload resolution does not submit the
    ///       receiver object r via the sender s, the program is ill-formed with
    ///       no diagnostic required.
    ///
    ///     * Otherwise, start((newsubmit-state<S, R>{s,r})->state_),
    ///       where submit-state is an implementation-defined class template
    ///       equivalent to
    ///
    ///           template<class S, class R>
    ///           struct submit-state {
    ///             struct submit-receiver {
    ///               submit-state * p_;
    ///               template<class...As>
    ///                 requires receiver_of<R, As...>
    ///               void set_value(As&&... as) && noexcept(is_nothrow_receiver_of_v<R, As...>) {
    ///                 set_value(std::move(p_->r_), (As&&) as...);
    ///                 delete p_;
    ///               }
    ///               template<class E>
    ///                 requires receiver<R, E>
    ///               void set_error(E&& e) && noexcept {
    ///                 set_error(std::move(p_->r_), (E&&) e);
    ///                 delete p_;
    ///               }
    ///               void set_done() && noexcept {
    ///                 set_done(std::move(p_->r_));
    ///                 delete p_;
    ///               }
    ///             };
    ///             remove_cvref_t<R> r_;
    ///             connect_result_t<S, submit-receiver> state_;
    ///             submit-state(S&& s, R&& r)
    ///               : r_((R&&) r)
    ///               , state_(connect((S&&) s, submit-receiver{this})) {}
    ///           };
    ///
    /// The customization is implemented in terms of
    /// `hpx::functional::tag_invoke`.
    template <typename S, typename R>
    auto submit(S&& s, R&& r);

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
    ///     * Otherwise, as-sender<remove_cvref_t<S>>{s} if S satisfies
    ///       executor, where as-sender is an implementation-defined class
    ///       template equivalent to
    ///
    ///           template<class E>
    ///           struct as-sender {
    ///           private:
    ///             E ex_;
    ///           public:
    ///             template<template<class...> class Tuple, template<class...> class Variant>
    ///               using value_types = Variant<Tuple<>>;
    ///             template<template<class...> class Variant>
    ///               using error_types = Variant<std::exception_ptr>;
    ///             static constexpr bool sends_done = true;
    ///
    ///             explicit as-sender(E e) noexcept
    ///               : ex_((E&&) e) {}
    ///             template<class R>
    ///               requires receiver_of<R>
    ///             connect_result_t<E, R> connect(R&& r) && {
    ///               return connect((E&&) ex_, (R&&) r);
    ///             }
    ///             template<class R>
    ///               requires receiver_of<R>
    ///             connect_result_t<const E &, R> connect(R&& r) const & {
    ///               return connect(ex_, (R&&) r);
    ///             }
    ///           };
    ///
    ///      * Otherwise, schedule(s) is ill-formed.
    ///
    /// The customization is implemented in terms of
    /// `hpx::functional::tag_invoke`.

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
            std::enable_if_t<
                hpx::traits::is_invocable<std::decay_t<F>&>::value &&
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

    HPX_INLINE_CONSTEXPR_VARIABLE struct execute_t
      : hpx::functional::tag_priority<execute_t>
    {
        template <typename Executor, typename F,
            typename = std::enable_if_t<
                hpx::traits::is_invocable<std::decay_t<F>&>::value &&
                (is_sender_v<Executor> ||
                    detail::is_executor_base<Executor>::value)>>
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(execute_t,
            Executor&& executor,
            F&& f) noexcept(noexcept(std::forward<Executor>(executor)
                                         .execute(std::forward<F>(f))))
            -> decltype(
                std::forward<Executor>(executor).execute(std::forward<F>(f)))
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

        template <typename F>
        struct is_as_invocable : std::false_type
        {
        };

        template <typename R, typename T>
        struct is_as_invocable<as_invocable<R, T>> : std::true_type
        {
        };

        template <typename S, typename R>
        struct as_operation
        {
            std::decay_t<S> e;
            std::decay_t<R> r;

            void start() noexcept
            try
            {
                execute(std::move(e), as_invocable<std::decay_t<R>, S>{r});
            }
            catch (...)
            {
                set_error(std::move(r), std::current_exception());
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
        template <typename S, typename R,
            typename = std::enable_if_t<is_sender_v<S> && is_receiver_v<R>>>
        friend constexpr HPX_FORCEINLINE auto
        tag_override_invoke(connect_t, S&& s, R&& r) noexcept(
            noexcept(std::forward<S>(s).connect(std::forward<R>(r))))
            -> decltype(std::forward<S>(s).connect(std::forward<R>(r)))
        {
            static_assert(is_operation_state_v<decltype(
                              std::forward<S>(s).connect(std::forward<R>(r)))>,
                "hpx::execution::experimental::connect needs to return a "
                "type satisfying the operation_state concept");

            return std::forward<S>(s).connect(std::forward<R>(r));
        }

        template <typename S, typename R,
            typename =
                std::enable_if_t<!detail::has_member_connect<S, R>::value &&
                    is_receiver_of_v<R> &&
                    detail::is_executor_of_base_impl<std::decay_t<S>,
                        detail::as_invocable<std::decay_t<R>, S>>::value>>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(connect_t,
            S&& s, R&& r) noexcept(noexcept(detail::as_operation<S, R>{
            std::forward<S>(s), std::forward<R>(r)}))
            -> decltype(detail::as_operation<S, R>{
                std::forward<S>(s), std::forward<R>(r)})
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
            std::enable_if_t<hpx::is_invocable<connect_t, S, R>::value>>
          : hpx::util::invoke_result<connect_t, S, R>
        {
        };

        template <typename S, typename R>
        struct submit_state
        {
            struct submit_receiver
            {
                submit_state* p;

                template <typename... Ts,
                    typename = std::enable_if_t<is_receiver_of_v<R, Ts...>>>
                    void set_value(Ts&&... ts) &&
                    noexcept(is_nothrow_receiver_of_v<R, Ts...>)
                {
                    hpx::execution::experimental::set_value(
                        std::move(p->r), std::forward<Ts>(ts)...);
                    delete p;
                }

                template <typename E,
                    typename = std::enable_if_t<is_receiver_v<R, E>>>
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

            std::decay_t<R> r;
            typename connect_result_helper<S, submit_receiver>::type state;

            submit_state(S&& s, R&& r)
              : r(std::forward<R>(r))
              , state(connect(std::forward<S>(s), submit_receiver{this}))
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
        template <typename S, typename R,
            typename = std::enable_if_t<is_sender_to<S, R>::value>>
        friend constexpr HPX_FORCEINLINE auto
        tag_override_invoke(submit_t, S&& s, R&& r) noexcept(
            noexcept(std::forward<S>(s).submit(std::forward<R>(r))))
            -> decltype(std::forward<S>(s).submit(std::forward<R>(r)))
        {
            std::forward<S>(s).submit(std::forward<R>(r));
        }

        template <typename S, typename R,
            typename =
                std::enable_if_t<!detail::has_member_submit<S, R>::value>>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(submit_t,
            S&& s,
            R&& r) noexcept(noexcept(start((new detail::submit_state<S, R>{
                                                std::forward<S>(s),
                                                std::forward<R>(r)})
                                               ->state)))
            -> decltype(start((new detail::submit_state<S, R>{
                                   std::forward<S>(s), std::forward<R>(r)})
                                  ->state))
        {
            start((new detail::submit_state<S, R>{
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

    template <typename Executor, typename F,
        typename = std::enable_if_t<
            hpx::traits::is_invocable<std::decay_t<F>&>::value &&
            !detail::has_member_execute<Executor, F>::value &&
            !detail::is_as_invocable<F>::value>>
    constexpr HPX_FORCEINLINE auto tag_fallback_invoke(execute_t,
        Executor&& executor,
        F&& f) noexcept(noexcept(submit(std::forward<Executor>(executor),
        detail::as_receiver<std::decay_t<F>, Executor>{std::forward<F>(f)})))
        -> decltype(submit(std::forward<Executor>(executor),
            detail::as_receiver<std::decay_t<F>, Executor>{std::forward<F>(f)}))
    {
        return submit(std::forward<Executor>(executor),
            detail::as_receiver<std::decay_t<F>, Executor>{std::forward<F>(f)});
    }

    namespace detail {
        template <typename Executor, typename F, typename Enable = void>
        struct is_executor_of_impl : std::false_type
        {
        };

        template <typename Executor, typename F>
        struct is_executor_of_impl<Executor, F,
            std::enable_if_t<
                hpx::traits::is_invocable<execute_t, Executor, F>::value>>
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

            template <typename R,
                typename = std::enable_if_t<is_receiver_of_v<R>>>
            auto connect(R&& r) && -> decltype(
                hpx::execution::experimental::connect(
                    std::move(exec), std::forward<R>(r)))
            {
                return hpx::execution::experimental::connect(
                    std::move(exec), std::forward<R>(r));
            }

            template <typename R,
                typename = std::enable_if_t<is_receiver_of_v<R>>>
            auto connect(R&& r) const& -> decltype(
                hpx::execution::experimental::connect(exec, std::forward<R>(r)))
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
        template <typename S, typename = std::enable_if_t<is_sender_v<S>>>
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(
            schedule_t, S&& s) noexcept(noexcept(std::forward<S>(s).schedule()))
            -> decltype(std::forward<S>(s).schedule())
        {
            return std::forward<S>(s).schedule();
        }

        template <typename S,
            typename = std::enable_if_t<
                !detail::has_member_schedule<S>::value && is_executor_v<S>>>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(schedule_t,
            S&& s) noexcept(noexcept(detail::as_sender<std::decay_t<S>>{
            std::forward<S>(s)}))
            -> decltype(detail::as_sender<std::decay_t<S>>{std::forward<S>(s)})
        {
            return detail::as_sender<std::decay_t<S>>{std::forward<S>(s)};
        }
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
            std::enable_if_t<is_executor_of_base_impl<Sender,
                detail::as_invocable<void_receiver, Sender>>::value>>
          : std::false_type
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
        std::enable_if_t<
            hpx::traits::is_invocable<schedule_t, Scheduler&&>::value &&
            std::is_copy_constructible<Scheduler>::value &&
            hpx::traits::is_equality_comparable<Scheduler>::value>>
      : std::true_type
    {
    };
}}}    // namespace hpx::execution::experimental
