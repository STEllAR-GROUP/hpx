//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if defined(HPX_HAVE_CXX17_STD_VARIANT)
#include <hpx/execution_base/detail/try_catch_exception_ptr.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/type_support/detail/with_result_of.hpp>
#include <hpx/type_support/pack.hpp>

#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace hpx { namespace execution { namespace experimental {
    namespace detail {
        template <typename PS, typename F>
        struct let_error_sender
        {
            typename std::decay_t<PS> ps;
            typename std::decay_t<F> f;

            // Type of the potential values returned from the predecessor sender
            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using predecessor_value_types =
                typename hpx::execution::experimental::sender_traits<
                    std::decay_t<PS>>::template value_types<Tuple, Variant>;

            // Type of the potential errors returned from the predecessor sender
            template <template <typename...> class Variant>
            using predecessor_error_types =
                typename hpx::execution::experimental::sender_traits<
                    std::decay_t<PS>>::template error_types<Variant>;
            static_assert(
                !std::is_same<predecessor_error_types<hpx::util::pack>,
                    hpx::util::pack<>>::value,
                "let_error used with a predecessor that has an empty "
                "error_types. Is let_error misplaced?");

            template <typename E>
            struct successor_sender_types_helper
            {
                using type = hpx::util::invoke_result_t<F,
                    std::add_lvalue_reference_t<E>>;
                static_assert(hpx::execution::experimental::is_sender<
                                  std::decay_t<type>>::value,
                    "let_error expects the invocable sender factory to return "
                    "a sender");
            };

            // Type of the potential senders returned from the sender factor F
            template <template <typename...> class Variant>
            using successor_sender_types = hpx::util::detail::unique_t<
                hpx::util::detail::transform_t<predecessor_error_types<Variant>,
                    successor_sender_types_helper>>;

            // The workaround for clang is due to a parsing bug in clang < 11
            // in CUDA mode (where >>> also has a different meaning in kernel
            // launches).
            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types = hpx::util::detail::unique_concat_t<
                predecessor_value_types<Tuple, Variant>,
                hpx::util::detail::concat_pack_of_packs_t<hpx::util::detail::
                        transform_t<successor_sender_types<Variant>,
                            value_types<Tuple, Variant>::template apply
#if defined(HPX_CLANG_VERSION) && HPX_CLANG_VERSION < 110000
                            >
                    //
                    >>;
#else
                            >>>;
#endif

            template <template <typename...> class Variant>
            using error_types =
                hpx::util::detail::unique_t<hpx::util::detail::prepend_t<
                    hpx::util::detail::concat_pack_of_packs_t<hpx::util::
                            detail::transform_t<successor_sender_types<Variant>,
                                error_types<Variant>::template apply>>,
                    std::exception_ptr>>;

            static constexpr bool sends_done = false;

            template <typename R>
            struct operation_state
            {
                struct let_error_predecessor_receiver
                {
                    std::decay_t<R> r;
                    std::decay_t<F> f;
                    operation_state& os;

                    template <typename R_, typename F_>
                    let_error_predecessor_receiver(
                        R_&& r, F_&& f, operation_state& os)
                      : r(std::forward<R_>(r))
                      , f(std::forward<F_>(f))
                      , os(os)
                    {
                    }

                    struct start_visitor
                    {
                        HPX_NORETURN void operator()(std::monostate) const
                        {
                            HPX_UNREACHABLE;
                        }

                        template <typename OS_,
                            typename = std::enable_if_t<!std::is_same_v<
                                std::decay_t<OS_>, std::monostate>>>
                        void operator()(OS_& os) const
                        {
                            hpx::execution::experimental::start(os);
                        }
                    };

                    struct set_error_visitor
                    {
                        std::decay_t<R> r;
                        std::decay_t<F> f;
                        operation_state& os;

                        HPX_NORETURN void operator()(std::monostate) const
                        {
                            HPX_UNREACHABLE;
                        }

                        template <typename E,
                            typename = std::enable_if_t<!std::is_same_v<
                                std::decay_t<E>, std::monostate>>>
                        void operator()(E& e)
                        {
                            using operation_state_type =
                                decltype(hpx::execution::experimental::connect(
                                    HPX_INVOKE(std::move(f), e),
                                    std::declval<R>()));

                            // with_result_of is used to emplace the operation
                            // state returned from connect without any
                            // intermediate copy construction (the operation
                            // state is not required to be copyable nor movable.
                            os.successor_os
                                .template emplace<operation_state_type>(
                                    hpx::util::detail::with_result_of([&]() {
                                        return hpx::execution::experimental::
                                            connect(HPX_INVOKE(std::move(f), e),
                                                std::move(r));
                                    }));
                            std::visit(start_visitor{}, os.successor_os);
                        }
                    };

                    template <typename E>
                        void set_error(E&& e) && noexcept
                    {
                        hpx::detail::try_catch_exception_ptr(
                            [&]() {
                                // TODO: r is moved before the visit, but the
                                // invoke inside the visit may throw.
                                os.predecessor_e.template emplace<E>(
                                    std::forward<E>(e));
                                std::visit(set_error_visitor{std::move(r),
                                               std::move(f), os},
                                    os.predecessor_e);
                            },
                            [&](std::exception_ptr ep) {
                                hpx::execution::experimental::set_error(
                                    std::move(r), std::move(ep));
                            });
                    }

                    void set_done() && noexcept
                    {
                        hpx::execution::experimental::set_done(std::move(r));
                    };

                    template <typename... Ts>
                        void set_value(Ts&&... ts) && noexcept
                    {
                        hpx::execution::experimental::set_value(
                            std::move(r), std::forward<Ts>(ts)...);
                    }
                };

                // Type of the operation state returned when connecting the
                // predecessor sender to the let_error_predecessor_receiver
                using predecessor_operation_state_type = std::decay_t<
                    connect_result_t<PS&&, let_error_predecessor_receiver>>;

                // Type of the potential operation states returned when
                // connecting a sender in successor_sender_types to the receiver
                // connected to the let_error_sender
                template <typename Sender>
                struct successor_operation_state_types_helper
                {
                    using type = connect_result_t<Sender, R>;
                };
                template <template <typename...> class Variant>
                using successor_operation_state_types =
                    hpx::util::detail::transform_t<
                        successor_sender_types<Variant>,
                        successor_operation_state_types_helper>;

                // Operation state from connecting predecessor sender to
                // let_error_predecessor_receiver
                predecessor_operation_state_type predecessor_os;

                // Potential errors returned from the predecessor sender
                hpx::util::detail::prepend_t<
                    predecessor_error_types<std::variant>, std::monostate>
                    predecessor_e;

                // Potential operation states returned when connecting a sender
                // in successor_sender_types to the receiver connected to the
                // let_error_sender
                hpx::util::detail::prepend_t<
                    successor_operation_state_types<std::variant>,
                    std::monostate>
                    successor_os;

                template <typename PS_, typename R_, typename F_>
                operation_state(PS_&& ps, R_&& r, F_&& f)
                  : predecessor_os{hpx::execution::experimental::connect(
                        std::forward<PS_>(ps),
                        let_error_predecessor_receiver(
                            std::forward<R_>(r), std::forward<F_>(f), *this))}
                {
                }

                operation_state(operation_state&&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                void start() & noexcept
                {
                    hpx::execution::experimental::start(predecessor_os);
                }
            };

            template <typename R>
            auto connect(R&& r) &&
            {
                return operation_state<R>(
                    std::move(ps), std::forward<R>(r), std::move(f));
            }
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct let_error_t final
      : hpx::functional::tag_fallback<let_error_t>
    {
    private:
        template <typename PS, typename F>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            let_error_t, PS&& ps, F&& f)
        {
            return detail::let_error_sender<PS, F>{
                std::forward<PS>(ps), std::forward<F>(f)};
        }

        template <typename F>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            let_error_t, F&& f)
        {
            return detail::partial_algorithm<let_error_t, F>{
                std::forward<F>(f)};
        }
    } let_error{};
}}}    // namespace hpx::execution::experimental
#endif
