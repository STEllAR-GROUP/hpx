//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c) 2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/async_mpi/transform_mpi.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_mpi/mpi_future.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/traits/is_invocable.hpp>
#include <hpx/mpi_base/mpi.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace hpx { namespace mpi { namespace experimental {
    namespace detail {

        template <typename R, typename... Ts>
        void set_value_request_callback_helper(
            int mpi_status, R&& r, Ts&&... ts)
        {
            static_assert(sizeof...(Ts) <= 1, "Expecting at most one value");
            if (mpi_status == MPI_SUCCESS)
            {
                hpx::execution::experimental::set_value(
                    HPX_FORWARD(R, r), HPX_FORWARD(Ts, ts)...);
            }
            else
            {
                hpx::execution::experimental::set_error(HPX_FORWARD(R, r),
                    std::make_exception_ptr(mpi_exception(mpi_status)));
            }
        }

        template <typename R, typename... Ts>
        void set_value_request_callback_void(
            MPI_Request request, R&& r, Ts&&... ts)
        {
            detail::add_request_callback(
                [r = HPX_FORWARD(R, r),
                    keep_alive = hpx::make_tuple(HPX_FORWARD(Ts, ts)...)](
                    int status) mutable {
                    set_value_request_callback_helper(status, HPX_MOVE(r));
                },
                request);
        }

        template <typename R, typename InvokeResult, typename... Ts>
        void set_value_request_callback_non_void(
            MPI_Request request, R&& r, InvokeResult&& res, Ts&&... ts)
        {
            detail::add_request_callback(
                [r = HPX_FORWARD(R, r), res = HPX_FORWARD(InvokeResult, res),
                    keep_alive = hpx::make_tuple(HPX_FORWARD(Ts, ts)...)](
                    int status) mutable {
                    set_value_request_callback_helper(
                        status, HPX_MOVE(r), HPX_MOVE(res));
                },
                request);
        }

        template <typename R, typename F>
        struct transform_mpi_receiver
        {
#ifdef HPX_HAVE_STDEXEC
            using is_receiver = void;
#endif
            HPX_NO_UNIQUE_ADDRESS std::decay_t<R> r;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;

            template <typename R_, typename F_>
            transform_mpi_receiver(R_&& r, F_&& f)
              : r(HPX_FORWARD(R_, r))
              , f(HPX_FORWARD(F_, f))
            {
            }

            template <typename E>
            friend constexpr void tag_invoke(
                hpx::execution::experimental::set_error_t,
                transform_mpi_receiver&& r, E&& e) noexcept
            {
                hpx::execution::experimental::set_error(
                    HPX_MOVE(r.r), HPX_FORWARD(E, e));
            }

            friend constexpr void tag_invoke(
                hpx::execution::experimental::set_stopped_t,
                transform_mpi_receiver&& r) noexcept
            {
                hpx::execution::experimental::set_stopped(HPX_MOVE(r.r));
            };

            template <typename... Ts,
                typename = std::enable_if_t<
                    hpx::is_invocable_v<F, Ts..., MPI_Request*>>>
            friend constexpr void tag_invoke(
                hpx::execution::experimental::set_value_t,
                transform_mpi_receiver&& r, Ts&&... ts) noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        if constexpr (std::is_void_v<util::invoke_result_t<F,
                                          Ts..., MPI_Request*>>)
                        {
                            MPI_Request request;
                            HPX_INVOKE(r.f, ts..., &request);
                            // When the return type is void, there is no value
                            // to forward to the receiver
                            set_value_request_callback_void(
                                request, HPX_MOVE(r.r), HPX_FORWARD(Ts, ts)...);
                        }
                        else
                        {
                            MPI_Request request;
                            // When the return type is non-void, we have to
                            // forward the value to the receiver
                            auto&& result = HPX_INVOKE(
                                r.f, HPX_FORWARD(Ts, ts)..., &request);
                            set_value_request_callback_non_void(request,
                                HPX_MOVE(r.r), HPX_MOVE(result),
                                HPX_FORWARD(Ts, ts)...);
                        }
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(r.r), HPX_MOVE(ep));
                    });
            }
        };

        template <typename Sender, typename F>
        struct transform_mpi_sender
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Sender> s;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;

#ifdef HPX_HAVE_STDEXEC
            using is_sender = void;

            template <typename... Args>
            struct invoke_function_transformation_helper
            {
                template <bool IsVoid, typename T>
                struct set_value_void_checked
                {
                    using type = hpx::execution::experimental::set_value_t(T);
                };

                template <typename T>
                struct set_value_void_checked<true, T>
                {
                    using type = hpx::execution::experimental::set_value_t();
                };

                static_assert(hpx::is_invocable_v<F, Args..., MPI_Request*>,
                    "F not invocable with the value_types specified.");

                using result_type =
                    hpx::util::invoke_result_t<F, Args..., MPI_Request*>;
                using set_value_result_type =
                    typename set_value_void_checked<std::is_void_v<result_type>,
                        result_type>::type;
                using type =
                    hpx::execution::experimental::completion_signatures<
                        set_value_result_type>;
            };

            template <typename... Args>
            using invoke_function_transformation =
                invoke_function_transformation_helper<Args...>::type;

            template <typename Err>
            using default_set_error =
                hpx::execution::experimental::completion_signatures<
                    hpx::execution::experimental::set_error_t(Err)>;

            using no_set_stopped_signature =
                hpx::execution::experimental::completion_signatures<>;

            // clang-format off
            template <typename Env>
            friend auto tag_invoke(
                hpx::execution::experimental::get_completion_signatures_t,
                transform_mpi_sender const&, Env const&)
            -> hpx::execution::experimental::transform_completion_signatures_of<
                Sender, Env,
                hpx::execution::experimental::completion_signatures<
                    hpx::execution::experimental::set_error_t(std::exception_ptr)
                >,
                invoke_function_transformation,
                default_set_error,
                no_set_stopped_signature
            >{};
            // clang-format on
#else
            template <typename Env>
            struct generate_completion_signatures
            {
                template <typename Tuple>
                struct invoke_result_helper;

                template <template <typename...> class Tuple, typename... Ts>
                struct invoke_result_helper<Tuple<Ts...>>
                {
                    static_assert(hpx::is_invocable_v<F, Ts..., MPI_Request*>,
                        "F not invocable with the value_types specified.");
                    using result_type =
                        hpx::util::invoke_result_t<F, Ts..., MPI_Request*>;
                    using type =
                        std::conditional_t<std::is_void<result_type>::value,
                            Tuple<>, Tuple<result_type>>;
                };

                template <template <typename...> class Tuple,
                    template <typename...> class Variant>
                using value_types =
                    hpx::util::detail::unique_t<hpx::util::detail::transform_t<
                        hpx::execution::experimental::value_types_of_t<Sender,
                            Env, Tuple, Variant>,
                        invoke_result_helper>>;

                template <template <typename...> class Variant>
                using error_types =
                    hpx::util::detail::unique_t<hpx::util::detail::prepend_t<
                        hpx::execution::experimental::error_types_of_t<Sender,
                            Env, Variant>,
                        std::exception_ptr>>;

                static constexpr bool sends_stopped = false;
            };

            // clang-format off
            template <typename Env>
            friend auto tag_invoke(
                hpx::execution::experimental::get_completion_signatures_t,
                transform_mpi_sender const&, Env)
            -> generate_completion_signatures<Env>;
            // clang-format on
#endif

            template <typename R>
            friend constexpr auto tag_invoke(
                hpx::execution::experimental::connect_t,
                transform_mpi_sender& s, R&& r)
            {
                return hpx::execution::experimental::connect(
                    s.s, transform_mpi_receiver<R, F>(HPX_FORWARD(R, r), s.f));
            }

            template <typename R>
            friend constexpr auto tag_invoke(
                hpx::execution::experimental::connect_t,
                transform_mpi_sender&& s, R&& r)
            {
                return hpx::execution::experimental::connect(HPX_MOVE(s.s),
                    transform_mpi_receiver<R, F>(
                        HPX_FORWARD(R, r), HPX_MOVE(s.f)));
            }
        };
    }    // namespace detail

    inline constexpr struct transform_mpi_t final
      : hpx::functional::detail::tag_fallback<transform_mpi_t>
    {
    private:
        template <typename Sender, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_sender_v<Sender>)>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            transform_mpi_t, Sender&& s, F&& f)
        {
            return detail::transform_mpi_sender<Sender, F>{
                HPX_FORWARD(Sender, s), HPX_FORWARD(F, f)};
        }

        template <typename F>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            transform_mpi_t, F&& f)
        {
            return ::hpx::execution::experimental::detail::partial_algorithm<
                transform_mpi_t, F>{HPX_FORWARD(F, f)};
        }
    } transform_mpi{};
}}}    // namespace hpx::mpi::experimental
