//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_cuda/custom_gpu_api.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution/algorithms/transform.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/type_support/pack.hpp>

#include <exception>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace cuda { namespace experimental {
    namespace detail {
        template <typename R, typename... Ts>
        void set_value_event_callback_helper(
            cudaError_t status, R&& r, Ts&&... ts)
        {
            static_assert(sizeof...(Ts) <= 1, "Expecting at most one value");

            HPX_ASSERT(status != cudaErrorNotReady);

            if (status == cudaSuccess)
            {
                hpx::execution::experimental::set_value(
                    std::forward<R>(r), std::forward<Ts>(ts)...);
            }
            else
            {
                hpx::execution::experimental::set_error(std::forward<R>(r),
                    std::make_exception_ptr(
                        cuda_exception(std::string("Getting event after "
                                                   "CUDA stream transform "
                                                   "failed with status ") +
                                cudaGetErrorString(status),
                            status)));
            }
        }

        template <typename... Ts>
        void extend_argument_lifetimes(cudaStream_t stream, Ts&&... ts)
        {
            if constexpr (sizeof...(Ts) > 0)
            {
                detail::add_event_callback(
                    [keep_alive = hpx::make_tuple(std::forward<Ts>(ts)...)](
                        cudaError_t status) {
                        HPX_ASSERT(status != cudaErrorNotReady);
                        HPX_UNUSED(status);
                    },
                    stream);
            }
        }

        template <typename R, typename... Ts>
        void set_value_immediate_void(cudaStream_t stream, R&& r, Ts&&... ts)
        {
            hpx::execution::experimental::set_value(std::forward<R>(r));

            // Even though we call set_value immediately, we still extend the
            // life time of the arguments by capturing them in a callback that
            // is triggered when the event is ready.
            extend_argument_lifetimes(stream, std::forward<Ts>(ts)...);
        }

        template <typename R, typename... Ts>
        void set_value_event_callback_void(
            cudaStream_t stream, R&& r, Ts&&... ts)
        {
            detail::add_event_callback(
                [r = std::forward<R>(r),
                    keep_alive = hpx::make_tuple(std::forward<Ts>(ts)...)](
                    cudaError_t status) mutable {
                    set_value_event_callback_helper(status, std::move(r));
                },
                stream);
        }

        template <typename R, typename T, typename... Ts>
        void set_value_immediate_non_void(
            cudaStream_t stream, R&& r, T&& t, Ts&&... ts)
        {
            hpx::execution::experimental::set_value(
                std::forward<R>(r), std::forward<T>(t));

            // Even though we call set_value immediately, we still extend the
            // life time of the arguments by capturing them in a callback that
            // is triggered when the event is ready.
            extend_argument_lifetimes(stream, std::forward<Ts>(ts)...);
        }

        template <typename R, typename T, typename... Ts>
        void set_value_event_callback_non_void(
            cudaStream_t stream, R&& r, T&& t, Ts&&... ts)
        {
            detail::add_event_callback(
                [t = std::forward<T>(t), r = std::forward<R>(r),
                    keep_alive = hpx::make_tuple(std::forward<Ts>(ts)...)](
                    cudaError_t status) mutable {
                    set_value_event_callback_helper(
                        status, std::move(r), std::move(t));
                },
                stream);
        }

        template <typename R, typename F>
        struct transform_stream_receiver;

        template <typename R>
        struct is_transform_stream_receiver : std::false_type
        {
        };

        template <typename R, typename F>
        struct is_transform_stream_receiver<transform_stream_receiver<R, F>>
          : std::true_type
        {
        };

        template <typename R, typename F>
        struct transform_stream_receiver
        {
            std::decay_t<R> r;
            std::decay_t<F> f;
            cudaStream_t stream;

            template <typename R_, typename F_>
            transform_stream_receiver(R_&& r, F_&& f, cudaStream_t stream)
              : r(std::forward<R_>(r))
              , f(std::forward<F_>(f))
              , stream(stream)
            {
            }

            template <typename E>
            friend void tag_dispatch(hpx::execution::experimental::set_error_t,
                transform_stream_receiver&& r, E&& e) noexcept
            {
                hpx::execution::experimental::set_error(
                    std::move(r.r), std::forward<E>(e));
            }

            friend void tag_dispatch(hpx::execution::experimental::set_done_t,
                transform_stream_receiver&& r) noexcept
            {
                hpx::execution::experimental::set_done(std::move(r.r));
            }

            template <typename... Ts>
            void set_value(Ts&&... ts) noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        if constexpr (std::is_void_v<
                                          typename hpx::util::invoke_result<F,
                                              Ts..., cudaStream_t>::type>)
                        {
                            // When the return type is void, there is no value to
                            // forward to the receiver
                            HPX_INVOKE(f, ts..., stream);

                            if constexpr (is_transform_stream_receiver<
                                              std::decay_t<R>>::value)
                            {
                                if (r.stream == stream)
                                {
                                    // When the next receiver is also a
                                    // transform_stream_receiver, we can immediately
                                    // call set_value, with the knowledge that a
                                    // later receiver will synchronize the stream
                                    // when a non-transform_stream receiver is
                                    // connected.
                                    set_value_immediate_void(stream,
                                        std::move(r), std::forward<Ts>(ts)...);
                                }
                                else
                                {
                                    // When the streams are different, we add a
                                    // callback which will call set_value on the
                                    // receiver.
                                    set_value_event_callback_void(stream,
                                        std::move(r), std::forward<Ts>(ts)...);
                                }
                            }
                            else
                            {
                                // When the next receiver is not a
                                // transform_stream_receiver, we add a callback
                                // which will call set_value on the receiver.
                                set_value_event_callback_void(stream,
                                    std::move(r), std::forward<Ts>(ts)...);
                            }
                        }
                        else
                        {
                            // When the return type is non-void, we have to forward
                            // the value to the receiver
                            auto t =
                                HPX_INVOKE(f, std::forward<Ts>(ts)..., stream);

                            if constexpr (is_transform_stream_receiver<
                                              std::decay_t<R>>::value)
                            {
                                if (r.stream == stream)
                                {
                                    // When the next receiver is also a
                                    // transform_stream_receiver, we can immediately
                                    // call set_value, with the knowledge that a
                                    // later receiver will synchronize the stream
                                    // when a non-transform_stream receiver is
                                    // connected.
                                    set_value_immediate_non_void(stream,
                                        std::move(r), std::move(t),
                                        std::forward<Ts>(ts)...);
                                }
                                else
                                {
                                    // When the streams are different, we add a
                                    // callback which will call set_value on the
                                    // receiver.
                                    set_value_event_callback_non_void(stream,
                                        std::move(r), std::move(t),
                                        std::forward<Ts>(ts)...);
                                }
                            }
                            else
                            {
                                // When the next receiver is not a
                                // transform_stream_receiver, we add a callback
                                // which will call set_value on the receiver.
                                set_value_event_callback_non_void(stream,
                                    std::move(r), std::move(t),
                                    std::forward<Ts>(ts)...);
                            }
                        }
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            std::move(r), std::move(ep));
                    });
            }
        };

        template <typename S, typename F>
        struct transform_stream_sender
        {
            std::decay_t<S> s;
            std::decay_t<F> f;
            cudaStream_t stream{};

            template <typename Tuple>
            struct invoke_result_helper;

            template <template <typename...> class Tuple, typename... Ts>
            struct invoke_result_helper<Tuple<Ts...>>
            {
                using result_type = typename hpx::util::invoke_result<F, Ts...,
                    cudaStream_t>::type;
                using type =
                    typename std::conditional<std::is_void<result_type>::value,
                        Tuple<>, Tuple<result_type>>::type;
            };

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types =
                hpx::util::detail::unique_t<hpx::util::detail::transform_t<
                    typename hpx::execution::experimental::sender_traits<
                        S>::template value_types<Tuple, Variant>,
                    invoke_result_helper>>;

            template <template <typename...> class Variant>
            using error_types =
                hpx::util::detail::unique_t<hpx::util::detail::prepend_t<
                    typename hpx::execution::experimental::sender_traits<
                        S>::template error_types<Variant>,
                    std::exception_ptr>>;

            static constexpr bool sends_done = false;

            template <typename R>
            friend auto tag_dispatch(hpx::execution::experimental::connect_t,
                transform_stream_sender&& s, R&& r)
            {
                return hpx::execution::experimental::connect(std::move(s.s),
                    transform_stream_receiver<R, F>{
                        std::forward<R>(r), std::move(s.f), s.stream});
            }
        };

        // This should be a hidden friend in transform_stream_receiver. However,
        // nvcc does not know how to compile it with some argument types
        // ("error: no instance of overloaded function std::forward matches the
        // argument list").
        template <typename R, typename F, typename... Ts>
        void tag_dispatch(hpx::execution::experimental::set_value_t,
            transform_stream_receiver<R, F>&& r, Ts&&... ts)
        {
            r.set_value(std::forward<Ts>(ts)...);
        }
    }    // namespace detail

    // NOTE: This is not a customization of
    // hpx::execution::experimental::transform. It has different semantics:
    // - a cudaStream_t is inserted as an additional argument into the call to f
    // - values from the predecessor sender are not forwarded, only passed by
    //   reference, to the call to f to keep them alive until the event is ready
    HPX_INLINE_CONSTEXPR_VARIABLE struct transform_stream_t final
      : hpx::functional::tag_fallback<transform_stream_t>
    {
    private:
        template <typename S, typename F,
            typename = std::enable_if_t<
                !std::is_same<std::decay_t<F>, cudaStream_t>::value>>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            transform_stream_t, S&& s, F&& f, cudaStream_t stream = {})
        {
            return detail::transform_stream_sender<S, F>{
                std::forward<S>(s), std::forward<F>(f), stream};
        }

        template <typename F>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            transform_stream_t, F&& f, cudaStream_t stream = {})
        {
            return hpx::execution::experimental::detail::partial_algorithm<
                transform_stream_t, F, cudaStream_t>{
                std::forward<F>(f), stream};
        }
    } transform_stream{};
}}}    // namespace hpx::cuda::experimental
