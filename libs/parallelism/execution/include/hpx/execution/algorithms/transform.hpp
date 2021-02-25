//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/type_support/pack.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace hpx { namespace execution { namespace experimental {
    namespace detail {
        template <typename Variant>
        struct value_types_result;

        template <>
        struct value_types_result<hpx::util::pack<hpx::util::pack<>>>
        {
            using type = void;
        };

        template <typename T, typename... Variants>
        struct value_types_result<
            hpx::util::pack<hpx::util::pack<T>, Variants...>>
        {
            using type = T;
        };

        template <typename R, typename F>
        struct transform_receiver
        {
            std::decay_t<R> r;
            std::decay_t<F> f;

            template <typename R_, typename F_>
            transform_receiver(R_&& r, F_&& f)
              : r(std::forward<R>(r))
              , f(std::forward<F>(f))
            {
            }

            template <typename E>
            void set_error(E&& e) noexcept
            {
                hpx::execution::experimental::set_error(
                    std::move(r), std::forward<E>(e));
            }

            void set_done() noexcept
            {
                hpx::execution::experimental::set_done(std::move(r));
            };

            template <typename... Ts>
            void set_value_helper(std::true_type, Ts&&... ts) noexcept
            {
                HPX_INVOKE(f, std::forward<Ts>(ts)...);
                hpx::execution::experimental::set_value(std::move(r));
            }

            template <typename... Ts>
            void set_value_helper(std::false_type, Ts&&... ts) noexcept
            {
                hpx::execution::experimental::set_value(
                    std::move(r), HPX_INVOKE(f, std::forward<Ts>(ts)...));
            }

            template <typename... Ts,
                typename = std::enable_if_t<hpx::is_invocable_v<F, Ts...>>>
            void set_value(Ts&&... ts) noexcept
            {
                using is_void_result = std::is_void<
                    typename hpx::util::invoke_result<F, Ts...>::type>;
                set_value_helper(is_void_result{}, std::forward<Ts>(ts)...);
            }
        };

        template <typename S, typename F>
        struct transform_sender
        {
            std::decay_t<S> s;
            std::decay_t<F> f;

            template <typename Tuple>
            struct invoke_result_helper;

            template <template <typename...> class Tuple, typename... Ts>
            struct invoke_result_helper<Tuple<Ts...>>
            {
                using result_type =
                    typename hpx::util::invoke_result<F, Ts...>::type;
                using type =
                    typename std::conditional<std::is_void<result_type>::value,
                        Tuple<>, Tuple<result_type>>::type;
            };

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types = typename hpx::util::detail::unique<
                typename hpx::util::detail::transform<
                    typename hpx::execution::experimental::sender_traits<
                        S>::template value_types<Tuple, Variant>,
                    invoke_result_helper>::type>::type;

            template <template <typename...> class Variant>
            using error_types =
                typename hpx::execution::experimental::sender_traits<
                    S>::template error_types<Variant>;

            static constexpr bool sends_done = false;

            template <typename R>
            auto connect(R&& r)
            {
                return hpx::execution::experimental::connect(std::move(s),
                    transform_receiver<R, F>(std::forward<R>(r), std::move(f)));
            }
        };
    }    // namespace detail

    template <typename S, typename F>
    auto transform(S&& s, F&& f)
    {
        return detail::transform_sender<S, F>{
            std::forward<S>(s), std::forward<F>(f)};
    }
}}}    // namespace hpx::execution::experimental
