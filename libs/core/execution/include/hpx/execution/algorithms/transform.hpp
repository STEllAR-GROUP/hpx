//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution_base/completion_scheduler.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/type_support/pack.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace hpx { namespace execution { namespace experimental {
    namespace detail {
        template <typename Receiver, typename F>
        struct transform_receiver
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;

            template <typename Error>
            friend void tag_dispatch(
                set_error_t, transform_receiver&& r, Error&& error) noexcept
            {
                hpx::execution::experimental::set_error(
                    std::move(r.receiver), std::forward<Error>(error));
            }

            friend void tag_dispatch(
                set_done_t, transform_receiver&& r) noexcept
            {
                hpx::execution::experimental::set_done(std::move(r.receiver));
            }

        private:
            template <typename... Ts>
            void set_value_helper(std::true_type, Ts&&... ts) noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        HPX_INVOKE(f, std::forward<Ts>(ts)...);
                        hpx::execution::experimental::set_value(
                            std::move(receiver));
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            std::move(receiver), std::move(ep));
                    });
            }

            template <typename... Ts>
            void set_value_helper(std::false_type, Ts&&... ts) noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        auto&& result = HPX_INVOKE(f, std::forward<Ts>(ts)...);
                        hpx::execution::experimental::set_value(
                            std::move(receiver), std::move(result));
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            std::move(receiver), std::move(ep));
                    });
            }

        public:
            template <typename... Ts,
                typename = std::enable_if_t<hpx::is_invocable_v<F, Ts...>>>
            friend void tag_dispatch(
                set_value_t, transform_receiver&& r, Ts&&... ts) noexcept
            {
                using is_void_result =
                    std::is_void<hpx::util::invoke_result_t<F, Ts...>>;
                r.set_value_helper(is_void_result{}, std::forward<Ts>(ts)...);
            }
        };

        template <typename Sender, typename F>
        struct transform_sender
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;

            template <typename Tuple>
            struct invoke_result_helper;

            template <template <typename...> class Tuple, typename... Ts>
            struct invoke_result_helper<Tuple<Ts...>>
            {
                using result_type = hpx::util::invoke_result_t<F, Ts...>;
                using type =
                    typename std::conditional<std::is_void<result_type>::value,
                        Tuple<>, Tuple<result_type>>::type;
            };

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types =
                hpx::util::detail::unique_t<hpx::util::detail::transform_t<
                    typename hpx::execution::experimental::sender_traits<
                        Sender>::template value_types<Tuple, Variant>,
                    invoke_result_helper>>;

            template <template <typename...> class Variant>
            using error_types =
                hpx::util::detail::unique_t<hpx::util::detail::prepend_t<
                    typename hpx::execution::experimental::sender_traits<
                        Sender>::template error_types<Variant>,
                    std::exception_ptr>>;

            static constexpr bool sends_done = false;

            template <typename CPO,
                // clang-format off
                HPX_CONCEPT_REQUIRES_(
                    hpx::execution::experimental::detail::is_receiver_cpo_v<CPO> &&
                    hpx::execution::experimental::detail::has_completion_scheduler_v<
                        CPO, std::decay_t<Sender>>)
                // clang-format on
                >
            friend constexpr auto tag_dispatch(
                hpx::execution::experimental::get_completion_scheduler_t<CPO>,
                transform_sender const& sender)
            {
                return hpx::execution::experimental::get_completion_scheduler<
                    CPO>(sender.sender);
            }

            template <typename Receiver>
            friend auto tag_dispatch(
                connect_t, transform_sender&& s, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(
                    std::move(s.sender),
                    transform_receiver<Receiver, F>{
                        std::forward<Receiver>(receiver), std::move(s.f)});
            }

            template <typename Receiver>
            friend auto tag_dispatch(
                connect_t, transform_sender& r, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(r.sender,
                    transform_receiver<Receiver, F>{
                        std::forward<Receiver>(receiver), r.f});
            }
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct transform_t final
      : hpx::functional::tag_fallback<transform_t>
    {
    private:
        // clang-format off
        template <typename Sender, typename F,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            transform_t, Sender&& sender, F&& f)
        {
            return detail::transform_sender<Sender, F>{
                std::forward<Sender>(sender), std::forward<F>(f)};
        }

        template <typename F>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            transform_t, F&& f)
        {
            return detail::partial_algorithm<transform_t, F>{
                std::forward<F>(f)};
        }
    } transform{};
}}}    // namespace hpx::execution::experimental
