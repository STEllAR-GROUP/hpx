//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_STDEXEC)
#include <hpx/modules/execution_base.hpp>
#else

#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution_base/receiver_inlining.hpp>
#include <hpx/modules/concepts.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/tag_invoke.hpp>
#include <hpx/modules/type_support.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    namespace detail {

        template <typename Receiver, typename F>
        struct upon_error_receiver
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;

            template <typename... Ts>
            friend void tag_invoke(
                set_value_t, upon_error_receiver&& r, Ts&&... ts) noexcept
            {
                hpx::execution::experimental::set_value(
                    HPX_MOVE(r.receiver), HPX_FORWARD(Ts, ts)...);
            }

            friend void tag_invoke(
                set_stopped_t, upon_error_receiver&& r) noexcept
            {
                hpx::execution::experimental::set_stopped(HPX_MOVE(r.receiver));
            }

        private:
            template <typename Error>
            void set_error_helper(Error&& error) && noexcept
            {
                using result_type = hpx::util::invoke_result_t<F, Error>;
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        if constexpr (std::is_void_v<result_type>)
                        {
                            HPX_INVOKE(HPX_MOVE(f), HPX_FORWARD(Error, error));
                            hpx::execution::experimental::set_value(
                                HPX_MOVE(receiver));
                        }
                        else
                        {
                            auto&& result = HPX_INVOKE(
                                HPX_MOVE(f), HPX_FORWARD(Error, error));
                            hpx::execution::experimental::set_value(
                                HPX_MOVE(receiver), HPX_MOVE(result));
                        }
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(receiver), HPX_MOVE(ep));
                    });
            }

        public:
            template <typename Error,
                HPX_CONCEPT_REQUIRES_(hpx::is_invocable_v<F, Error>)>
            friend void tag_invoke(
                set_error_t, upon_error_receiver&& r, Error&& error) noexcept
            {
                HPX_MOVE(r).set_error_helper(HPX_FORWARD(Error, error));
            }

            friend auto tag_invoke(
                get_env_t, upon_error_receiver const& r) noexcept
                -> env_of_t<std::decay_t<Receiver>>
            {
                return hpx::execution::experimental::get_env(r.receiver);
            }
        };

        template <typename Sender, typename F>
        struct upon_error_sender
        {
            using is_sender = void;

            HPX_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;

            template <typename Error, typename Enable = void>
            struct generate_set_value_signature
            {
                using type = meta::pack<>;
            };

            template <typename Error>
            struct generate_set_value_signature<Error,
                std::enable_if_t<hpx::is_invocable_v<F, Error>>>
            {
                using result_type = hpx::util::invoke_result_t<F, Error>;
                using type = std::conditional_t<std::is_void_v<result_type>,
                    meta::pack<>, meta::pack<result_type>>;
            };

            template <typename Env>
            struct generate_completion_signatures
            {
                template <template <typename...> typename Tuple,
                    template <typename...> typename Variant>
                using value_types = hpx::util::detail::unique_concat_t<
                    value_types_of_t<Sender, Env, Tuple, Variant>,
                    hpx::util::detail::concat_inner_packs_t<
                        hpx::util::detail::transform_t<
                            error_types_of_t<Sender, Env, meta::pack>,
                            generate_set_value_signature>>>;

                template <template <typename...> typename Variant>
                using error_types = Variant<std::exception_ptr>;

                static constexpr bool sends_stopped =
                    sends_stopped_of_v<Sender, Env>;
            };

            template <typename Env>
            friend auto tag_invoke(get_completion_signatures_t,
                upon_error_sender const&, Env) noexcept
                -> generate_completion_signatures<Env>;

            template <typename CPO,
                HPX_CONCEPT_REQUIRES_(meta::value<meta::one_of<
                        std::decay_t<CPO>, set_value_t, set_stopped_t>>&&
                        detail::has_completion_scheduler_v<CPO, Sender>)>
            friend constexpr auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<CPO>
                    tag,
                upon_error_sender const& s)
            {
                return tag(s.sender);
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, upon_error_sender&& s, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(HPX_MOVE(s.sender),
                    upon_error_receiver<Receiver, F>{
                        HPX_FORWARD(Receiver, receiver), HPX_MOVE(s.f)});
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, upon_error_sender& s, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(s.sender,
                    upon_error_receiver<Receiver, F>{
                        HPX_FORWARD(Receiver, receiver), s.f});
            }
        };
    }    // namespace detail

    inline constexpr struct upon_error_t final
      : hpx::functional::detail::tag_priority<upon_error_t>
    {
    private:
        template <typename Sender, typename F,
            HPX_CONCEPT_REQUIRES_(is_sender_v<Sender>&&
                    std::is_invocable_v<std::decay_t<F>, std::exception_ptr>)>
        friend constexpr HPX_FORCEINLINE auto tag_invoke(
            upon_error_t, Sender&& sender, F&& f)
        {
            return detail::upon_error_sender<Sender, F>{
                HPX_FORWARD(Sender, sender), HPX_FORWARD(F, f)};
        }

        template <typename F>
        friend constexpr HPX_FORCEINLINE auto tag_invoke(upon_error_t, F&& f)
        {
            return detail::partial_algorithm<upon_error_t, F>{
                HPX_FORWARD(F, f)};
        }
    } upon_error{};
}    // namespace hpx::execution::experimental

#endif
