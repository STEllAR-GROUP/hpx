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
        struct upon_stopped_receiver
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;

            template <typename... Ts>
            friend void tag_invoke(
                set_value_t, upon_stopped_receiver&& r, Ts&&... ts) noexcept
            {
                hpx::execution::experimental::set_value(
                    HPX_MOVE(r.receiver), HPX_FORWARD(Ts, ts)...);
            }

            template <typename Error>
            friend void tag_invoke(
                set_error_t, upon_stopped_receiver&& r, Error&& error) noexcept
            {
                hpx::execution::experimental::set_error(
                    HPX_MOVE(r.receiver), HPX_FORWARD(Error, error));
            }

        private:
            void set_stopped_helper() && noexcept
            {
                using result_type = hpx::util::invoke_result_t<F>;
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        if constexpr (std::is_void_v<result_type>)
                        {
                            HPX_INVOKE(HPX_MOVE(f));
                            hpx::execution::experimental::set_value(
                                HPX_MOVE(receiver));
                        }
                        else
                        {
                            auto&& result = HPX_INVOKE(HPX_MOVE(f));
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
            template <HPX_CONCEPT_REQUIRES_(hpx::is_invocable_v<F>)>
            friend void tag_invoke(
                set_stopped_t, upon_stopped_receiver&& r) noexcept
            {
                HPX_MOVE(r).set_stopped_helper();
            }

            friend auto tag_invoke(
                get_env_t, upon_stopped_receiver const& r) noexcept
                -> env_of_t<std::decay_t<Receiver>>
            {
                return hpx::execution::experimental::get_env(r.receiver);
            }
        };

        template <typename Sender, typename F>
        struct upon_stopped_sender
        {
            using is_sender = void;

            HPX_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;

            template <typename Env>
            struct generate_completion_signatures
            {
                using result_type = hpx::util::invoke_result_t<F>;
                using stopped_value_type =
                    std::conditional_t<std::is_void_v<result_type>,
                        meta::pack<>, meta::pack<result_type>>;

                template <template <typename...> typename Tuple,
                    template <typename...> typename Variant>
                using value_types = hpx::util::detail::unique_concat_t<
                    value_types_of_t<Sender, Env, Tuple, Variant>,
                    stopped_value_type>;

                template <template <typename...> typename Variant>
                using error_types = hpx::util::detail::unique_concat_t<
                    error_types_of_t<Sender, Env, Variant>,
                    Variant<std::exception_ptr>>;

                static constexpr bool sends_stopped = false;
            };

            template <typename Env>
            friend auto tag_invoke(get_completion_signatures_t,
                upon_stopped_sender const&, Env) noexcept
                -> generate_completion_signatures<Env>;

            template <typename CPO,
                HPX_CONCEPT_REQUIRES_(meta::value<
                    meta::one_of<std::decay_t<CPO>, set_value_t, set_error_t>>&&
                        detail::has_completion_scheduler_v<CPO, Sender>)>
            friend constexpr auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<CPO>
                    tag,
                upon_stopped_sender const& s)
            {
                return tag(s.sender);
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, upon_stopped_sender&& s, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(HPX_MOVE(s.sender),
                    upon_stopped_receiver<Receiver, F>{
                        HPX_FORWARD(Receiver, receiver), HPX_MOVE(s.f)});
            }

            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, upon_stopped_sender& s, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(s.sender,
                    upon_stopped_receiver<Receiver, F>{
                        HPX_FORWARD(Receiver, receiver), s.f});
            }
        };
    }    // namespace detail

    inline constexpr struct upon_stopped_t final
      : hpx::functional::detail::tag_priority<upon_stopped_t>
    {
    private:
        template <typename Sender, typename F,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender>&& hpx::is_invocable_v<std::decay_t<F>>)>
        friend constexpr HPX_FORCEINLINE auto tag_invoke(
            upon_stopped_t, Sender&& sender, F&& f)
        {
            return detail::upon_stopped_sender<Sender, F>{
                HPX_FORWARD(Sender, sender), HPX_FORWARD(F, f)};
        }

        template <typename F>
        friend constexpr HPX_FORCEINLINE auto tag_invoke(upon_stopped_t, F&& f)
        {
            return detail::partial_algorithm<upon_stopped_t, F>{
                HPX_FORWARD(F, f)};
        }
    } upon_stopped{};
}    // namespace hpx::execution::experimental

#endif
