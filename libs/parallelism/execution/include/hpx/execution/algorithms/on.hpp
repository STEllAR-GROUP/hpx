//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if defined(HPX_HAVE_CXX17_STD_VARIANT)
#include <hpx/datastructures/optional.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/type_support/detail/with_result_of.hpp>
#include <hpx/type_support/pack.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <type_traits>
#include <utility>
#include <variant>

namespace hpx { namespace execution { namespace experimental {
    namespace detail {
        template <typename Sender, typename Scheduler>
        struct on_sender
        {
            std::decay_t<Sender> predecessor_sender;
            std::decay_t<Scheduler> scheduler;

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types =
                typename hpx::execution::experimental::sender_traits<
                    Sender>::template value_types<Tuple, Variant>;

            template <template <typename...> class Variant>
            using predecessor_sender_error_types =
                typename hpx::execution::experimental::sender_traits<
                    Sender>::template error_types<Variant>;

            using scheduler_sender_type = typename hpx::util::invoke_result<
                hpx::execution::experimental::schedule_t, Scheduler>::type;
            template <template <typename...> class Variant>
            using scheduler_sender_error_types =
                typename hpx::execution::experimental::sender_traits<
                    scheduler_sender_type>::template error_types<Variant>;

            template <template <typename...> class Variant>
            using error_types = hpx::util::detail::unique_concat_t<
                predecessor_sender_error_types<Variant>,
                scheduler_sender_error_types<Variant>>;

            static constexpr bool sends_done = false;

            template <typename Receiver>
            struct operation_state
            {
                std::decay_t<Scheduler> scheduler;
                std::decay_t<Receiver> receiver;

                struct predecessor_sender_receiver;
                struct scheduler_sender_receiver;

                using value_type = hpx::util::detail::prepend_t<
                    typename hpx::execution::experimental::sender_traits<
                        Sender>::template value_types<hpx::tuple, std::variant>,
                    std::monostate>;
                value_type ts;

                using sender_operation_state_type =
                    connect_result_t<Sender, predecessor_sender_receiver>;
                sender_operation_state_type sender_os;

                using scheduler_operation_state_type =
                    connect_result_t<typename hpx::util::invoke_result<
                                         schedule_t, Scheduler>::type,
                        scheduler_sender_receiver>;
                hpx::util::optional<scheduler_operation_state_type>
                    scheduler_os;

                template <typename Sender_, typename Scheduler_,
                    typename Receiver_>
                operation_state(Sender_&& predecessor_sender,
                    Scheduler_&& scheduler, Receiver_&& receiver)
                  : scheduler(std::forward<Scheduler>(scheduler))
                  , receiver(std::forward<Receiver_>(receiver))
                  , sender_os(hpx::execution::experimental::connect(
                        std::forward<Sender_>(predecessor_sender),
                        predecessor_sender_receiver{*this}))
                {
                }

                operation_state(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                struct predecessor_sender_receiver
                {
                    operation_state& os;

                    template <typename E>
                        void set_error(E&& e) && noexcept
                    {
                        os.set_error_predecessor_sender(std::forward<E>(e));
                    }

                    void set_done() && noexcept
                    {
                        os.set_done_predecessor_sender();
                    };

                    template <typename... Ts>
                        void set_value(Ts&&... ts) && noexcept
                    {
                        os.set_value_predecessor_sender(
                            std::forward<Ts>(ts)...);
                    }
                };

                template <typename E>
                void set_error_predecessor_sender(E&& e) noexcept
                {
                    hpx::execution::experimental::set_error(
                        std::move(receiver), std::forward<E>(e));
                }

                void set_done_predecessor_sender() noexcept
                {
                    hpx::execution::experimental::set_done(std::move(receiver));
                }

                template <typename... Us>
                void set_value_predecessor_sender(Us&&... us) noexcept
                {
                    ts.template emplace<hpx::tuple<Us...>>(
                        std::forward<Us>(us)...);
                    scheduler_os.template emplace(
                        hpx::util::detail::with_result_of([&]() {
                            return hpx::execution::experimental::connect(
                                hpx::execution::experimental::schedule(
                                    std::move(scheduler)),
                                scheduler_sender_receiver{*this});
                        }));
                    hpx::execution::experimental::start(scheduler_os.value());
                }

                struct scheduler_sender_receiver
                {
                    operation_state& os;

                    template <typename E>
                        void set_error(E&& e) && noexcept
                    {
                        os.set_error_scheduler_sender(std::forward<E>(e));
                    }

                    void set_done() && noexcept
                    {
                        os.set_done_scheduler_sender();
                    };

                    void set_value() && noexcept
                    {
                        os.set_value_scheduler_sender();
                    }
                };

                struct scheduler_sender_value_visitor
                {
                    std::decay_t<Receiver> receiver;

                    HPX_NORETURN void operator()(std::monostate) const
                    {
                        HPX_UNREACHABLE;
                    }

                    template <typename Ts,
                        typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<Ts>, std::monostate>>>
                    void operator()(Ts&& ts)
                    {
                        hpx::util::invoke_fused(
                            hpx::util::bind_front(
                                hpx::execution::experimental::set_value,
                                std::move(receiver)),
                            std::forward<Ts>(ts));
                    }
                };

                template <typename E>
                void set_error_scheduler_sender(E&& e) noexcept
                {
                    scheduler_os.reset();
                    hpx::execution::experimental::set_error(
                        std::move(receiver), std::forward<E>(e));
                }

                void set_done_scheduler_sender() noexcept
                {
                    scheduler_os.reset();
                    hpx::execution::experimental::set_done(std::move(receiver));
                }

                void set_value_scheduler_sender() noexcept
                {
                    scheduler_os.reset();
                    std::visit(
                        scheduler_sender_value_visitor{std::move(receiver)},
                        std::move(ts));
                }

                void start() & noexcept
                {
                    hpx::execution::experimental::start(sender_os);
                }
            };

            template <typename Receiver>
            operation_state<Receiver> connect(Receiver&& receiver) &&
            {
                return {std::move(predecessor_sender), std::move(scheduler),
                    std::forward<Receiver>(receiver)};
            }

            template <typename Receiver>
            operation_state<Receiver> connect(Receiver&& receiver) &
            {
                return {predecessor_sender, scheduler,
                    std::forward<Receiver>(receiver)};
            }
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct on_t final
      : hpx::functional::tag_fallback<on_t>
    {
    private:
        template <typename Sender, typename Scheduler>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            on_t, Sender&& predecessor_sender, Scheduler&& scheduler)
        {
            return detail::on_sender<Sender, Scheduler>{
                std::forward<Sender>(predecessor_sender),
                std::forward<Scheduler>(scheduler)};
        }

        template <typename Scheduler>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            on_t, Scheduler&& scheduler)
        {
            return detail::partial_algorithm<on_t, Scheduler>{
                std::forward<Scheduler>(scheduler)};
        }
    } on{};
}}}    // namespace hpx::execution::experimental
#endif
