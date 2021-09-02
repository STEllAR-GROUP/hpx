//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/datastructures/optional.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/datastructures/variant.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution_base/completion_scheduler.hpp>
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

namespace hpx { namespace execution { namespace experimental {
    namespace detail {
        template <typename Sender, typename Scheduler>
        struct on_sender
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Sender> predecessor_sender;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Scheduler> scheduler;

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

            template <typename CPO,
                // clang-format off
                HPX_CONCEPT_REQUIRES_(
                    hpx::execution::experimental::detail::is_receiver_cpo_v<CPO> &&
                    (std::is_same_v<CPO, hpx::execution::experimental::set_value_t> ||
                        hpx::execution::experimental::detail::has_completion_scheduler_v<
                                hpx::execution::experimental::set_error_t,
                                std::decay_t<Sender>> ||
                        hpx::execution::experimental::detail::has_completion_scheduler_v<
                                hpx::execution::experimental::set_done_t,
                                std::decay_t<Sender>>))
                // clang-format on
                >
            friend constexpr auto tag_dispatch(
                hpx::execution::experimental::get_completion_scheduler_t<CPO>,
                on_sender const& sender)
            {
                if constexpr (std::is_same_v<std::decay_t<CPO>,
                                  hpx::execution::experimental::set_value_t>)
                {
                    return sender.scheduler;
                }
                else
                {
                    return hpx::execution::experimental::
                        get_completion_scheduler<CPO>(
                            sender.predecessor_sender);
                }
            }

            template <typename Receiver>
            struct operation_state
            {
                HPX_NO_UNIQUE_ADDRESS std::decay_t<Scheduler> scheduler;
                HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

                struct predecessor_sender_receiver;
                struct scheduler_sender_receiver;

                using value_type = hpx::util::detail::prepend_t<
                    typename hpx::execution::experimental::sender_traits<
                        Sender>::template value_types<hpx::tuple, hpx::variant>,
                    hpx::monostate>;
                value_type ts;

                using sender_operation_state_type =
                    connect_result_t<Sender, predecessor_sender_receiver>;
                sender_operation_state_type sender_os;

                using scheduler_operation_state_type =
                    connect_result_t<typename hpx::util::invoke_result<
                                         schedule_t, Scheduler>::type,
                        scheduler_sender_receiver>;
                hpx::util::optional<scheduler_operation_state_type>
                    scheduler_op_state;

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
                    operation_state& op_state;

                    template <typename Error>
                    friend void tag_dispatch(set_error_t,
                        predecessor_sender_receiver&& r, Error&& error) noexcept
                    {
                        r.op_state.set_error_predecessor_sender(
                            std::forward<Error>(error));
                    }

                    friend void tag_dispatch(
                        set_done_t, predecessor_sender_receiver&& r) noexcept
                    {
                        r.op_state.set_done_predecessor_sender();
                    }

                    // This typedef is duplicated from the parent struct. The
                    // parent typedef is not instantiated early enough for use
                    // here.
                    using value_type = hpx::util::detail::prepend_t<
                        typename hpx::execution::experimental::sender_traits<
                            Sender>::template value_types<hpx::tuple,
                            hpx::variant>,
                        hpx::monostate>;

                    template <typename... Ts>
                    friend auto tag_dispatch(set_value_t,
                        predecessor_sender_receiver&& r, Ts&&... ts) noexcept
                        -> decltype(std::declval<value_type>()
                                        .template emplace<hpx::tuple<Ts...>>(
                                            std::forward<Ts>(ts)...),
                            void())
                    {
                        r.op_state.set_value_predecessor_sender(
                            std::forward<Ts>(ts)...);
                    }
                };

                template <typename Error>
                void set_error_predecessor_sender(Error&& error) noexcept
                {
                    hpx::execution::experimental::set_error(
                        std::move(receiver), std::forward<Error>(error));
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
#if defined(HPX_HAVE_CXX17_COPY_ELISION)
                    // with_result_of is used to emplace the operation
                    // state returned from connect without any
                    // intermediate copy construction (the operation
                    // state is not required to be copyable nor movable).
                    scheduler_op_state.emplace(
                        hpx::util::detail::with_result_of([&]() {
                            return hpx::execution::experimental::connect(
                                hpx::execution::experimental::schedule(
                                    std::move(scheduler)),
                                scheduler_sender_receiver{*this});
                        }));
#else
                    // MSVC doesn't get copy elision quite right, the operation
                    // state must be constructed explicitly directly in place
                    scheduler_op_state.emplace_f(
                        hpx::execution::experimental::connect,
                        hpx::execution::experimental::schedule(
                            std::move(scheduler)),
                        scheduler_sender_receiver{*this});
#endif
                    hpx::execution::experimental::start(
                        scheduler_op_state.value());
                }

                struct scheduler_sender_receiver
                {
                    operation_state& op_state;

                    template <typename Error>
                    friend void tag_dispatch(set_error_t,
                        scheduler_sender_receiver&& r, Error&& error) noexcept
                    {
                        r.op_state.set_error_scheduler_sender(
                            std::forward<Error>(error));
                    }

                    friend void tag_dispatch(
                        set_done_t, scheduler_sender_receiver&& r) noexcept
                    {
                        r.op_state.set_done_scheduler_sender();
                    }

                    friend void tag_dispatch(
                        set_value_t, scheduler_sender_receiver&& r) noexcept
                    {
                        r.op_state.set_value_scheduler_sender();
                    }
                };

                struct scheduler_sender_value_visitor
                {
                    HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

                    HPX_NORETURN void operator()(hpx::monostate) const
                    {
                        HPX_UNREACHABLE;
                    }

                    template <typename Ts,
                        typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<Ts>, hpx::monostate>>>
                    void operator()(Ts&& ts)
                    {
                        hpx::util::invoke_fused(
                            hpx::util::bind_front(
                                hpx::execution::experimental::set_value,
                                std::move(receiver)),
                            std::forward<Ts>(ts));
                    }
                };

                template <typename Error>
                void set_error_scheduler_sender(Error&& error) noexcept
                {
                    scheduler_op_state.reset();
                    hpx::execution::experimental::set_error(
                        std::move(receiver), std::forward<Error>(error));
                }

                void set_done_scheduler_sender() noexcept
                {
                    scheduler_op_state.reset();
                    hpx::execution::experimental::set_done(std::move(receiver));
                }

                void set_value_scheduler_sender() noexcept
                {
                    scheduler_op_state.reset();
                    hpx::visit(
                        scheduler_sender_value_visitor{std::move(receiver)},
                        std::move(ts));
                }

                friend void tag_dispatch(start_t, operation_state& os) noexcept
                {
                    hpx::execution::experimental::start(os.sender_os);
                }
            };

            template <typename Receiver>
            friend operation_state<Receiver> tag_dispatch(
                connect_t, on_sender&& s, Receiver&& receiver)
            {
                return {std::move(s.predecessor_sender), std::move(s.scheduler),
                    std::forward<Receiver>(receiver)};
            }

            template <typename Receiver>
            friend operation_state<Receiver> tag_dispatch(
                connect_t, on_sender& s, Receiver&& receiver)
            {
                return {s.predecessor_sender, s.scheduler,
                    std::forward<Receiver>(receiver)};
            }
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct on_t final
      : hpx::functional::tag_fallback<on_t>
    {
    private:
        // clang-format off
        template <typename Sender, typename Scheduler,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender>
            )>
        // clang-format on
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
