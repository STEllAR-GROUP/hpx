//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/datastructures/variant.hpp>
#include <hpx/execution/algorithms/bulk.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution_base/completion_scheduler.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/executors/thread_pool_scheduler.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/functional/tag_dispatch.hpp>
#include <hpx/iterator_support/counting_iterator.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/threading_base/register_thread.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace execution { namespace experimental {
    namespace detail {
        template <typename Sender, typename Shape, typename F>
        struct thread_pool_bulk_sender
        {
            thread_pool_scheduler scheduler;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Shape> shape;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types =
                typename hpx::execution::experimental::sender_traits<
                    Sender>::template value_types<Tuple, Variant>;

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
                thread_pool_bulk_sender const& s)
            {
                if constexpr (std::is_same_v<std::decay_t<CPO>,
                                  hpx::execution::experimental::set_value_t>)
                {
                    return s.scheduler;
                }
                else
                {
                    return hpx::execution::experimental::
                        get_completion_scheduler<CPO>(s);
                }
            }

            template <typename Receiver>
            struct operation_state
            {
                struct bulk_receiver
                {
                    operation_state* op_state;

                    template <typename E>
                    friend void tag_dispatch(
                        set_error_t, bulk_receiver&& r, E&& e) noexcept
                    {
                        hpx::execution::experimental::set_error(
                            std::move(r.op_state->receiver),
                            std::forward<E>(e));
                    }

                    friend void tag_dispatch(
                        set_done_t, bulk_receiver&& r) noexcept
                    {
                        hpx::execution::experimental::set_done(
                            std::move(r.op_state->receiver));
                    };

                    template <typename Iterator>
                    struct set_value_loop_visitor
                    {
                        HPX_NO_UNIQUE_ADDRESS std::decay_t<Iterator> it;
                        operation_state* op_state;

                        void operator()(hpx::monostate const&) const
                        {
                            HPX_UNREACHABLE;
                        }

                        template <typename Ts,
                            typename = std::enable_if_t<!std::is_same_v<
                                std::decay_t<Ts>, hpx::monostate>>>
                        void operator()(Ts& ts)
                        {
                            hpx::util::invoke_fused(
                                hpx::util::bind_front(
                                    op_state->f, *std::move(it)),
                                ts);
                        }
                    };

                    struct set_value_end_loop_visitor
                    {
                        operation_state* op_state;

                        void operator()(hpx::monostate&&) const
                        {
                            std::terminate();
                        }

                        template <typename Ts,
                            typename = std::enable_if_t<!std::is_same_v<
                                std::decay_t<Ts>, hpx::monostate>>>
                        void operator()(Ts&& ts) const
                        {
                            hpx::util::invoke_fused(
                                hpx::util::bind_front(
                                    hpx::execution::experimental::set_value,
                                    std::move(op_state->receiver)),
                                std::forward<Ts>(ts));
                        }
                    };

                    using range_value_type = hpx::traits::iter_value_t<
                        hpx::traits::range_iterator_t<Shape>>;

                    template <typename... Ts,
                        typename = std::enable_if_t<
                            hpx::is_invocable_v<F, range_value_type,
                                std::add_lvalue_reference_t<Ts>...>>>
                    friend void tag_dispatch(
                        set_value_t, bulk_receiver&& r, Ts&&... ts) noexcept
                    {
                        auto const n = hpx::util::size(r.op_state->shape);

                        if (n == 0)
                        {
                            hpx::execution::experimental::set_value(
                                std::move(r.op_state->receiver),
                                std::forward<Ts>(ts)...);
                            return;
                        }

                        r.op_state->ts.template emplace<hpx::tuple<Ts...>>(
                            std::forward<Ts>(ts)...);

                        // TODO: chunking and hierarchical spawning?
                        using iterator_type =
                            hpx::traits::range_iterator_t<Shape>;
                        for (iterator_type it = std::begin(r.op_state->shape);
                             it != std::end(r.op_state->shape); ++it)
                        {
                            auto task_f = [op_state = r.op_state,
                                              it]() mutable {
                                try
                                {
                                    hpx::visit(
                                        set_value_loop_visitor<iterator_type>{
                                            std::move(it), op_state},
                                        op_state->ts);
                                }
                                catch (...)
                                {
                                    if (!op_state->exception_thrown.exchange(
                                            true))
                                    {
                                        op_state->exception =
                                            std::current_exception();
                                    }
                                }

                                if (--(op_state->tasks_remaining) == 0)
                                {
                                    if (op_state->exception_thrown)
                                    {
                                        HPX_ASSERT(
                                            op_state->exception.has_value());
                                        hpx::execution::experimental::set_error(
                                            std::move(op_state->receiver),
                                            std::move(
                                                op_state->exception.value()));
                                    }
                                    else
                                    {
                                        hpx::visit(
                                            set_value_end_loop_visitor{
                                                op_state},
                                            std::move(op_state->ts));
                                    }
                                }
                            };

                            threads::thread_init_data data(
                                threads::make_thread_function_nullary(task_f),
                                "thread_pool_bulk_sender task",
                                get_priority(r.op_state->scheduler),
                                get_hint(r.op_state->scheduler),
                                get_stacksize(r.op_state->scheduler));
                            threads::register_work(
                                data, r.op_state->scheduler.get_thread_pool());
                        }
                    }
                };

                using operation_state_type =
                    hpx::execution::experimental::connect_result_t<Sender,
                        bulk_receiver>;

                thread_pool_scheduler scheduler;
                operation_state_type op_state;
                HPX_NO_UNIQUE_ADDRESS std::decay_t<Shape> shape;
                HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;
                HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
                std::atomic<decltype(hpx::util::size(shape))> tasks_remaining{
                    hpx::util::size(shape)};
                hpx::util::detail::prepend_t<
                    value_types<hpx::tuple, hpx::variant>, hpx::monostate>
                    ts;
                std::atomic<bool> exception_thrown{false};
                std::optional<std::exception_ptr> exception;

                template <typename Sender_, typename Shape_, typename F_,
                    typename Receiver_>
                operation_state(thread_pool_scheduler&& scheduler,
                    Sender_&& sender, Shape_&& shape, F_&& f,
                    Receiver_&& receiver)
                  : scheduler(std::move(scheduler))
                  , op_state(hpx::execution::experimental::connect(
                        std::forward<Sender_>(sender), bulk_receiver{this}))
                  , shape(std::forward<Shape_>(shape))
                  , f(std::forward<F_>(f))
                  , receiver(std::forward<Receiver_>(receiver))
                {
                }

                friend void tag_dispatch(start_t, operation_state& os) noexcept
                {
                    hpx::execution::experimental::start(os.op_state);
                }
            };

            template <typename Receiver>
            friend auto tag_dispatch(
                connect_t, thread_pool_bulk_sender&& s, Receiver&& receiver)
            {
                return operation_state<std::decay_t<Receiver>>{
                    std::move(s.scheduler), std::move(s.sender),
                    std::move(s.shape), std::move(s.f),
                    std::forward<Receiver>(receiver)};
            }

            template <typename Receiver>
            auto tag_dispatch(
                connect_t, thread_pool_bulk_sender& s, Receiver&& receiver)
            {
                return operation_state<std::decay_t<Receiver>>{s.scheduler,
                    s.sender, s.shape, s.f, std::forward<Receiver>(receiver)};
            }
        };
    }    // namespace detail

    template <typename Sender, typename Shape, typename F,
        HPX_CONCEPT_REQUIRES_(std::is_integral_v<std::decay_t<Shape>>)>
    constexpr auto tag_dispatch(bulk_t, thread_pool_scheduler scheduler,
        Sender&& sender, Shape&& shape, F&& f)
    {
        return detail::thread_pool_bulk_sender<std::decay_t<Sender>,
            hpx::util::detail::counting_shape_type<std::decay_t<Shape>>,
            std::decay_t<F>>{std::move(scheduler), std::forward<Sender>(sender),
            hpx::util::detail::make_counting_shape(shape), std::forward<F>(f)};
    }

    template <typename Sender, typename Shape, typename F,
        HPX_CONCEPT_REQUIRES_(!std::is_integral_v<std::decay_t<Shape>>)>
    constexpr auto tag_dispatch(bulk_t, thread_pool_scheduler scheduler,
        Sender&& sender, Shape&& shape, F&& f)
    {
        return detail::thread_pool_bulk_sender<std::decay_t<Sender>,
            std::decay_t<Shape>, std::decay_t<F>>{std::move(scheduler),
            std::forward<Sender>(sender), std::forward<Shape>(shape),
            std::forward<F>(f)};
    }
}}}    // namespace hpx::execution::experimental
