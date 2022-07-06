//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/concurrency/detail/contiguous_index_queue.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/datastructures/variant.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/algorithms/bulk.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution_base/completion_scheduler.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/executors/thread_pool_scheduler.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/iterator_support/counting_iterator.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/threading_base/register_thread.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::execution::experimental {

    namespace detail {
        /// This sender represents bulk work that will be performed using the
        /// thread_pool_scheduler.
        ///
        /// The work is chunked into a number of chunks larger than the number
        /// of worker threads available on the underlying thread pool. The
        /// chunks are then assigned to worker thread-specific thread-safe index
        /// queues. One HPX thread is spawned for each underlying worker (OS)
        /// thread. The HPX thread is responsible for work in one queue. If the
        /// queue is empty, no HPX thread will be spawned. Once the HPX thread
        /// has finished working on its own queue, it will attempt to steal work
        /// from other queues. Since predecessor sender must complete on an HPX
        /// thread (the completion scheduler is a thread_pool_scheduler;
        /// otherwise the customization defined in this file is not chosen) it
        /// will be reused as one of the worker threads.
        template <typename Policy, typename Sender, typename Shape, typename F>
        class thread_pool_bulk_sender
        {
        private:
            thread_pool_policy_scheduler<Policy> scheduler;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Shape> shape;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;

            using size_type = decltype(hpx::util::size(shape));

        public:
            template <typename Sender_, typename Shape_, typename F_>
            thread_pool_bulk_sender(
                thread_pool_policy_scheduler<Policy>&& scheduler,
                Sender_&& sender, Shape_&& shape, F_&& f)
              : scheduler(HPX_MOVE(scheduler))
              , sender(HPX_FORWARD(Sender_, sender))
              , shape(HPX_FORWARD(Shape_, shape))
              , f(HPX_FORWARD(F_, f))
            {
            }

            thread_pool_bulk_sender(thread_pool_bulk_sender&&) = default;
            thread_pool_bulk_sender(thread_pool_bulk_sender const&) = default;
            thread_pool_bulk_sender& operator=(
                thread_pool_bulk_sender&&) = default;
            thread_pool_bulk_sender& operator=(
                thread_pool_bulk_sender const&) = default;

            template <typename Env>
            struct generate_completion_signatures
            {
                template <template <typename...> typename Tuple,
                    template <typename...> typename Variant>
                using value_types =
                    value_types_of_t<Sender, Env, Tuple, Variant>;

                template <template <typename...> typename Variant>
                using error_types = hpx::util::detail::unique_concat_t<
                    error_types_of_t<Sender, Env, Variant>,
                    Variant<std::exception_ptr>>;

                static constexpr bool sends_stopped = false;
            };

            template <typename Env>
            friend auto tag_invoke(get_completion_signatures_t,
                thread_pool_bulk_sender const&, Env)
                -> generate_completion_signatures<Env>;

            // clang-format off
            template <typename CPO,
                HPX_CONCEPT_REQUIRES_(
                    hpx::execution::experimental::detail::is_receiver_cpo_v<CPO> &&
                    (std::is_same_v<CPO, hpx::execution::experimental::set_value_t> ||
                        hpx::execution::experimental::detail::has_completion_scheduler_v<
                                hpx::execution::experimental::set_error_t,
                                std::decay_t<Sender>> ||
                        hpx::execution::experimental::detail::has_completion_scheduler_v<
                                hpx::execution::experimental::set_stopped_t,
                                std::decay_t<Sender>>)
                )>
            // clang-format on
            friend constexpr auto tag_invoke(
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

        private:
            template <typename Receiver>
            struct operation_state
            {
                struct bulk_receiver
                {
                    operation_state* op_state;

                    template <typename E>
                    friend void tag_invoke(
                        set_error_t, bulk_receiver&& r, E&& e) noexcept
                    {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(r.op_state->receiver), HPX_FORWARD(E, e));
                    }

                    friend void tag_invoke(
                        set_stopped_t, bulk_receiver&& r) noexcept
                    {
                        hpx::execution::experimental::set_stopped(
                            HPX_MOVE(r.op_state->receiver));
                    };

                    struct task_function;

                    struct set_value_loop_visitor
                    {
                        operation_state* const op_state;
                        task_function const* const task_f;

                        void operator()(hpx::monostate const&) const
                        {
                            HPX_UNREACHABLE;
                        }

                        // Perform the work in one chunk indexed by index.  The
                        // index represents a range of indices (iterators) in
                        // the given shape.
                        template <typename Ts>
                        void do_work_chunk(
                            Ts& ts, std::uint32_t const index) const
                        {
                            auto const i_begin = static_cast<size_type>(index) *
                                task_f->chunk_size;
                            auto const i_end =
                                (std::min)(static_cast<size_type>(index + 1) *
                                        task_f->chunk_size,
                                    task_f->n);
                            auto it = hpx::util::begin(op_state->shape);
                            std::advance(it, i_begin);
                            for (size_type i = i_begin; i < i_end; ++i)
                            {
                                hpx::util::invoke_fused(
                                    hpx::bind_front(op_state->f, *it), ts);
                                ++it;
                            }
                        }

                        // Visit the values sent from the predecessor sender.
                        // This function first tries to handle all chunks in the
                        // queue owned by worker_thread. It then tries to steal
                        // chunks from neighboring threads.
                        template <typename Ts,
                            typename = std::enable_if_t<!std::is_same_v<
                                std::decay_t<Ts>, hpx::monostate>>>
                        void operator()(Ts& ts) const
                        {
                            auto& local_queue =
                                op_state->queues[task_f->worker_thread].data_;

                            // Handle local queue first
                            hpx::optional<std::uint32_t> index;
                            while ((index = local_queue.pop_left()))
                            {
                                do_work_chunk(ts, index.value());
                            }

                            // Then steal from neighboring queues
                            for (std::uint32_t offset = 1;
                                 offset < op_state->num_worker_threads;
                                 ++offset)
                            {
                                std::size_t neighbor_worker_thread =
                                    (task_f->worker_thread + offset) %
                                    op_state->num_worker_threads;
                                auto& neighbor_queue =
                                    op_state->queues[neighbor_worker_thread]
                                        .data_;

                                while ((index = neighbor_queue.pop_right()))
                                {
                                    do_work_chunk(ts, index.value());
                                }
                            }
                        }
                    };

                    struct set_value_end_loop_visitor
                    {
                        operation_state* const op_state;

                        void operator()(hpx::monostate&&) const
                        {
                            std::terminate();
                        }

                        // Visit the values sent from the predecessor sender.
                        // This function is called once all worker threads have
                        // processed their chunks and the connected receiver
                        // should be signalled.
                        template <typename Ts,
                            typename = std::enable_if_t<!std::is_same_v<
                                std::decay_t<Ts>, hpx::monostate>>>
                        void operator()(Ts&& ts) const
                        {
                            hpx::util::invoke_fused(
                                hpx::bind_front(
                                    hpx::execution::experimental::set_value,
                                    HPX_MOVE(op_state->receiver)),
                                HPX_FORWARD(Ts, ts));
                        }
                    };

                    // This struct encapsulates the work done by one worker thread.
                    struct task_function
                    {
                        operation_state* const op_state;
                        size_type const n;
                        std::uint32_t const chunk_size;
                        std::uint32_t const worker_thread;

                        // Visit the values sent by the predecessor sender.
                        void do_work() const
                        {
                            hpx::visit(set_value_loop_visitor{op_state, this},
                                op_state->ts);
                        }

                        // Store an exception and mark that an exception was
                        // thrown in the operation state. This function assumes
                        // that there is a current exception.
                        void store_exception() const
                        {
                            if (!op_state->exception_thrown.exchange(true))
                            {
                                // NOLINTNEXTLINE(bugprone-throw-keyword-missing)
                                op_state->exception = std::current_exception();
                            }
                        }

                        // Finish the work for one worker thread. If this is not
                        // the last worker thread to finish, it will only
                        // decrement the counter. If it is the last thread it
                        // will call set_error if there is an exception.
                        // Otherwise it will call set_value on the connected
                        // receiver.
                        void finish() const
                        {
                            if (--(op_state->tasks_remaining) == 0)
                            {
                                if (op_state->exception_thrown)
                                {
                                    HPX_ASSERT(op_state->exception.has_value());
                                    hpx::execution::experimental::set_error(
                                        HPX_MOVE(op_state->receiver),
                                        HPX_MOVE(op_state->exception.value()));
                                }
                                else
                                {
                                    hpx::visit(
                                        set_value_end_loop_visitor{op_state},
                                        HPX_MOVE(op_state->ts));
                                }
                            }
                        }

                        // Entry point for the worker thread. It will attempt to
                        // do its local work, catch any exceptions, and then
                        // call set_value or set_error on the connected
                        // receiver.
                        void operator()()
                        {
                            try
                            {
                                do_work();
                            }
                            catch (...)
                            {
                                store_exception();
                            }

                            finish();
                        };
                    };

                    // Compute a chunk size given a number of worker threads and
                    // a total number of items n. Returns a power-of-2 chunk
                    // size that produces at most 8 and at least 4 chunks per
                    // worker thread.
                    static constexpr std::uint32_t get_chunk_size(
                        std::uint32_t const num_threads, size_type const n)
                    {
                        std::uint32_t chunk_size = 1;
                        while (chunk_size * num_threads * 8 < n)
                        {
                            chunk_size *= 2;
                        }
                        return chunk_size;
                    }

                    // Initialize a queue for a worker thread.
                    void init_queue(std::uint32_t const worker_thread,
                        std::uint32_t const num_chunks)
                    {
                        auto& queue = op_state->queues[worker_thread].data_;
                        auto const part_begin = static_cast<std::uint32_t>(
                            (worker_thread * num_chunks) /
                            op_state->num_worker_threads);
                        auto const part_end = static_cast<std::uint32_t>(
                            ((worker_thread + 1) * num_chunks) /
                            op_state->num_worker_threads);
                        queue.reset(part_begin, part_end);
                    }

                    // Spawn a task which will process a number of chunks. If
                    // the queue contains no chunks no task will be spawned.
                    void do_work_task(size_type const n,
                        std::uint32_t const chunk_size,
                        std::uint32_t const worker_thread) const
                    {
                        task_function task_f{
                            this->op_state, n, chunk_size, worker_thread};

                        auto& queue = op_state->queues[worker_thread].data_;
                        if (queue.empty())
                        {
                            // If the queue is empty we don't spawn a task. We
                            // only signal that this "task" is ready.
                            task_f.finish();
                            return;
                        }

                        // Only apply hint if none was given.
                        auto hint = get_hint(op_state->scheduler);
                        if (hint == hpx::threads::thread_schedule_hint())
                        {
                            hint = hpx::threads::thread_schedule_hint(
                                hpx::threads::thread_schedule_hint_mode::thread,
                                worker_thread);
                        }

                        // Spawn the task.
                        char const* scheduler_annotation =
                            get_annotation(op_state->scheduler);
                        char const* annotation =
                            scheduler_annotation == nullptr ?
                            traits::get_function_annotation<
                                std::decay_t<F>>::call(op_state->f) :
                            scheduler_annotation;

                        threads::thread_init_data data(
                            threads::make_thread_function_nullary(
                                HPX_MOVE(task_f)),
                            annotation, get_priority(op_state->scheduler), hint,
                            get_stacksize(op_state->scheduler));
                        threads::register_work(
                            data, op_state->scheduler.get_thread_pool());
                    }

                    // Do the work on the worker thread that called set_value
                    // from the predecessor sender. This thread participates in
                    // the work and does not need a new task since it already
                    // runs on a task.
                    void do_work_local(size_type n, std::uint32_t chunk_size,
                        std::uint32_t worker_thread) const
                    {
                        char const* scheduler_annotation =
                            get_annotation(op_state->scheduler);
                        if (scheduler_annotation)
                        {
                            hpx::scoped_annotation ann(scheduler_annotation);
                            task_function{
                                this->op_state, n, chunk_size, worker_thread}();
                        }
                        else
                        {
                            hpx::scoped_annotation ann(op_state->f);
                            task_function{
                                this->op_state, n, chunk_size, worker_thread}();
                        }
                    }

                    using range_value_type = hpx::traits::iter_value_t<
                        hpx::traits::range_iterator_t<Shape>>;

                    template <typename... Ts,
                        typename = std::enable_if_t<
                            hpx::is_invocable_v<F, range_value_type,
                                std::add_lvalue_reference_t<Ts>...>>>
                    friend void tag_invoke(
                        set_value_t, bulk_receiver&& r, Ts&&... ts) noexcept
                    {
                        hpx::detail::try_catch_exception_ptr(
                            [&]() {
                                // Don't spawn tasks if there is no work to be
                                // done
                                auto const n =
                                    hpx::util::size(r.op_state->shape);
                                if (n == 0)
                                {
                                    hpx::execution::experimental::set_value(
                                        HPX_MOVE(r.op_state->receiver),
                                        HPX_FORWARD(Ts, ts)...);
                                    return;
                                }

                                // Calculate chunk size and number of chunks
                                auto const chunk_size = get_chunk_size(
                                    std::uint32_t(
                                        r.op_state->num_worker_threads),
                                    n);
                                auto const num_chunks = std::uint32_t(
                                    (n + chunk_size - 1) / chunk_size);

                                // Store sent values in the operation state
                                r.op_state->ts
                                    .template emplace<hpx::tuple<Ts...>>(
                                        HPX_FORWARD(Ts, ts)...);

                                // Initialize the queues for all worker threads
                                // so that worker threads can start stealing
                                // immediately when they start.
                                for (std::uint32_t worker_thread = 0;
                                     worker_thread <
                                     r.op_state->num_worker_threads;
                                     ++worker_thread)
                                {
                                    r.init_queue(worker_thread, num_chunks);
                                }

                                // Spawn the worker threads for all except the
                                // local queue.
                                auto const local_worker_thread = std::uint32_t(
                                    hpx::get_local_worker_thread_num());
                                for (std::uint32_t worker_thread = 0;
                                     worker_thread <
                                     r.op_state->num_worker_threads;
                                     ++worker_thread)
                                {
                                    // The queue for the local thread is handled
                                    // later inline.
                                    if (worker_thread == local_worker_thread)
                                    {
                                        continue;
                                    }

                                    r.do_work_task(
                                        n, chunk_size, worker_thread);
                                }

                                // Handle the queue for the local thread.
                                r.do_work_local(
                                    n, chunk_size, local_worker_thread);
                            },
                            [&](std::exception_ptr ep) {
                                hpx::execution::experimental::set_error(
                                    HPX_MOVE(r.op_state->receiver),
                                    HPX_MOVE(ep));
                            });
                    }
                };

                using operation_state_type =
                    hpx::execution::experimental::connect_result_t<Sender,
                        bulk_receiver>;

                thread_pool_policy_scheduler<Policy> scheduler;
                operation_state_type op_state;
                std::size_t num_worker_threads =
                    scheduler.get_thread_pool()->get_os_thread_count();
                std::vector<hpx::util::cache_aligned_data<
                    hpx::concurrency::detail::contiguous_index_queue<>>>
                    queues{num_worker_threads};
                HPX_NO_UNIQUE_ADDRESS std::decay_t<Shape> shape;
                HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;
                HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
                std::atomic<decltype(hpx::util::size(shape))> tasks_remaining{
                    num_worker_threads};

                hpx::util::detail::prepend_t<value_types_of_t<Sender, empty_env,
                                                 decayed_tuple, hpx::variant>,
                    hpx::monostate>
                    ts;
                std::atomic<bool> exception_thrown{false};
                std::optional<std::exception_ptr> exception;

                template <typename Sender_, typename Shape_, typename F_,
                    typename Receiver_>
                operation_state(
                    thread_pool_policy_scheduler<Policy>&& scheduler,
                    Sender_&& sender, Shape_&& shape, F_&& f,
                    Receiver_&& receiver)
                  : scheduler(HPX_MOVE(scheduler))
                  , op_state(hpx::execution::experimental::connect(
                        HPX_FORWARD(Sender_, sender), bulk_receiver{this}))
                  , shape(HPX_FORWARD(Shape_, shape))
                  , f(HPX_FORWARD(F_, f))
                  , receiver(HPX_FORWARD(Receiver_, receiver))
                {
                }

                friend void tag_invoke(start_t, operation_state& os) noexcept
                {
                    hpx::execution::experimental::start(os.op_state);
                }
            };

        public:
            template <typename Receiver>
            friend auto tag_invoke(
                connect_t, thread_pool_bulk_sender&& s, Receiver&& receiver)
            {
                return operation_state<std::decay_t<Receiver>>{
                    HPX_MOVE(s.scheduler), HPX_MOVE(s.sender),
                    HPX_MOVE(s.shape), HPX_MOVE(s.f),
                    HPX_FORWARD(Receiver, receiver)};
            }

            template <typename Receiver>
            auto tag_invoke(
                connect_t, thread_pool_bulk_sender& s, Receiver&& receiver)
            {
                return operation_state<std::decay_t<Receiver>>{s.scheduler,
                    s.sender, s.shape, s.f, HPX_FORWARD(Receiver, receiver)};
            }
        };
    }    // namespace detail

    // clang-format off
    template <typename Policy, typename Sender, typename Shape, typename F,
        HPX_CONCEPT_REQUIRES_(
            !std::is_integral_v<std::decay_t<Shape>>
        )>
    // clang-format on
    constexpr auto tag_invoke(bulk_t,
        thread_pool_policy_scheduler<Policy> scheduler, Sender&& sender,
        Shape&& shape, F&& f)
    {
        if constexpr (std::is_same_v<Policy, launch::sync_policy>)
        {
            return hpx::functional::detail::tag_fallback_invoke(bulk_t{},
                HPX_FORWARD(Sender, sender), HPX_FORWARD(Shape, shape),
                HPX_FORWARD(F, f));
        }
        else
        {
            return detail::thread_pool_bulk_sender<Policy, std::decay_t<Sender>,
                std::decay_t<Shape>, std::decay_t<F>>{HPX_MOVE(scheduler),
                HPX_FORWARD(Sender, sender), HPX_FORWARD(Shape, shape),
                HPX_FORWARD(F, f)};
        }
    }

    // clang-format off
    template <typename Policy, typename Sender, typename Shape, typename F,
        HPX_CONCEPT_REQUIRES_(
            std::is_integral_v<std::decay_t<Shape>>
        )>
    // clang-format on
    constexpr decltype(auto) tag_invoke(bulk_t,
        thread_pool_policy_scheduler<Policy> scheduler, Sender&& sender,
        Shape&& shape, F&& f)
    {
        return tag_invoke(bulk_t{}, HPX_MOVE(scheduler),
            HPX_FORWARD(Sender, sender),
            hpx::util::detail::make_counting_shape(shape), HPX_FORWARD(F, f));
    }
}    // namespace hpx::execution::experimental
