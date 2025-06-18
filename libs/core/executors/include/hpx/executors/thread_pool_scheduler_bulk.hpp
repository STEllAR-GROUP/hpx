//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2022-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if defined(HPX_HAVE_STDEXEC)
#include <hpx/execution_base/stdexec_forward.hpp>
#else
static_assert(false, "This code requires HPX_HAVE_STDEXEC to be defined");
#endif

#include <hpx/assert.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/concurrency/cache_line_data.hpp>
#include <hpx/concurrency/detail/non_contiguous_index_queue.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/datastructures/variant.hpp>
#include <hpx/errors/exception.hpp>
#include <hpx/errors/exception_list.hpp>
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
#include <hpx/resource_partitioner/detail/partitioner.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/topology/cpu_mask.hpp>
#include <hpx/type_support/pack.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::execution::experimental::detail {

    ///////////////////////////////////////////////////////////////////////////
    // Compute a chunk size given a number of worker threads and a total number
    // of items n. Returns a power-of-2 chunk size that produces at most 8 and
    // at least 4 chunks per worker thread.
    static constexpr std::uint32_t get_bulk_scheduler_chunk_size(
        std::uint32_t const num_threads, std::size_t const n) noexcept
    {
        std::uint64_t chunk_size = 1;
        while (chunk_size * num_threads * 8 < n)
        {
            chunk_size *= 2;
        }
        return static_cast<std::uint32_t>(chunk_size);
    }

    // Helper for invoking f with either range-based (begin, end
    // or single-index arguments
    template <std::size_t... Is, typename F, typename T, typename... Ts>
    constexpr void bulk_scheduler_invoke_helper(
        std::index_sequence<Is...>, F&& f, T&& t, bool is_range, Ts&&... ts)
    {
        if (is_range)
        {
            // Range-based invocation: f(begin, end, ts...)
            HPX_INVOKE(
                HPX_FORWARD(F, f), t.first, t.second, std::forward<Ts>(ts)...);
        }
        else
        {
            // Single-index invocation: f(index, ts...)
            HPX_INVOKE(
                HPX_FORWARD(F, f), HPX_FORWARD(T, t), std::forward<Ts>(ts)...);
        }
    }

    inline hpx::threads::mask_type full_mask(
        std::size_t first_thread, std::size_t num_threads)
    {
        auto const& rp = hpx::resource::get_partitioner();

        std::size_t const overall_threads =
            hpx::threads::hardware_concurrency();
        auto mask = hpx::threads::mask_type();
        hpx::threads::resize(mask, overall_threads);
        for (std::size_t i = 0; i != num_threads; ++i)
        {
            auto thread_mask = rp.get_pu_mask(i + first_thread);
            for (std::size_t j = 0; j != overall_threads; ++j)
            {
                if (threads::test(thread_mask, j))
                {
                    threads::set(mask, j);
                }
            }
        }
        std::cout << "Full mask created: first_thread=" << first_thread
                  << ", num_threads=" << num_threads << std::endl;
        return mask;
    }

    inline hpx::threads::mask_type limit_mask(
        hpx::threads::mask_cref_type orgmask, std::size_t num_threads)
    {
        std::size_t const num_cores = hpx::threads::hardware_concurrency();

        auto mask = hpx::threads::mask_type();
        hpx::threads::resize(mask, num_cores);
        for (std::size_t i = 0, j = 0; i != num_threads && j != num_cores; ++j)
        {
            if (hpx::threads::test(orgmask, j))
            {
                hpx::threads::set(mask, j);
                ++i;
            }
        }
        std::cout << "Limit mask created: num_threads=" << num_threads
                  << std::endl;
        return mask;
    }

    template <typename OperationState, typename F, typename Shape>
    struct task_function;

    // Adapter for single-index functions to support range-based invocation
    template <typename F>
    struct single_index_adapter
    {
        F f;

        template <typename... Ts>
        void operator()(
            std::uint32_t begin, std::uint32_t end, Ts&&... ts) const
        {
            for (std::uint32_t i = begin; i < end; ++i)
            {
                auto index_pack = std::index_sequence_for<Ts...>{};
                bulk_scheduler_invoke_helper(
                    index_pack, f, i, false, std::forward<Ts>(ts)...);
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////
    // Receiver for bulk execution
    // The user-supplied function f 
    //is expected to accept f(uint32_t begin, uint32_t end, Ts&&...)
    // for range-based processing in chunked mode.
    // For backward compatibility, single-index
    // functions f(index, Ts&&...) are supported via single_index_adapter.
    template <typename OperationState, typename F, typename Shape>
    struct bulk_receiver
    {
#if defined(HPX_HAVE_STDEXEC)
        using receiver_concept = hpx::execution::experimental::receiver_t;
#endif
        OperationState* op_state;

        template <typename E>
        friend void tag_invoke(hpx::execution::experimental::set_error_t,
            bulk_receiver&& r, E&& e) noexcept
        {
            hpx::execution::experimental::set_error(
                HPX_MOVE(r.op_state->receiver), HPX_FORWARD(E, e));
        }

        friend void tag_invoke(hpx::execution::experimental::set_stopped_t,
            bulk_receiver&& r) noexcept
        {
            hpx::execution::experimental::set_stopped(
                HPX_MOVE(r.op_state->receiver));
        }

        // Initialize a queue for a worker thread.
        void init_queue_depth_first(std::uint32_t const worker_thread,
            std::uint32_t const size, std::uint32_t num_threads) noexcept
        {
            auto& queue = op_state->queues[worker_thread].data_;
            auto const part_begin = static_cast<std::uint32_t>(
                (worker_thread * size) / num_threads);
            auto const part_end = static_cast<std::uint32_t>(
                ((worker_thread + 1) * size) / num_threads);
            queue.reset(part_begin, part_end);
            std::cout << "Depth-first queue initialized: worker_thread="
                      << worker_thread << ", part_begin=" << part_begin
                      << ", part_end=" << part_end
                      << ", num_threads=" << num_threads << std::endl;
        }

        void init_queue_breadth_first(std::uint32_t const worker_thread,
            std::uint32_t const size, std::uint32_t num_threads) noexcept
        {
            auto& queue = op_state->queues[worker_thread].data_;
            auto const num_steps = size / num_threads + 1;
            auto const part_begin = worker_thread;
            auto part_end = (std::min) (size + num_threads - 1,
                part_begin + num_steps * num_threads);
            auto const remainder = (part_end - part_begin) % num_threads;
            if (remainder != 0)
            {
                part_end -= remainder;
            }
            queue.reset(part_begin, part_end, num_threads);
            std::cout << "Breadth-first queue initialized: worker_thread="
                      << worker_thread << ", part_begin=" << part_begin
                      << ", part_end=" << part_end
                      << ", num_threads=" << num_threads << std::endl;
        }

        // Spawn a task which will process a number of chunks. If the queue
        // contains no chunks no task will be spawned.
        template <typename Task>
        void do_work_task(Task&& task_f) const
        {
            std::uint32_t const worker_thread = task_f.worker_thread;
            auto& queue = op_state->queues[worker_thread].data_;
            if (queue.empty())
            {
                // If the queue is empty we don't spawn a task. We only signal
                // that this "task" is ready.
                std::cout << "Queue empty for worker_thread=" << worker_thread
                          << ", finishing task" << std::endl;
                task_f.finish();
                return;
            }

            auto hint =
                hpx::execution::experimental::get_hint(op_state->scheduler);
            if (hint.mode == hpx::threads::thread_schedule_hint_mode::none &&
                hint.hint == -1)
            {
                // apply hint if none was given
                hint.mode = hpx::threads::thread_schedule_hint_mode::thread;
                hint.hint = worker_thread + op_state->first_thread;
                std::cout << "Applying thread hint: worker_thread="
                          << worker_thread << ", hint=" << hint.hint
                          << ", first_thread=" << op_state->first_thread
                          << std::endl;

                auto policy = hpx::execution::experimental::with_hint(
                    op_state->scheduler.policy(), hint);

                std::cout << "Executing task for worker_thread="
                          << worker_thread << " with policy hint" << std::endl;
                op_state->scheduler.execute(HPX_FORWARD(Task, task_f), policy);
            }
            else
            {
                std::cout << "Using existing hint: worker_thread="
                          << worker_thread
                          << ", mode=" << static_cast<int>(hint.mode)
                          << ", hint=" << hint.hint << std::endl;
                op_state->scheduler.execute(HPX_FORWARD(Task, task_f));
            }
        }

        // Do the work on the worker thread that called set_value from the
        // predecessor sender. This thread participates in the work and does not
        // need a new task since it already runs on a task.
        template <typename Task>
        void do_work_local(Task&& task_f) const
        {
            std::cout << "Executing local work on thread "
                      << std::this_thread::get_id() << std::endl;
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            if (char const* scheduler_annotation =
                    hpx::execution::experimental::get_annotation(
                        op_state->scheduler))
            {
                hpx::scoped_annotation ann(scheduler_annotation);
                task_f();
            }
            else
#endif
            {
                hpx::scoped_annotation ann(op_state->f);
                task_f();
            }
        }

        using range_value_type =
            hpx::traits::iter_value_t<hpx::traits::range_iterator_t<Shape>>;

        // Support try_query for stop token
        template <typename P>
        std::optional<P> try_query() noexcept
        {
            // Enable querying inplace_stop_token
            if constexpr (std::is_same_v<P,
                              p2300_stop_token::in_place_stop_token>)
            {
                return hpx::execution::experimental::get_stop_token(
                    hpx::execution::experimental::get_env(op_state->receiver));
            }
            return std::nullopt;
        }

        // Execute a range of indices [begin, end)
        // for both chunked and unchunked modes
        // f is expected to accept f(uint32_t begin, uint32_t end, Ts&&...)
        // for chunked mode
        // Single-index f(index, Ts&&...) is supported via single_index_adapter
        template <typename... Ts>
        void execute(
            std::uint32_t begin, std::uint32_t end, Ts&&... ts) noexcept
        {
            // Check stop token
            auto stop_token = hpx::execution::experimental::get_stop_token(
                hpx::execution::experimental::get_env(op_state->receiver));
            if (stop_token.stop_requested())
            {
                hpx::execution::experimental::set_stopped(
                    HPX_MOVE(op_state->receiver));
                return;
            }

            std::cout << "Executing range: begin=" << begin << ", end=" << end
                      << " on thread " << std::this_thread::get_id()
                      << std::endl;

            // Dispatch based on f's invocability
            if constexpr (std::is_invocable_v<F, std::uint32_t, std::uint32_t,
                              std::add_lvalue_reference_t<Ts>...>)
            {
                // Range-based f: invoke directly
                auto index_pack = std::index_sequence_for<Ts...>{};
                bulk_scheduler_invoke_helper(index_pack, op_state->f,
                    std::pair{begin, end}, true, std::forward<Ts>(ts)...);
            }
            else
            {
                // Single-index f: use adapter
                single_index_adapter<F>{op_state->f}(
                    begin, end, std::forward<Ts>(ts)...);
            }

            // Signal completion if this is the last task
            if (--(op_state->tasks_remaining.data_) == 0)
            {
                hpx::execution::experimental::set_value(
                    HPX_MOVE(op_state->receiver), HPX_FORWARD(Ts, ts)...);
            }
        }

        // clang-format off
        template <typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_invocable_v<F, std::uint32_t, std::uint32_t,
                    std::add_lvalue_reference_t<Ts>...> ||
                hpx::is_invocable_v<F, range_value_type,
                    std::add_lvalue_reference_t<Ts>...>
            )>
        // clang-format on
        friend void tag_invoke(hpx::execution::experimental::set_value_t,
            bulk_receiver&& r, Ts&&... ts) noexcept
        {
            hpx::detail::try_catch_exception_ptr(
                [&]() {
                    // Delegate to execute with the full range
                    auto const size = static_cast<std::uint32_t>(
                        hpx::util::size(r.op_state->shape));
                    r.execute(0, size, HPX_FORWARD(Ts, ts)...);
                },
                [&](std::exception_ptr ep) {
                    hpx::execution::experimental::set_error(
                        HPX_MOVE(r.op_state->receiver), HPX_MOVE(ep));
                });
        }
    };

    template <typename OperationState, typename F, typename Shape>
    struct set_value_loop_visitor
    {
        OperationState* const op_state;
        task_function<OperationState, F, Shape> const* const task_f;

        [[noreturn]] void operator()(hpx::monostate) const noexcept
        {
            HPX_UNREACHABLE;
        }

    private:
        // Perform the work in one chunk indexed by index. The index
        // represents a range of indices (iterators) in the given shape.
        template <typename Ts>
        void do_work_chunk(Ts& ts, std::uint32_t const index) const
        {
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
            static hpx::util::itt::event notify_event(
                "set_value_loop_visitor_static::do_work_chunk(chunking)");

            hpx::util::itt::mark_event e(notify_event);
#endif
            // Calculate the range for this chunk
            auto const i_begin =
                static_cast<std::size_t>(index) * task_f->chunk_size;
            auto const i_end =
                (std::min) (i_begin + task_f->chunk_size, task_f->size);

            std::cout << "Processing chunk: index=" << index << ", range=["
                      << i_begin << "," << i_end << "]"
                      << " on thread " << std::this_thread::get_id()
                      << std::endl;

            // Call execute with the range
            hpx::invoke_fused(
                hpx::bind_front(
                    &bulk_receiver<OperationState, F, Shape>::execute,
                    bulk_receiver<OperationState, F, Shape>{op_state}),
                hpx::tuple_cat(
                    hpx::make_tuple(i_begin, i_end), HPX_FORWARD(Ts, ts)));
        }

        template <hpx::concurrency::detail::queue_end Which, typename Ts>
        void do_work(Ts& ts) const
        {
            auto worker_thread = task_f->worker_thread;
            auto& local_queue = op_state->queues[worker_thread].data_;

            std::cout << "Starting work for worker_thread=" << worker_thread
                      << " on thread " << std::this_thread::get_id()
                      << std::endl;

            // Handle local queue first
            hpx::optional<std::uint32_t> index;
            while ((index = local_queue.template pop<Which>()))
            {
                do_work_chunk(ts, *index);
            }

            if (task_f->allow_stealing)
            {
                // Then steal from the opposite end of the neighboring queues
                static constexpr auto opposite_end =
                    hpx::concurrency::detail::opposite_end_v<Which>;

                for (std::uint32_t offset = 1;
                    offset != op_state->num_worker_threads; ++offset)
                {
                    std::size_t neighbor_thread =
                        (worker_thread + offset) % op_state->num_worker_threads;
                    auto& neighbor_queue =
                        op_state->queues[neighbor_thread].data_;

                    while (
                        (index = neighbor_queue.template pop<opposite_end>()))
                    {
                        do_work_chunk(ts, *index);
                    }
                }
            }
        }

    public:
        // Visit the values sent from the predecessor sender. This function
        // first tries to handle all chunks in the queue owned by worker_thread.
        // It then tries to steal chunks from neighboring threads.
        //
        // clang-format off
        template <typename Ts,
            HPX_CONCEPT_REQUIRES_(
                !std::is_same_v<std::decay_t<Ts>, hpx::monostate>
            )>
        // clang-format on
        void operator()(Ts& ts) const
        {
            // schedule chunks from the end, if needed
            if (task_f->reverse_placement)
            {
                do_work<hpx::concurrency::detail::queue_end::right>(ts);
            }
            else
            {
                do_work<hpx::concurrency::detail::queue_end::left>(ts);
            }
        }
    };

    template <typename OperationState>
    struct set_value_end_loop_visitor
    {
        OperationState* const op_state;

        [[noreturn]] void operator()(hpx::monostate) const noexcept
        {
            HPX_UNREACHABLE;
        }

        // Visit the values sent from the predecessor sender. This function is
        // called once all worker threads have processed their chunks and the
        // connected receiver should be signaled.
        //
        // clang-format off
        template <typename Ts,
            HPX_CONCEPT_REQUIRES_(
                !std::is_same_v<std::decay_t<Ts>, hpx::monostate>
            )>
        // clang-format on
        void operator()(Ts&& ts) const
        {
            hpx::invoke_fused(
                hpx::bind_front(hpx::execution::experimental::set_value,
                    HPX_MOVE(op_state->receiver)),
                HPX_FORWARD(Ts, ts));
        }
    };

    // This struct encapsulates the work done by one worker thread.
    template <typename OperationState, typename F, typename Shape>
    struct task_function
    {
        OperationState* const op_state;
        std::size_t const size;
        std::uint32_t const chunk_size;
        std::uint32_t const worker_thread;
        bool reverse_placement;
        bool allow_stealing;

        // Visit the values sent by the predecessor sender.
        void do_work() const
        {
            auto visitor = set_value_loop_visitor<OperationState, F, Shape>{
                op_state, this};
            hpx::visit(HPX_MOVE(visitor), op_state->ts);
        }

        // Store an exception and mark that an exception was thrown in the
        // operation state. This function assumes that there is a current
        // exception.
        template <typename Exception>
        void store_exception(Exception e) const
        {
            // NOLINTNEXTLINE(bugprone-throw-keyword-missing)
            op_state->exceptions.add(HPX_MOVE(e));
        }

        // Finish the work for one worker thread. If this is not the last worker
        // thread to finish, it will only decrement the counter. If it is the
        // last thread it will call set_error if there is an exception.
        // Otherwise it will call set_value on the connected receiver.
        void finish() const
        {
            if (--(op_state->tasks_remaining.data_) == 0)
            {
                if (op_state->bad_alloc_thrown.load(std::memory_order_relaxed))
                {
                    try
                    {
                        throw std::bad_alloc();
                    }
                    catch (...)
                    {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(op_state->receiver),
                            std::current_exception());
                    }
                }
                else if (op_state->exceptions.size() != 0)
                {
                    hpx::execution::experimental::set_error(
                        HPX_MOVE(op_state->receiver),
                        hpx::detail::construct_lightweight_exception(
                            HPX_MOVE(op_state->exceptions)));
                }
                else
                {
                    auto visitor =
                        set_value_end_loop_visitor<OperationState>{op_state};
                    hpx::visit(HPX_MOVE(visitor), HPX_MOVE(op_state->ts));
                }
            }
        }

        // Entry point for the worker thread. It will attempt to do its local
        // work, catch any exceptions, and then call set_value or set_error on
        // the connected receiver.
        void operator()() const
        {
            std::cout << "Task function started for worker_thread="
                      << worker_thread << " on thread "
                      << std::this_thread::get_id() << std::endl;
            try
            {
                do_work();
            }
            catch (std::bad_alloc const&)
            {
                op_state->bad_alloc_thrown = true;
            }
            catch (...)
            {
                store_exception(std::current_exception());
            }

            finish();
        }
    };

    // This sender represents bulk work that will be performed using the
    // thread_pool_scheduler.
    //
    // The work is chunked into a number of chunks larger than the number of
    // worker threads available on the underlying thread pool. The chunks are
    // then assigned to worker thread-specific thread-safe index queues. One HPX
    // thread is spawned for each underlying worker (OS) thread. The HPX thread
    // is responsible for work in one queue. If the queue is empty, no HPX
    // thread will be spawned. Once the HPX thread has finished working on its
    // own queue, it will attempt to steal work from other queues.
    //
    // Since predecessor sender must complete on an HPX thread (the completion
    // scheduler is a thread_pool_scheduler; otherwise the customization defined
    // in this file is not chosen) it will be reused as one of the worker
    // threads.
    //
    template <typename Policy, typename Sender, typename Shape, typename F>
    class thread_pool_bulk_sender
    {
    private:
        thread_pool_policy_scheduler<Policy> scheduler;
        HPX_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
        HPX_NO_UNIQUE_ADDRESS std::decay_t<Shape> shape;
        HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;
        hpx::threads::mask_type pu_mask;
        bool is_unchunked;

    public:
        template <typename Sender_, typename Shape_, typename F_>
        thread_pool_bulk_sender(
            thread_pool_policy_scheduler<Policy>&& scheduler, Sender_&& sender,
            Shape_&& shape, F_&& f, hpx::threads::mask_rvref_type pu_mask)
          : scheduler(HPX_MOVE(scheduler))
          , sender(HPX_FORWARD(Sender_, sender))
          , shape(HPX_FORWARD(Shape_, shape))
          , f(HPX_FORWARD(F_, f))
          , pu_mask(HPX_MOVE(pu_mask))
        {
            std::cout << "Bulk sender constructed with pu_mask" << std::endl;
        }

        template <typename Sender_, typename Shape_, typename F_>
        thread_pool_bulk_sender(thread_pool_policy_scheduler<Policy>&& sched,
            Sender_&& sender, Shape_&& shape, F_&& f, bool is_unchunked = false)
          : scheduler(HPX_MOVE(sched))
          , sender(HPX_FORWARD(Sender_, sender))
          , shape(HPX_FORWARD(Shape_, shape))
          , f(HPX_FORWARD(F_, f))
          , is_unchunked(is_unchunked)
          , pu_mask(detail::full_mask(
                hpx::execution::experimental::get_first_core(scheduler),
                std::max<std::size_t>(
                    hpx::execution::experimental::processing_units_count(
                        hpx::execution::experimental::null_parameters,
                        scheduler, hpx::chrono::null_duration, 0),
                    hpx::threads::hardware_concurrency())))
        {
            std::cout << "Bulk sender constructed: is_unchunked="
                      << is_unchunked << std::endl;
        }

        thread_pool_bulk_sender(thread_pool_bulk_sender&&) = default;
        thread_pool_bulk_sender(thread_pool_bulk_sender const&) = default;
        thread_pool_bulk_sender& operator=(thread_pool_bulk_sender&&) = default;
        thread_pool_bulk_sender& operator=(
            thread_pool_bulk_sender const&) = default;

#if defined(HPX_HAVE_STDEXEC)
        using sender_concept = hpx::execution::experimental::sender_t;

        // added: Explicit completion signatures per P2079R8
        using completion_signatures =
            hpx::execution::experimental::completion_signatures<
                hpx::execution::experimental::set_value_t(),
                hpx::execution::experimental::set_error_t(std::exception_ptr),
                hpx::execution::experimental::set_stopped_t()>;

        template <typename Env>
        friend auto tag_invoke(
            hpx::execution::experimental::get_completion_signatures_t,
            thread_pool_bulk_sender const&, Env const&)
            -> hpx::execution::experimental::transform_completion_signatures_of<
                Sender, Env,
                hpx::execution::experimental::completion_signatures<
                    hpx::execution::experimental::set_error_t(
                        std::exception_ptr),
                    hpx::execution::experimental::set_stopped_t()>>;

        struct env
        {
            std::decay_t<Sender> const& pred_snd;
            thread_pool_policy_scheduler<Policy> const& sch;

            // clang-format off
            template <typename CPO,
                HPX_CONCEPT_REQUIRES_(
                    meta::value<meta::one_of<CPO,
                        hpx::execution::experimental::set_error_t,
                        hpx::execution::experimental::set_stopped_t>> &&
                    hpx::execution::experimental::detail::has_completion_scheduler_v<
                        CPO, std::decay_t<Sender>>
                )>
            // clang-format on
            friend constexpr auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<CPO>
                    tag,
                env const& e) noexcept
            {
                return tag(hpx::execution::experimental::get_env(e.pred_snd));
            }

            friend constexpr auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<
                    hpx::execution::experimental::set_value_t>,
                env const& e) noexcept
            {
                return e.sch;
            }
        };

        // It may be also be correct to forward the entire env of the
        // pred. sender.
        friend constexpr auto tag_invoke(
            hpx::execution::experimental::get_env_t,
            thread_pool_bulk_sender const& s) noexcept
        {
            return env{s.sender, s.scheduler};
        }
#endif

    private:
        template <typename Receiver>
        struct operation_state
        {
            using operation_state_type =
                hpx::execution::experimental::connect_result_t<Sender,
                    bulk_receiver<operation_state, F, Shape>>;

            thread_pool_policy_scheduler<Policy> scheduler;
            bool is_unchunked = false;
            operation_state_type op_state;
            std::size_t first_thread;
            std::size_t num_worker_threads;
            hpx::threads::mask_type pu_mask;
            std::vector<hpx::util::cache_aligned_data<
                hpx::concurrency::detail::non_contiguous_index_queue<>>>
                queues;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Shape> shape;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            hpx::util::cache_aligned_data<std::atomic<std::size_t>>
                tasks_remaining;

            using value_types = value_types_of_t<Sender, empty_env,
                decayed_tuple, hpx::variant>;
            hpx::util::detail::prepend_t<value_types, hpx::monostate> ts;
            std::atomic<bool> bad_alloc_thrown{false};
            hpx::exception_list exceptions;

            template <typename Scheduler_, typename Sender_, typename Shape_,
                typename F_, typename Receiver_>
            operation_state(Scheduler_&& scheduler, Sender_&& sender,
                Shape_&& shape, F_&& f, hpx::threads::mask_type pumask,
                Receiver_&& receiver, bool is_unchunked = false)
              : scheduler(HPX_FORWARD(Scheduler_, scheduler))
              , is_unchunked(is_unchunked)
              , op_state(hpx::execution::experimental::connect(
                    HPX_FORWARD(Sender_, sender),
                    bulk_receiver<operation_state, F, Shape>{this}))
              , first_thread(
                    hpx::execution::experimental::get_first_core(scheduler))
              , num_worker_threads(std::max<std::size_t>(
                    hpx::execution::experimental::processing_units_count(
                        hpx::execution::experimental::null_parameters,
                        scheduler, hpx::chrono::null_duration, 0),
                    hpx::threads::hardware_concurrency()))
              , pu_mask(HPX_MOVE(pumask))
              , queues(num_worker_threads)
              , shape(HPX_FORWARD(Shape_, shape))
              , f(HPX_FORWARD(F_, f))
              , receiver(HPX_FORWARD(Receiver_, receiver))
            {
                tasks_remaining.data_.store(
                    num_worker_threads, std::memory_order_relaxed);
                HPX_ASSERT(hpx::threads::count(pu_mask) >= num_worker_threads);
                std::cout << "Operation state initialized: num_worker_threads="
                          << num_worker_threads
                          << ", first_thread=" << first_thread
                          << ", pu_mask_count=" << hpx::threads::count(pu_mask)
                          << ", is_unchunked=" << is_unchunked << std::endl;

                // Initialize queues for all worker threads
                for (std::uint32_t i = 0; i < num_worker_threads; ++i)
                {
                    bulk_receiver<operation_state, F, Shape>{this}
                        .init_queue_depth_first(i,
                            static_cast<std::uint32_t>(hpx::util::size(shape)),
                            num_worker_threads);
                }
            }

            friend void tag_invoke(start_t, operation_state& os) noexcept
            {
                std::cout << "Starting operation state with "
                          << os.num_worker_threads << " worker threads"
                          << std::endl;
                hpx::execution::experimental::start(os.op_state);
            }
        };

    public:
        template <typename Receiver>
        friend auto tag_invoke(
            connect_t, thread_pool_bulk_sender&& s, Receiver&& receiver)
        {
            std::cout << "Connecting bulk sender (move)" << std::endl;
            return operation_state<std::decay_t<Receiver>>{
                HPX_MOVE(s.scheduler), HPX_MOVE(s.sender), HPX_MOVE(s.shape),
                HPX_MOVE(s.f), HPX_MOVE(s.pu_mask),
                HPX_FORWARD(Receiver, receiver), s.is_unchunked};
        }

        template <typename Receiver>
        friend auto tag_invoke(
            connect_t, thread_pool_bulk_sender& s, Receiver&& receiver)
        {
            std::cout << "Connecting bulk sender (ref)" << std::endl;
            return operation_state<std::decay_t<Receiver>>{s.scheduler,
                s.sender, s.shape, s.f, s.pu_mask,
                HPX_FORWARD(Receiver, receiver), s.is_unchunked};
        }
    };
}    // namespace hpx::execution::experimental::detail

namespace hpx::execution::experimental {
    struct bulk_unchunked_t
    {
    };
    inline constexpr bulk_unchunked_t bulk_unchunked{};

    // clang-format off
    template <typename Policy, typename Sender, typename Shape, typename F,
        HPX_CONCEPT_REQUIRES_(
            !std::is_integral_v<Shape>
        )>
    // clang-format on
    constexpr auto tag_invoke(bulk_t,
        thread_pool_policy_scheduler<Policy> scheduler, Sender&& sender,
        Shape const& shape, F&& f)
    {
        if constexpr (std::is_same_v<Policy, launch::sync_policy>)
        {
            // fall back to sequential execution using stdexec bulk
            return hpx::execution::experimental::bulk(
                HPX_FORWARD(Sender, sender), shape, HPX_FORWARD(F, f));
        }
        else
        {
            return detail::thread_pool_bulk_sender<Policy, Sender, Shape, F>{
                HPX_MOVE(scheduler), HPX_FORWARD(Sender, sender), shape,
                HPX_FORWARD(F, f)};
        }
    }

    // clang-format off
    template <typename Policy, typename Sender, typename Count, typename F,
        HPX_CONCEPT_REQUIRES_(
            std::is_integral_v<Count>
        )>
    // clang-format on
    constexpr decltype(auto) tag_invoke(bulk_t,
        thread_pool_policy_scheduler<Policy> scheduler, Sender&& sender,
        Count const& count, F&& f)
    {
        if constexpr (std::is_same_v<Policy, launch::sync_policy>)
        {
            // fall back to sequential execution using stdexec bulk
            return hpx::execution::experimental::bulk(
                HPX_FORWARD(Sender, sender), hpx::util::counting_shape(count),
                HPX_FORWARD(F, f));
        }
        else
        {
            return detail::thread_pool_bulk_sender<Policy, Sender,
                hpx::util::counting_shape<Count>, F>{HPX_MOVE(scheduler),
                HPX_FORWARD(Sender, sender), hpx::util::counting_shape(count),
                HPX_FORWARD(F, f)};
        }
    }

    // Customization for bulk_unchunked
    template <typename Policy, typename Sender, typename Shape, typename F,
        HPX_CONCEPT_REQUIRES_(!std::is_integral_v<Shape>)>
    constexpr auto tag_invoke(bulk_unchunked_t,
        thread_pool_policy_scheduler<Policy> scheduler, Sender&& sender,
        Shape const& shape, F&& f)
    {
        // Use thread_pool_bulk_sender with is_unchunked = true
        if constexpr (std::is_same_v<Policy, launch::sync_policy>)
        {
            // Fall back to sequential execution using stdexec bulk
            return hpx::execution::experimental::bulk(
                HPX_FORWARD(Sender, sender), shape, HPX_FORWARD(F, f));
        }
        else
        {
            return detail::thread_pool_bulk_sender<Policy, Sender, Shape, F>{
                HPX_MOVE(scheduler), HPX_FORWARD(Sender, sender), shape,
                HPX_FORWARD(F, f), true};
        }
    }

    // Customization for bulk_unchunked with integral count
    template <typename Policy, typename Sender, typename Count, typename F,
        HPX_CONCEPT_REQUIRES_(std::is_integral_v<Count>)>
    constexpr decltype(auto) tag_invoke(bulk_unchunked_t,
        thread_pool_policy_scheduler<Policy> scheduler, Sender&& sender,
        Count const& count, F&& f)
    {
        // Use thread_pool_bulk_sender with is_unchunked = true
        if constexpr (std::is_same_v<Policy, launch::sync_policy>)
        {
            // Fall back to sequential execution using stdexec bulk
            return hpx::execution::experimental::bulk(
                HPX_FORWARD(Sender, sender), hpx::util::counting_shape(count),
                HPX_FORWARD(F, f));
        }
        else
        {
            return detail::thread_pool_bulk_sender<Policy, Sender,
                hpx::util::counting_shape<Count>, F>{HPX_MOVE(scheduler),
                HPX_FORWARD(Sender, sender), hpx::util::counting_shape(count),
                HPX_FORWARD(F, f), true};
        }
    }
}    // namespace hpx::execution::experimental
