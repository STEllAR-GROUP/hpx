//  Copyright (c) 2019-2020 ETH Zurich
//  Copyright (c) 2007-2023 Hartmut Kaiser
//  Copyright (c) 2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_base/scheduling_properties.hpp>
#include <hpx/concurrency/cache_line_data.hpp>
#include <hpx/concurrency/detail/non_contiguous_index_queue.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/errors/exception.hpp>
#include <hpx/errors/exception_list.hpp>
#include <hpx/execution/detail/async_launch_policy_dispatch.hpp>
#include <hpx/execution/detail/post_policy_dispatch.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/executors/detail/hierarchical_spawning.hpp>
#include <hpx/functional/detail/invoke.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/resource_partitioner/detail/partitioner.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>
#include <hpx/topology/cpu_mask.hpp>
#include <hpx/type_support/pack.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::parallel::execution::detail {

    ///////////////////////////////////////////////////////////////////////////
    // This struct encapsulates the work done by one worker thread.
    template <typename SharedState>
    struct task_function
    {
        hpx::intrusive_ptr<SharedState> const state;
        std::size_t const size;
        std::uint32_t const chunk_size;
        std::uint32_t const worker_thread;
        bool const reverse_placement;
        bool const allow_stealing;

        template <std::size_t... Is, typename F, typename T, typename Ts>
        static constexpr void bulk_invoke_helper(
            hpx::util::index_pack<Is...>, F&& f, T&& t, Ts&& ts)
        {
            HPX_INVOKE(HPX_FORWARD(F, f), HPX_FORWARD(T, t),
                hpx::get<Is>(HPX_FORWARD(Ts, ts))...);
        }

        // Perform the work in one element indexed by index. The index
        // represents a range of indices (iterators) in the given shape.
        template <typename F, typename Ts>
        void do_work_chunk(F&& f, Ts&& ts, std::uint32_t const index) const
        {
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
            static hpx::util::itt::event notify_event(
                "set_value_loop_visitor_static::do_work_chunk(chunking)");

            hpx::util::itt::mark_event e(notify_event);
#endif
            using index_pack_type =
                hpx::detail::fused_index_pack_t<std::decay_t<Ts>>;

            auto const i_begin = index * chunk_size;
            auto const i_end = (std::min)(
                i_begin + chunk_size, static_cast<std::uint32_t>(size));

            auto it = std::next(hpx::util::begin(state->shape), i_begin);
            for (std::uint32_t i = i_begin; i != i_end; (void) ++it, ++i)
            {
                bulk_invoke_helper(index_pack_type{}, f, *it, ts);
            }
        }

        template <hpx::concurrency::detail::queue_end Which>
        void do_work() const
        {
            // explicitly copy function (object) and arguments
            auto f = state->f;
            auto ts = state->ts;

            auto& local_queue = state->queues[worker_thread].data_;

            // Handle local queue first
            hpx::optional<std::uint32_t> index;
            while ((index = local_queue.template pop<Which>()))
            {
                do_work_chunk(f, ts, *index);
            }

            if (allow_stealing)
            {
                // Then steal from the opposite end of the neighboring queues
                static constexpr auto opposite_end =
                    hpx::concurrency::detail::opposite_end_v<Which>;

                for (std::uint32_t offset = 1; offset != state->num_threads;
                     ++offset)
                {
                    std::size_t neighbor_thread =
                        (worker_thread + offset) % state->num_threads;
                    auto& neighbor_queue = state->queues[neighbor_thread].data_;

                    while (
                        (index = neighbor_queue.template pop<opposite_end>()))
                    {
                        do_work_chunk(f, ts, *index);
                    }
                }
            }
        }

        // Execute task function
        void do_work() const
        {
            // schedule chunks from the end, if needed
            if (reverse_placement)
            {
                do_work<hpx::concurrency::detail::queue_end::right>();
            }
            else
            {
                do_work<hpx::concurrency::detail::queue_end::left>();
            }
        }

        // Store an exception and mark that an exception was thrown in the
        // operation state. This function assumes that there is a current
        // exception.
        template <typename Exception>
        void store_exception(Exception e) const
        {
            // NOLINTNEXTLINE(bugprone-throw-keyword-missing)
            state->exceptions.add(HPX_MOVE(e));
        }

        // Finish the work for one worker thread. If this is not the last worker
        // thread to finish, it will only decrement the counter. If it is the
        // last thread it will call set_exception if there is an exception.
        // Otherwise it will call set_value on the shared state.
        void finish() const
        {
            if (--(state->tasks_remaining.data_) == 0)
            {
                if (state->bad_alloc_thrown.load(std::memory_order_relaxed))
                {
                    try
                    {
                        throw std::bad_alloc();
                    }
                    catch (...)
                    {
                        state->set_exception(std::current_exception());
                    }
                }
                else if (state->exceptions.size() != 0)
                {
                    state->set_exception(
                        hpx::detail::construct_lightweight_exception(
                            HPX_MOVE(state->exceptions)));
                }
                else
                {
                    state->set_data(hpx::util::unused);
                }
            }
        }

        // Entry point for the worker thread. It will attempt to do its local
        // work, catch any exceptions, and then call set_value or set_exception
        // on the shared state.
        void operator()() const
        {
            try
            {
                do_work();
            }
            catch (std::bad_alloc const&)
            {
                state->bad_alloc_thrown = true;
            }
            catch (...)
            {
                store_exception(std::current_exception());
            }

            finish();
        }
    };

    ////////////////////////////////////////////////////////////////////////////
    // Extend the shared state of the returned future allowing to keep alive all
    // data needed for the scheduling
    template <typename Launch, typename F, typename Shape, typename... Ts>
    struct index_queue_bulk_state : lcos::detail::future_data<void>
    {
    private:
        using base_type = lcos::detail::future_data<void>;
        using init_no_addref = base_type::init_no_addref;

        // Compute a chunk size given a number of worker threads and a total
        // number of items n. Returns a power-of-2 chunk size that produces at
        // most 8 and at least 4 chunks per worker thread.
        constexpr std::uint32_t get_chunk_size(
            std::uint32_t const n) const noexcept
        {
            std::uint32_t chunk_size = 1;
            while (chunk_size * num_threads * 8 < n)
            {
                chunk_size *= 2;
            }
            return chunk_size;
        }

        static hpx::threads::mask_type full_mask(
            std::size_t first_thread, std::size_t num_threads)
        {
            std::size_t const overall_threads =    //-V101
                hpx::threads::hardware_concurrency();
            auto mask = hpx::threads::mask_type();
            hpx::threads::resize(mask, overall_threads);

            auto const& rp = hpx::resource::get_partitioner();
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
            return mask;
        }

        static hpx::threads::mask_type limit_mask(
            hpx::threads::mask_cref_type orgmask, std::size_t num_threads)
        {
            std::size_t const num_cores =    //-V101
                hpx::threads::hardware_concurrency();

            auto mask = hpx::threads::mask_type();
            hpx::threads::resize(mask, num_cores);
            for (std::size_t i = 0, j = 0; i != num_threads && j != num_cores;
                 ++j)
            {
                if (hpx::threads::test(orgmask, j))
                {
                    hpx::threads::set(mask, j);
                    ++i;
                }
            }
            return mask;
        }

        // Initialize a queue for a worker thread.
        void init_queue_depth_first(std::uint32_t const worker_thread,
            std::uint32_t const size) noexcept
        {
            auto& queue = queues[worker_thread].data_;
            auto const part_begin = static_cast<std::uint32_t>(
                (worker_thread * size) / num_threads);
            auto const part_end = static_cast<std::uint32_t>(
                ((worker_thread + 1) * size) / num_threads);
            queue.reset(part_begin, part_end);
        }

        void init_queue_breadth_first(std::uint32_t const worker_thread,
            std::uint32_t const size) noexcept
        {
            auto& queue = queues[worker_thread].data_;
            auto const num_steps = size / num_threads + 1;
            auto const part_begin = worker_thread;
            auto part_end = static_cast<std::uint32_t>((std::min)(
                size + num_threads - 1, part_begin + num_steps * num_threads));
            auto const remainder = static_cast<std::uint32_t>(
                (part_end - part_begin) % num_threads);
            if (remainder != 0)
            {
                part_end -= remainder;
            }
            queue.reset(
                part_begin, part_end, static_cast<std::uint32_t>(num_threads));
        }

        // Spawn a task which will process a number of chunks. If the queue
        // contains no chunks no task will be spawned.
        template <typename Task>
        void do_work_task(hpx::threads::thread_description const& desc,
            threads::thread_pool_base* pool, bool dont_bind_to_core,
            Task&& task_f) const
        {
            std::uint32_t const worker_thread = task_f.worker_thread;
            if (queues[worker_thread].data_.empty())
            {
                // If the queue is empty we don't spawn a task. We only signal
                // that this "task" is ready.
                task_f.finish();
                return;
            }

            // run task on small stack
            auto post_policy = hpx::execution::experimental::with_stacksize(
                policy, threads::thread_stacksize::small_);

            if (dont_bind_to_core)
            {
                // Make sure the new task is not bound to a particular core, if
                // requested. This prevents the main thread from potentially
                // being occupied in asynchronous scenarios.
                hpx::threads::thread_priority const priority =
                    hpx::execution::experimental::get_priority(post_policy);
                if (priority == hpx::threads::thread_priority::bound)
                {
                    post_policy = hpx::execution::experimental::with_priority(
                        post_policy, hpx::threads::thread_priority::normal);
                }
            }

            // launch task on new HPX-thread
            auto hint = hpx::execution::experimental::get_hint(policy);
            if (hint.mode == hpx::threads::thread_schedule_hint_mode::none &&
                hint.hint == -1)
            {
                // apply hint if none was given
                hint.mode = hpx::threads::thread_schedule_hint_mode::thread;
                hint.hint = worker_thread + first_thread;

                hpx::detail::post_policy_dispatch<Launch>::call(
                    hpx::execution::experimental::with_hint(post_policy, hint),
                    desc, pool, HPX_FORWARD(Task, task_f));
            }
            else
            {
                hpx::detail::post_policy_dispatch<Launch>::call(
                    post_policy, desc, pool, HPX_FORWARD(Task, task_f));
            }
        }

    public:
        template <typename F_, typename... Ts_>
        index_queue_bulk_state(std::size_t first_thread,
            std::size_t num_threads, Launch l, F_&& f, Shape const& shape,
            Ts_&&... ts) noexcept
          : base_type(init_no_addref{})
          , first_thread(static_cast<std::uint32_t>(first_thread))
          , num_threads(num_threads)
          , policy(HPX_MOVE(l))
          , f(HPX_FORWARD(F_, f))
          , shape(shape)
          , ts(HPX_FORWARD(Ts_, ts)...)
          , pu_mask(full_mask(first_thread, num_threads))
          , queues(num_threads)
        {
            tasks_remaining.data_.store(num_threads, std::memory_order_relaxed);
            HPX_ASSERT(hpx::threads::count(pu_mask) == num_threads);
        }

        void execute(hpx::threads::thread_description const& desc,
            threads::thread_pool_base* pool)
        {
            auto const size =
                static_cast<std::uint32_t>(hpx::util::size(shape));

            // Calculate chunk size and number of chunks
            std::uint32_t chunk_size = get_chunk_size(size);
            std::uint32_t const num_chunks =
                (size + chunk_size - 1) / chunk_size;

            // launch only as many tasks as we have chunks
            std::size_t const num_pus = num_threads;
            if (num_chunks < static_cast<std::uint32_t>(num_threads))
            {
                num_threads = num_chunks;    //-V101
                tasks_remaining.data_ = num_chunks;
                pu_mask = limit_mask(pu_mask, num_chunks);    //-V106
            }

            HPX_ASSERT(hpx::threads::count(pu_mask) == num_threads);

            // thread placement
            hpx::threads::thread_schedule_hint const hint =
                hpx::execution::experimental::get_hint(policy);

            using placement = hpx::threads::thread_placement_hint;

            // Initialize the queues for all worker threads so that worker
            // threads can start stealing immediately when they start.
            for (std::uint32_t worker_thread = 0; worker_thread != num_threads;
                 ++worker_thread)
            {
                if (hint.placement_mode() == placement::breadth_first ||
                    hint.placement_mode() == placement::breadth_first_reverse)
                {
                    init_queue_breadth_first(worker_thread, num_chunks);
                }
                else
                {
                    // the default for this executor is depth-first placement
                    init_queue_depth_first(worker_thread, num_chunks);
                }
            }

            // Spawn the worker threads for all except the local queue.
            auto local_worker_thread =
                static_cast<std::uint32_t>(hpx::get_local_worker_thread_num());
            std::uint32_t worker_thread = 0;
            bool main_thread_ok = false;

            auto const& rp = hpx::resource::get_partitioner();
            std::size_t main_pu_num =
                rp.get_pu_num(local_worker_thread);    //-V106
            if (!hpx::threads::test(pu_mask, main_pu_num) || num_threads == 1)
            {
                main_thread_ok = true;
                local_worker_thread = worker_thread++;
                main_pu_num = rp.get_pu_num(
                    local_worker_thread + first_thread);    //-V106
            }

            bool reverse_placement =
                hint.placement_mode() == placement::depth_first_reverse ||
                hint.placement_mode() == placement::breadth_first_reverse;
            bool allow_stealing =
                !hpx::threads::do_not_share_function(hint.sharing_mode());

            for (std::uint32_t pu = 0;
                 worker_thread != num_threads && pu != num_pus; ++pu)
            {
                std::size_t const pu_num =
                    rp.get_pu_num(pu + first_thread);    //-V106

                // The queue for the local thread is handled later inline.
                if (!main_thread_ok && pu == local_worker_thread)
                {
                    // the initializing thread is expected to participate in
                    // evaluating parallel regions
                    HPX_ASSERT(hpx::threads::test(pu_mask, pu_num));
                    main_thread_ok = true;
                    local_worker_thread = worker_thread++;
                    main_pu_num = rp.get_pu_num(
                        local_worker_thread + first_thread);    //-V106
                    continue;
                }

                // don't double-book core that runs main thread
                if (main_thread_ok && main_pu_num == pu_num)
                {
                    continue;
                }

                // create an HPX thread only for cores in the given PU-mask
                if (!hpx::threads::test(pu_mask, pu_num))
                {
                    continue;
                }

                // Schedule task for this worker thread
                do_work_task(desc, pool, false,
                    task_function<index_queue_bulk_state>{
                        hpx::intrusive_ptr<index_queue_bulk_state>(this), size,
                        chunk_size, worker_thread, reverse_placement,
                        allow_stealing});

                ++worker_thread;
            }

            // there have to be as many HPX threads as there are set bits in
            // the PU-mask
            HPX_ASSERT(worker_thread == num_threads);

            // the main thread should have been associated with a queue
            if (main_thread_ok)
            {
                // Handle the queue for the local thread.
                do_work_task(desc, pool, true,
                    task_function<index_queue_bulk_state>{
                        hpx::intrusive_ptr<index_queue_bulk_state>(this), size,
                        chunk_size, local_worker_thread, reverse_placement,
                        allow_stealing});
            }
        }

        std::uint32_t first_thread;
        std::size_t num_threads;
        Launch policy;
        std::decay_t<F> f;
        Shape shape;
        hpx::tuple<std::decay_t<Ts>...> ts;

        hpx::threads::mask_type pu_mask;
        std::vector<hpx::util::cache_aligned_data<
            hpx::concurrency::detail::non_contiguous_index_queue<>>>
            queues;
        hpx::util::cache_aligned_data<std::atomic<std::size_t>> tasks_remaining;

        std::atomic<bool> bad_alloc_thrown{false};
        hpx::exception_list exceptions;
    };

    ///////////////////////////////////////////////////////////////////////////
    // This specialization avoids creating a future for each of the scheduled
    // tasks. It also avoids an additional allocation by directly returning a
    // hpx::future.
    template <typename Launch, typename F, typename S, typename... Ts>
    decltype(auto) index_queue_bulk_async_execute_void(
        hpx::threads::thread_description const& desc,
        threads::thread_pool_base* pool, std::size_t first_thread,
        std::size_t num_threads, Launch policy, F&& f, S const& shape,
        Ts&&... ts)
    {
        HPX_ASSERT(pool);

        // Don't spawn tasks if there is no work to be done
        std::size_t const size = hpx::util::size(shape);
        if (size == 0)
        {
            return hpx::make_ready_future();
        }

        using shared_state = index_queue_bulk_state<Launch, F, S, Ts...>;
        hpx::intrusive_ptr<shared_state> p(
            new shared_state(first_thread, num_threads, HPX_MOVE(policy),
                HPX_FORWARD(F, f), shape, HPX_FORWARD(Ts, ts)...),
            false);

        p->execute(desc, pool);

        return hpx::traits::future_access<hpx::future<void>>::create(
            HPX_MOVE(p));
    }

    template <typename Launch, typename F, typename S, typename... Ts>
    decltype(auto) index_queue_bulk_async_execute(
        hpx::threads::thread_description const& desc,
        threads::thread_pool_base* pool, std::size_t first_thread,
        std::size_t num_threads, std::size_t hierarchical_threshold,
        Launch policy, F&& f, S const& shape, Ts&&... ts)
    {
        using result_type = detail::bulk_function_result_t<F, S, Ts...>;
        if constexpr (!std::is_void_v<result_type>)
        {
            return hierarchical_bulk_async_execute_helper(desc, pool,
                first_thread, num_threads, hierarchical_threshold, policy,
                HPX_FORWARD(F, f), shape, HPX_FORWARD(Ts, ts)...);
        }
        else
        {
            return index_queue_bulk_async_execute_void(desc, pool, first_thread,
                num_threads, policy, HPX_FORWARD(F, f), shape,
                HPX_FORWARD(Ts, ts)...);
        }
    }

    template <typename Launch, typename F, typename S, typename... Ts>
    decltype(auto) index_queue_bulk_async_execute(
        threads::thread_pool_base* pool, std::size_t first_thread,
        std::size_t num_threads, std::size_t hierarchical_threshold,
        Launch policy, F&& f, S const& shape, Ts&&... ts)
    {
        hpx::threads::thread_description const desc(
            f, "hierarchical_bulk_async_execute");

        return index_queue_bulk_async_execute(desc, pool, first_thread,
            num_threads, hierarchical_threshold, policy, HPX_FORWARD(F, f),
            shape, HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx::parallel::execution::detail
