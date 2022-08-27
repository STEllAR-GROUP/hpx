//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/concurrency/cache_line_data.hpp>
#include <hpx/concurrency/detail/contiguous_index_queue.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/execution/detail/async_launch_policy_dispatch.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/static_chunk_size.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/modules/hardware.hpp>
#include <hpx/modules/itt_notify.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/threading/thread.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iosfwd>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace execution { namespace experimental {
    /// \brief An executor with fork-join (blocking) semantics.
    ///
    /// The fork_join_executor creates on construction a set of worker threads
    /// that are kept alive for the duration of the executor. Copying the
    /// executor has reference semantics, i.e. copies of a fork_join_executor
    /// hold a reference to the worker threads of the original instance.
    /// Scheduling work through the executor concurrently from different
    /// threads is undefined behaviour.
    ///
    /// The executor keeps a set of worker threads alive for the lifetime of the
    /// executor, meaning other work will not be executed while the executor is
    /// busy or waiting for work. The executor has a customizable delay after
    /// which it will yield to other work.  Since starting and resuming the
    /// worker threads is a slow operation the executor should be reused
    /// whenever possible for multiple adjacent parallel algorithms or
    /// invocations of bulk_(a)sync_execute.
    class fork_join_executor
    {
    public:
        /// Type of loop schedule for use with the fork_join_executor.
        /// loop_schedule::static_ implies no work-stealing;
        /// loop_schedule::dynamic allows stealing when a worker has finished
        /// its local work.
        enum class loop_schedule
        {
            static_,
            dynamic,
        };

        /// \cond nointernal
        using execution_category = hpx::execution::parallel_execution_tag;
        using executor_parameters_type = hpx::execution::static_chunk_size;

    private:
        /// This struct implements the actual functionality of the executor.
        /// This is separated to allow for reference semantics of the executor.
        class shared_data
        {
            // Type definitions.
            enum class thread_state
            {
                starting = 0,
                idle = 1,
                partitioning_work = 2,
                active = 3,
                stopping = 4,
                stopped = 5,
            };

            using queue_type =
                hpx::concurrency::detail::contiguous_index_queue<std::uint32_t>;
            using queues_type =
                std::vector<hpx::util::cache_aligned_data<queue_type>>;

            struct region_data_type;
            using thread_function_helper_type = void(region_data_type&,
                std::size_t, std::size_t, queues_type&,
                hpx::lcos::local::spinlock&, std::exception_ptr&) noexcept;

            // Members that change for each parallel region.
            struct region_data
            {
                // the thread state for each of the executed threads
                std::atomic<thread_state> state_;

                // The helper function that does the actual work for a single
                // parallel region.
                thread_function_helper_type* thread_function_helper_;

                // Pointers to inputs to bulk_sync_execute.
                void* element_function_;
                void const* shape_;
                void* argument_pack_;
            };

            // Can't apply 'using' here as the type needs to be forward
            // declared
            struct region_data_type
              : std::vector<hpx::util::cache_aligned_data<region_data>>
            {
                using base_type =
                    std::vector<hpx::util::cache_aligned_data<region_data>>;
                using base_type::base_type;
            };

            // Members that are used for all parallel regions executed through
            // this executor.
            threads::thread_pool_base* pool_ = nullptr;
            threads::thread_priority priority_ =
                threads::thread_priority::default_;
            threads::thread_stacksize stacksize_ =
                threads::thread_stacksize::small_;
            loop_schedule schedule_ = loop_schedule::static_;
            std::uint64_t yield_delay_;

            std::size_t main_thread_;
            std::size_t num_threads_;
            hpx::lcos::local::spinlock exception_mutex_;
            std::exception_ptr exception_;

            // Data for each parallel region.
            region_data_type region_data_;

            // The current queues for each worker HPX thread.
            queues_type queues_;

            template <typename Op>
            static thread_state wait_state_this_thread_while(
                std::atomic<thread_state> const& tstate, thread_state state,
                std::uint64_t yield_delay, Op&& op)
            {
                auto current = tstate.load(std::memory_order_acquire);
                if (op(current, state))
                {
                    std::uint64_t base_time = util::hardware::timestamp();
                    current = tstate.load(std::memory_order_acquire);
                    while (op(current, state))
                    {
                        for (int i = 0; i < 128; ++i)
                        {
                            HPX_SMT_PAUSE;

                            current = tstate.load(std::memory_order_acquire);
                            if (!op(current, state))
                            {
                                return current;
                            }
                        }

                        if ((util::hardware::timestamp() - base_time) >
                            yield_delay)
                        {
                            hpx::this_thread::yield();
                        }

                        current = tstate.load(std::memory_order_acquire);
                    }
                }
                return current;
            }

            // Entry point for each worker HPX thread. Holds references to the
            // member variables of fork_join_executor.
            struct thread_function
            {
                // Fixed data for the duration of the executor.
                std::size_t const num_threads_;
                std::size_t const thread_index_;
                loop_schedule const schedule_;
                hpx::lcos::local::spinlock& exception_mutex_;
                std::exception_ptr& exception_;
                std::uint64_t yield_delay_;

                // Changing data for each parallel region.
                region_data_type& region_data_;
                queues_type& queues_;

                void set_state_this_thread(thread_state state) noexcept
                {
                    region_data_[thread_index_].data_.state_.store(
                        state, std::memory_order_release);
                }

                thread_state get_state_this_thread() const noexcept
                {
                    return region_data_[thread_index_].data_.state_.load(
                        std::memory_order_relaxed);
                }

                void operator()() noexcept
                {
                    HPX_ASSERT(
                        get_state_this_thread() == thread_state::starting);
                    set_state_this_thread(thread_state::idle);

                    region_data& data = region_data_[thread_index_].data_;

                    // wait as long the state is 'idle'
                    auto state = shared_data::wait_state_this_thread_while(
                        data.state_, thread_state::idle, yield_delay_,
                        std::equal_to<>());

                    while (state != thread_state::stopping)
                    {
                        data.thread_function_helper_(region_data_,
                            thread_index_, num_threads_, queues_,
                            exception_mutex_, exception_);

                        // wait as long the state is 'idle'
                        state = shared_data::wait_state_this_thread_while(
                            data.state_, thread_state::idle, yield_delay_,
                            std::equal_to<>());
                    }

                    HPX_ASSERT(
                        get_state_this_thread() == thread_state::stopping);
                    set_state_this_thread(thread_state::stopped);
                }
            };

            void set_state_main_thread(thread_state state) noexcept
            {
                region_data_[main_thread_].data_.state_.store(
                    state, std::memory_order_relaxed);
            }

            void set_state_all(thread_state state) noexcept
            {
                for (std::size_t t = 0; t < num_threads_; ++t)
                {
                    region_data_[t].data_.state_.store(
                        state, std::memory_order_release);
                    HPX_SMT_PAUSE;
                }
            }

            void wait_state_all(thread_state state) const noexcept
            {
                for (std::size_t t = 0; t < num_threads_; ++t)
                {
                    // wait for thread-state to be equal to 'state'
                    wait_state_this_thread_while(region_data_[t].data_.state_,
                        state, yield_delay_, std::not_equal_to<>());
                }
            }

            void init_threads()
            {
                main_thread_ = get_local_worker_thread_num();
                num_threads_ = pool_->get_os_thread_count();
                if (schedule_ == loop_schedule::dynamic || num_threads_ > 1)
                {
                    queues_.resize(num_threads_);
                }

                hpx::util::thread_description desc("fork_join_executor");
                for (std::size_t t = 0; t < num_threads_; ++t)
                {
                    if (t == main_thread_)
                    {
                        set_state_main_thread(thread_state::idle);
                        continue;
                    }

                    region_data_[t].data_.state_.store(
                        thread_state::starting, std::memory_order_relaxed);

                    auto policy = launch::async_policy(priority_, stacksize_,
                        threads::thread_schedule_hint{
                            static_cast<std::int16_t>(t)});

                    hpx::detail::async_launch_policy_dispatch<
                        launch::async_policy>::call(policy, desc, pool_,
                        thread_function{num_threads_, t, schedule_,
                            exception_mutex_, exception_, yield_delay_,
                            region_data_, queues_});
                }

                wait_state_all(thread_state::idle);
            }

            static constexpr void init_local_work_queue(queue_type& queue,
                std::size_t thread_index, std::size_t num_threads,
                std::size_t size) noexcept
            {
                auto const part_begin = static_cast<std::uint32_t>(
                    (thread_index * size) / num_threads);
                auto const part_end = static_cast<std::uint32_t>(
                    ((thread_index + 1) * size) / num_threads);
                queue.reset(part_begin, part_end);
            }

        public:
            explicit shared_data(threads::thread_priority priority,
                threads::thread_stacksize stacksize, loop_schedule schedule,
                std::chrono::nanoseconds yield_delay)
              : pool_(this_thread::get_pool())
              , priority_(priority)
              , stacksize_(stacksize)
              , schedule_(schedule)
              , yield_delay_(std::uint64_t(
                    yield_delay.count() / pool_->timestamp_scale()))
              , num_threads_(pool_->get_os_thread_count())
              , exception_mutex_()
              , exception_()
              , region_data_(num_threads_)
            {
                HPX_ASSERT(pool_);
                init_threads();
            }

            ~shared_data()
            {
                set_state_all(thread_state::stopping);
                set_state_main_thread(thread_state::stopped);
                wait_state_all(thread_state::stopped);
            }

            /// \cond NOINTERNAL
            bool operator==(shared_data const& rhs) const noexcept
            {
                return pool_ == rhs.pool_ && priority_ == rhs.priority_ &&
                    stacksize_ == rhs.stacksize_ &&
                    schedule_ == rhs.schedule_ &&
                    yield_delay_ == rhs.yield_delay_;
            }

            bool operator!=(shared_data const& rhs) const noexcept
            {
                return !(*this == rhs);
            }

        private:
            /// This struct implements the main work loop for a single parallel
            /// for loop. The indirection through this struct is done to allow
            /// passing the original template parameters F, S, and Tuple
            /// (additional arguments packed into a tuple) given to
            /// bulk_sync_execute without wrapping it into hpx::function or
            /// similar.
            template <typename F, typename S, typename Tuple>
            struct thread_function_helper
            {
                using argument_pack_type = std::decay_t<Tuple>;
                using index_pack_type =
                    typename hpx::util::detail::fused_index_pack<Tuple>::type;

                template <std::size_t... Is_, typename F_, typename A_,
                    typename Tuple_>
                static constexpr void invoke_helper(
                    hpx::util::index_pack<Is_...>, F_&& f, A_&& a, Tuple_&& t)
                {
                    HPX_INVOKE(HPX_FORWARD(F_, f), HPX_FORWARD(A_, a),
                        hpx::get<Is_>(HPX_FORWARD(Tuple_, t))...);
                }

                static void set_state(std::atomic<thread_state>& tstate,
                    thread_state state) noexcept
                {
                    tstate.store(state, std::memory_order_release);
                }

                /// Main entry point for a single parallel region (static
                /// scheduling).
                static void call_static(region_data_type& rdata,
                    std::size_t thread_index, std::size_t num_threads,
                    queues_type&, hpx::lcos::local::spinlock& exception_mutex,
                    std::exception_ptr& exception) noexcept
                {
                    region_data& data = rdata[thread_index].data_;
                    try
                    {
                        // Cast void pointers back to the actual types given to
                        // bulk_sync_execute.
                        auto& element_function =
                            *static_cast<F*>(data.element_function_);
                        auto& shape = *static_cast<S const*>(data.shape_);
                        auto& argument_pack =
                            *static_cast<Tuple*>(data.argument_pack_);

                        // Set up the local queues and state.
                        std::size_t size = hpx::util::size(shape);

                        auto part_begin = static_cast<std::uint32_t>(
                            (thread_index * size) / num_threads);
                        auto const part_end = static_cast<std::uint32_t>(
                            ((thread_index + 1) * size) / num_threads);

                        set_state(data.state_, thread_state::active);

                        // Process local items.
                        for (; part_begin != part_end; ++part_begin)
                        {
                            auto it =
                                std::next(hpx::util::begin(shape), part_begin);
                            invoke_helper(index_pack_type{}, element_function,
                                *it, argument_pack);
                        }
                    }
                    catch (...)
                    {
                        std::lock_guard l(exception_mutex);
                        if (!exception)
                        {
                            exception = std::current_exception();
                        }
                    }

                    set_state(data.state_, thread_state::idle);
                }

                /// Main entry point for a single parallel region (dynamic
                /// scheduling).
                static void call_dynamic(region_data_type& rdata,
                    std::size_t thread_index, std::size_t num_threads,
                    queues_type& queues,
                    hpx::lcos::local::spinlock& exception_mutex,
                    std::exception_ptr& exception) noexcept
                {
                    region_data& data = rdata[thread_index].data_;
                    try
                    {
                        // Cast void pointers back to the actual types given to
                        // bulk_sync_execute.
                        auto& element_function =
                            *static_cast<F*>(data.element_function_);
                        auto& shape = *static_cast<S const*>(data.shape_);
                        auto& argument_pack =
                            *static_cast<Tuple*>(data.argument_pack_);

                        // Set up the local queues and state.
                        queue_type& local_queue = queues[thread_index].data_;
                        std::size_t size = hpx::util::size(shape);
                        init_local_work_queue(
                            local_queue, thread_index, num_threads, size);

                        set_state(data.state_, thread_state::active);

                        // Process local items first.
                        hpx::optional<std::uint32_t> index;
                        while ((index = local_queue.pop_left()))
                        {
                            auto it = std::next(
                                hpx::util::begin(shape), index.value());
                            invoke_helper(index_pack_type{}, element_function,
                                *it, argument_pack);
                        }

                        // As loop schedule is dynamic, steal from neighboring
                        // threads.
                        for (std::size_t offset = 1; offset < num_threads;
                             ++offset)
                        {
                            std::size_t neighbor_index =
                                (thread_index + offset) % num_threads;

                            if (rdata[neighbor_index].data_.state_.load(
                                    std::memory_order_acquire) !=
                                thread_state::active)
                            {
                                continue;
                            }

                            queue_type& neighbor_queue =
                                queues[neighbor_index].data_;

                            while ((index = neighbor_queue.pop_right()))
                            {
                                auto it = std::next(
                                    hpx::util::begin(shape), index.value());
                                invoke_helper(index_pack_type{},
                                    element_function, *it, argument_pack);
                            }
                        }
                    }
                    catch (...)
                    {
                        std::lock_guard l(exception_mutex);
                        if (!exception)
                        {
                            exception = std::current_exception();
                        }
                    }

                    set_state(data.state_, thread_state::idle);
                }
            };

            template <typename F, typename S, typename Args>
            thread_function_helper_type* set_all_states_and_region_data(
                thread_state state, F& f, S const& shape,
                Args& argument_pack) noexcept
            {
                thread_function_helper_type* func = nullptr;
                if (schedule_ == loop_schedule::static_ || num_threads_ == 1)
                {
                    func = &thread_function_helper<F, S, Args>::call_static;
                }
                else
                {
                    func = &thread_function_helper<F, S, Args>::call_dynamic;
                }

                for (std::size_t t = 0; t < num_threads_; ++t)
                {
                    region_data& data = region_data_[t].data_;

                    data.element_function_ = &f;
                    data.shape_ = &shape;
                    data.argument_pack_ = &argument_pack;
                    data.thread_function_helper_ = func;

                    data.state_.store(state, std::memory_order_release);
                }
                return func;
            }

        public:
            template <typename F, typename S, typename... Ts>
            void bulk_sync_execute(F&& f, S const& shape, Ts&&... ts)
            {
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
                static hpx::util::itt::event notify_event(
                    "fork_join_executor::bulk_sync_execute");

                hpx::util::itt::mark_event e(notify_event);
#endif

                // Set the data for this parallel region
                auto argument_pack =
                    hpx::forward_as_tuple(HPX_FORWARD(Ts, ts)...);

                // Signal all worker threads to start partitioning work for
                // themselves, and then starting the actual work.
                thread_function_helper_type* func =
                    set_all_states_and_region_data(
                        thread_state::partitioning_work, f, shape,
                        argument_pack);

                // Start work on the main thread.
                func(region_data_, main_thread_, num_threads_, queues_,
                    exception_mutex_, exception_);

                // Wait for all threads to finish their work assigned to
                // them in this parallel region.
                wait_state_all(thread_state::idle);

                std::lock_guard l(exception_mutex_);
                if (exception_)
                {
                    std::rethrow_exception(HPX_MOVE(exception_));
                }
            }

            template <typename F, typename S, typename... Ts>
            hpx::future<void> bulk_async_execute(
                F&& f, S const& shape, Ts&&... ts)
            {
                // Forward to the synchronous version as we can't create
                // futures to the completion of the parallel region (this HPX
                // thread participates in computation).
                try
                {
                    bulk_sync_execute(
                        HPX_FORWARD(F, f), shape, HPX_FORWARD(Ts, ts)...);
                }
                catch (...)
                {
                    using result_type = hpx::parallel::execution::detail::
                        bulk_function_result_t<F, S, Ts...>;
                    return hpx::make_exceptional_future<result_type>(
                        std::current_exception());
                }
                return hpx::make_ready_future();
            }
        };

    private:
        std::shared_ptr<shared_data> shared_data_ = nullptr;

    public:
        template <typename F, typename S, typename... Ts>
        void bulk_sync_execute(F&& f, S const& shape, Ts&&... ts)
        {
            shared_data_->bulk_sync_execute(
                HPX_FORWARD(F, f), shape, HPX_FORWARD(Ts, ts)...);
        }

        template <typename F, typename S, typename... Ts>
        decltype(auto) bulk_async_execute(F&& f, S const& shape, Ts&&... ts)
        {
            return shared_data_->bulk_async_execute(
                HPX_FORWARD(F, f), shape, HPX_FORWARD(Ts, ts)...);
        }

        bool operator==(fork_join_executor const& rhs) const noexcept
        {
            return *shared_data_ == *rhs.shared_data_;
        }

        bool operator!=(fork_join_executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        fork_join_executor const& context() const noexcept
        {
            return *this;
        }
        /// \endcond

        /// \brief Construct a fork_join_executor.
        ///
        /// \param priority The priority of the worker threads.
        /// \param stacksize The stacksize of the worker threads. Must not be
        ///                  nostack.
        /// \param schedule The loop schedule of the parallel regions.
        /// \param yield_delay The time after which the executor yields to
        ///        other work if it hasn't received any new work for bulk
        ///        execution.
        explicit fork_join_executor(
            threads::thread_priority priority = threads::thread_priority::high,
            threads::thread_stacksize stacksize =
                threads::thread_stacksize::small_,
            loop_schedule schedule = loop_schedule::static_,
            std::chrono::nanoseconds yield_delay = std::chrono::milliseconds(1))
        {
            if (stacksize == threads::thread_stacksize::nostack)
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "fork_join_executor::fork_join_executor",
                    "The fork_join_executor does not support using "
                    "thread_stacksize::nostack as the stacksize (stackful "
                    "threads are required to yield correctly when idle)");
            }

            shared_data_ = std::make_shared<shared_data>(
                priority, stacksize, schedule, yield_delay);
        }
    };

    HPX_CORE_EXPORT std::ostream& operator<<(
        std::ostream& os, fork_join_executor::loop_schedule const& schedule);
}}}    // namespace hpx::execution::experimental

namespace hpx { namespace parallel { namespace execution {
    /// \cond NOINTERNAL
    template <>
    struct is_bulk_one_way_executor<
        hpx::execution::experimental::fork_join_executor> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<
        hpx::execution::experimental::fork_join_executor> : std::true_type
    {
    };
    /// \endcond
}}}    // namespace hpx::parallel::execution
