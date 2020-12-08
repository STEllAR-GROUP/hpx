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
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/threading/thread.hpp>
#include <hpx/timing/high_resolution_timer.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
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
            using thread_states_type = std::vector<
                hpx::util::cache_aligned_data<std::atomic<thread_state>>>;
            using thread_function_helper_type = void(void*, void const*, void*,
                std::size_t, std::size_t, loop_schedule, queues_type&,
                thread_states_type&, hpx::lcos::local::spinlock&,
                std::exception_ptr&);

            // Members that are used for all parallel regions executed through
            // this executor.
            threads::thread_pool_base* pool_ = nullptr;
            threads::thread_priority priority_ =
                threads::thread_priority::default_;
            threads::thread_stacksize stacksize_ =
                threads::thread_stacksize::small_;
            loop_schedule schedule_ = loop_schedule::static_;
            std::size_t main_thread_;
            std::size_t num_threads_;
            thread_states_type thread_states_;
            hpx::lcos::local::spinlock exception_mutex_;
            std::exception_ptr exception_;
            std::chrono::nanoseconds yield_delay_;

            // Members that change for each parallel region.

            // The helper function that does the actual work for a single
            // parallel region.
            std::atomic<thread_function_helper_type*> thread_function_helper_{
                nullptr};

            // Pointers to inputs to bulk_sync_execute.
            std::atomic<void*> element_function_{nullptr};
            std::atomic<void const*> shape_{nullptr};
            std::atomic<std::size_t> size_{0};
            std::atomic<void*> argument_pack_{nullptr};

            // The current queues for each worker HPX thread.
            queues_type queues_;

            // Entry point for each worker HPX thread. Holds references to the
            // member variables of fork_join_executor.
            struct thread_function
            {
                // Fixed data for the duration of the executor.
                std::size_t const num_threads_;
                std::size_t const thread_index_;
                loop_schedule const schedule_;
                std::atomic<thread_state>& thread_state_;
                hpx::lcos::local::spinlock& exception_mutex_;
                std::exception_ptr& exception_;
                std::chrono::nanoseconds& yield_delay_;

                // Changing data for each parallel region.
                std::atomic<thread_function_helper_type*>&
                    thread_function_helper_;
                std::atomic<void*>& element_function_;
                std::atomic<void const*>& shape_;
                std::atomic<std::size_t>& size_;
                std::atomic<void*>& argument_pack_;
                queues_type& queues_;
                thread_states_type& thread_states_;

                void wait_not_state_this_thread(thread_state state)
                {
                    hpx::chrono::high_resolution_timer t;
                    while (
                        thread_state_.load(std::memory_order_acquire) == state)
                    {
                        if (t.elapsed_nanoseconds() > yield_delay_.count())
                        {
                            hpx::this_thread::yield();
                        }
                    }
                }

                void set_state_this_thread(thread_state state)
                {
                    thread_state_.store(state, std::memory_order_release);
                }

                void operator()()
                {
                    HPX_ASSERT(thread_state_ == thread_state::starting);
                    set_state_this_thread(thread_state::idle);

                    do
                    {
                        wait_not_state_this_thread(thread_state::idle);
                        if (thread_state_ == thread_state::stopping)
                        {
                            break;
                        }

                        (thread_function_helper_.load(
                            std::memory_order_relaxed))(
                            element_function_.load(std::memory_order_relaxed),
                            shape_.load(std::memory_order_relaxed),
                            argument_pack_.load(std::memory_order_relaxed),
                            thread_index_, num_threads_, schedule_, queues_,
                            thread_states_, exception_mutex_, exception_);
                    } while (true);

                    HPX_ASSERT(thread_state_ == thread_state::stopping);
                    set_state_this_thread(thread_state::stopped);
                }
            };

            void set_state_main_thread(thread_state state)
            {
                thread_states_[main_thread_].data_.store(
                    state, std::memory_order_relaxed);
            }

            void set_state_all(thread_state state)
            {
                for (std::size_t t = 0; t < num_threads_; ++t)
                {
                    thread_states_[t].data_.store(
                        state, std::memory_order_release);
                }
            }

            void wait_state_all(thread_state state)
            {
                for (std::size_t t = 0; t < num_threads_; ++t)
                {
                    while (thread_states_[t].data_.load(
                               std::memory_order_acquire) != state)
                    {
                    }
                }
            }

            void init_threads()
            {
                main_thread_ = get_local_worker_thread_num();
                num_threads_ = pool_->get_os_thread_count();
                queues_.resize(num_threads_);

                for (std::size_t t = 0; t < num_threads_; ++t)
                {
                    if (t == main_thread_)
                    {
                        thread_states_[t].data_ = thread_state::idle;
                        continue;
                    }

                    thread_states_[t].data_ = thread_state::starting;
                    threads::thread_schedule_hint hint{
                        static_cast<std::int16_t>(t)};
                    hpx::detail::async_launch_policy_dispatch<
                        launch::async_policy>::call(launch::async, pool_,
                        priority_, stacksize_, hint,
                        thread_function{num_threads_, t, schedule_,
                            thread_states_[t].data_, exception_mutex_,
                            exception_, yield_delay_, thread_function_helper_,
                            element_function_, shape_, size_, argument_pack_,
                            queues_, thread_states_});
                }

                wait_state_all(thread_state::idle);
            }

            static void init_local_work_queue(queue_type& queue,
                std::size_t thread_index, std::size_t num_threads,
                std::size_t size)
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
              , num_threads_(pool_->get_os_thread_count())
              , thread_states_(num_threads_)
              , exception_mutex_()
              , exception_()
              , yield_delay_(yield_delay)
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
                using function_type = typename std::decay<F>::type;
                using shape_type = typename std::decay<S>::type;
                using argument_pack_type = typename std::decay<Tuple>::type;
                using index_pack_type =
                    typename hpx::util::detail::fused_index_pack<Tuple>::type;

                template <std::size_t... Is_, typename F_, typename A_,
                    typename Tuple_>
                static void invoke_helper(
                    hpx::util::index_pack<Is_...>, F_&& f, A_&& a, Tuple_&& t)
                {
                    hpx::util::invoke(
                        f, a, hpx::get<Is_>(std::forward<Tuple_>(t))...);
                }

                static void set_state(thread_states_type& thread_states,
                    std::size_t thread_index, thread_state state)
                {
                    thread_states[thread_index].data_ = state;
                }

                /// Main entry point for a single parallel region.
                static void call(void* element_function_void,
                    void const* shape_void, void* argument_pack_void,
                    std::size_t thread_index, std::size_t num_threads,
                    loop_schedule schedule, queues_type& queues,
                    thread_states_type& thread_states,
                    hpx::lcos::local::spinlock& exception_mutex,
                    std::exception_ptr& exception)
                {
                    try
                    {
                        // Cast void pointers back to the actual types given to
                        // bulk_sync_execute.
                        F& element_function =
                            *static_cast<F*>(element_function_void);
                        S const& shape = *static_cast<S const*>(shape_void);
                        Tuple argument_pack =
                            *static_cast<Tuple*>(argument_pack_void);

                        // Set up the local queues and state.
                        queue_type& local_queue = queues[thread_index].data_;
                        std::size_t size = hpx::util::size(shape);
                        init_local_work_queue(
                            local_queue, thread_index, num_threads, size);

                        set_state(
                            thread_states, thread_index, thread_state::active);

                        // Process local items first.
                        hpx::util::optional<std::uint32_t> index;
                        while ((index = local_queue.pop_left()))
                        {
                            auto it = hpx::util::begin(shape);
                            std::advance(it, index.value());
                            invoke_helper(index_pack_type{}, element_function,
                                *it, argument_pack);
                        }

                        if (schedule == loop_schedule::static_ ||
                            num_threads == 1)
                        {
                            set_state(thread_states, thread_index,
                                thread_state::idle);
                            return;
                        }

                        // If loop schedule is dynamic, steal from neighboring threads.
                        for (std::size_t offset = 1; offset < num_threads;
                             ++offset)
                        {
                            std::size_t neighbor_index =
                                (thread_index + offset) % num_threads;

                            if (thread_states[neighbor_index].data_.load() !=
                                thread_state::active)
                            {
                                continue;
                            }

                            queue_type& neighbor_queue =
                                queues[neighbor_index].data_;

                            while ((index = neighbor_queue.pop_right()))
                            {
                                auto it = hpx::util::begin(shape);
                                std::advance(it, index.value());
                                invoke_helper(index_pack_type{},
                                    element_function, *it, argument_pack);
                            }
                            set_state(thread_states, thread_index,
                                thread_state::idle);
                        }
                    }
                    catch (...)
                    {
                        std::lock_guard<hpx::lcos::local::spinlock> l(
                            exception_mutex);
                        if (!exception)
                        {
                            exception = std::current_exception();
                        }
                        set_state(
                            thread_states, thread_index, thread_state::idle);
                    }
                };
            };

        public:
            template <typename F, typename S, typename... Ts>
            void bulk_sync_execute(F&& f, S const& shape, Ts&&... ts)
            {
                // Set the data for this parallel region
                element_function_ = static_cast<void*>(&f);
                shape_ = static_cast<void const*>(&shape);
                size_ = hpx::util::size(shape);
                auto argument_pack = hpx::make_tuple<>(std::forward<Ts>(ts)...);
                argument_pack_ = static_cast<void*>(&argument_pack);
                thread_function_helper_ =
                    static_cast<thread_function_helper_type*>(
                        &thread_function_helper<typename std::decay<F>::type,
                            typename std::decay<S>::type,
                            decltype(argument_pack)>::call);

                // Signal all worker threads to start partitioning work for
                // themselves, and then starting the actual work.
                set_state_all(thread_state::partitioning_work);

                // Start work on the main thread.
                thread_function_helper<typename std::decay<F>::type,
                    typename std::decay<S>::type,
                    decltype(argument_pack)>::call(element_function_, shape_,
                    argument_pack_, main_thread_, num_threads_, schedule_,
                    queues_, thread_states_, exception_mutex_, exception_);

                wait_state_all(thread_state::idle);

                if (exception_)
                {
                    std::rethrow_exception(std::move(exception_));
                }
            }

            template <typename F, typename S, typename... Ts>
            std::vector<hpx::future<typename hpx::parallel::execution::detail::
                    bulk_function_result<F, S, Ts...>::type>>
            bulk_async_execute(F&& f, S const& shape, Ts&&... ts)
            {
                // Forward to the synchronous version as we can't create
                // futures to the completion of the parallel region (this HPX
                // thread participates in computation).
                using result_type = typename hpx::parallel::execution::detail::
                    bulk_function_result<F, S, Ts...>::type;
                std::vector<hpx::future<result_type>> v;
                try
                {
                    bulk_sync_execute(
                        std::forward<F>(f), shape, std::forward<Ts>(ts)...);
                }
                catch (...)
                {
                    v.push_back(hpx::make_exceptional_future<result_type>(
                        std::current_exception()));
                }

                return v;
            }
        };

    private:
        std::shared_ptr<shared_data> shared_data_;

    public:
        template <typename F, typename S, typename... Ts>
        void bulk_sync_execute(F&& f, S const& shape, Ts&&... ts)
        {
            shared_data_->bulk_sync_execute(
                std::forward<F>(f), shape, std::forward<Ts>(ts)...);
        }

        template <typename F, typename S, typename... Ts>
        decltype(auto) bulk_async_execute(F&& f, S const& shape, Ts&&... ts)
        {
            return shared_data_->bulk_async_execute(
                std::forward<F>(f), shape, std::forward<Ts>(ts)...);
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
        /// \param stacksize The stacksize of the worker threads.
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
          : shared_data_(
                new shared_data(priority, stacksize, schedule, yield_delay))
        {
        }
    };
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
