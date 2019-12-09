//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/restricted_thread_pool_executors.hpp

#if !defined(                                                                  \
    HPX_PARALLEL_EXECUTORS_THREAD_POOL_ATTACHED_EXECUTORS_AUG_28_2015_0511PM)
#define HPX_PARALLEL_EXECUTORS_THREAD_POOL_ATTACHED_EXECUTORS_AUG_28_2015_0511PM

#include <hpx/config.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>

#include <cstdint>

namespace hpx { namespace parallel { namespace execution {
    // TODO: This is like thread_pool_executor but restricted to a subset of threads.
    // TODO: Remove duplication between this and thread_pool_executor.
    struct restricted_thread_pool_executor
    {
        /// Associate the parallel_execution_tag executor tag type as a default
        /// with this executor.
        typedef parallel_execution_tag execution_category;

        /// Associate the static_chunk_size executor parameters type as a default
        /// with this executor.
        typedef static_chunk_size executor_parameters_type;

        /// Create a new parallel executor
        restricted_thread_pool_executor(std::size_t first_thread = 0,
            std::size_t num_threads = 1,
            threads::thread_priority priority =
                threads::thread_priority_default,
            threads::thread_stacksize stacksize =
                threads::thread_stacksize_default,
            threads::thread_schedule_hint schedulehint = {})
          : pool_(this_thread::get_pool())
          , priority_(priority)
          , stacksize_(stacksize)
          , schedulehint_(schedulehint)
          , first_thread_(first_thread)
          , num_threads_(num_threads)
          , os_thread_(first_thread_)
        {
            HPX_ASSERT(pool_);
        }

        restricted_thread_pool_executor(
            restricted_thread_pool_executor const& other)
          : pool_(other.pool_)
          , priority_(other.priority_)
          , stacksize_(other.stacksize_)
          , schedulehint_(other.schedulehint_)
          , first_thread_(other.first_thread_)
          , num_threads_(other.num_threads_)
          , os_thread_(other.first_thread_)
        {
            HPX_ASSERT(pool_);
        }

        /// \cond NOINTERNAL
        bool operator==(restricted_thread_pool_executor const& rhs) const
            noexcept
        {
            return pool_ == rhs.pool_;
        }

        bool operator!=(restricted_thread_pool_executor const& rhs) const
            noexcept
        {
            return !(*this == rhs);
        }

        restricted_thread_pool_executor const& context() const noexcept
        {
            return *this;
        }
        /// \endcond

    private:
        std::int16_t get_next_thread_num()
        {
            return static_cast<std::int16_t>(
                first_thread_ + (os_thread_++ % num_threads_));
        }

        /// \cond NOINTERNAL

    public:
        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        hpx::future<
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F&& f, Ts&&... ts) const
        {
            return hpx::detail::async_launch_policy_dispatch<decltype(
                hpx::launch::async)>::call(hpx::launch::async, pool_,
                threads::thread_schedule_hint{
                    0    // get_next_thread_num()
                },
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename F, typename Future, typename... Ts>
        HPX_FORCEINLINE
            hpx::future<typename hpx::util::detail::invoke_deferred_result<F,
                Future, Ts...>::type>
            then_execute(F&& f, Future&& predecessor, Ts&&... ts)
        {
            using result_type =
                typename hpx::util::detail::invoke_deferred_result<F, Future,
                    Ts...>::type;

            auto&& func = hpx::util::one_shot(hpx::util::bind_back(
                std::forward<F>(f), std::forward<Ts>(ts)...));

            typename hpx::traits::detail::shared_state_ptr<result_type>::type
                p = lcos::detail::make_continuation_alloc_nounwrap<result_type>(
                    hpx::util::internal_allocator<>{},
                    std::forward<Future>(predecessor), hpx::launch::async,
                    std::move(func));

            return hpx::traits::future_access<hpx::future<result_type>>::create(
                std::move(p));
        }

        // NonBlockingOneWayExecutor (adapted) interface
        template <typename F, typename... Ts>
        void post(F&& f, Ts&&... ts) const
        {
            hpx::util::thread_description desc(
                f, "hpx::parallel::execution::parallel_executor::post");

            detail::post_policy_dispatch<decltype(hpx::launch::async)>::call(
                hpx::launch::async, desc, pool_,
                threads::thread_schedule_hint{
                    0    // get_next_thread_num()
                },
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        // BulkTwoWayExecutor interface
        template <typename F, typename S, typename... Ts>
        std::vector<hpx::future<
            typename detail::bulk_function_result<F, S, Ts...>::type>>
        bulk_async_execute(F&& f, S const& shape, Ts&&... ts) const
        {
            std::size_t const os_thread_count = pool_->get_os_thread_count();
            HPX_UNUSED(os_thread_count);
            HPX_ASSERT((first_thread_ + num_threads_) <= os_thread_count);
            hpx::util::thread_description const desc(f,
                "hpx::parallel::execution::restricted_thread_pool_executor::"
                "bulk_async_execute");

            typedef std::vector<hpx::future<
                typename detail::bulk_function_result<F, S, Ts...>::type>>
                result_type;

            result_type results;
            std::size_t const size = hpx::util::size(shape);
            results.resize(size);

            lcos::local::latch l(num_threads_);
            std::size_t part_begin = 0;
            auto it = std::begin(shape);
            for (std::size_t t = 0; t < num_threads_; ++t)
            {
                std::size_t const part_end = ((t + 1) * size) / num_threads_;
                threads::thread_schedule_hint hint{
                    static_cast<std::int16_t>(first_thread_ + t)};
                detail::post_policy_dispatch<decltype(
                    hpx::launch::async)>::call(hpx::launch::async, desc, pool_,
                    hint,

                    [&, this, hint, part_begin, part_end, f, it]() mutable {
                        for (std::size_t part_i = part_begin; part_i < part_end;
                             ++part_i)
                        {
                            results[part_i] =
                                hpx::detail::async_launch_policy_dispatch<
                                    decltype(hpx::launch::async)>::
                                    call(hpx::launch::async, pool_, hint, f,
                                        *it, ts...);
                            ++it;
                        }
                        l.count_down(1);
                    });
                std::advance(it, part_end - part_begin);
                part_begin = part_end;
            }

            l.wait();

            return results;
        }

        template <typename F, typename S, typename Future, typename... Ts>
        hpx::future<typename detail::bulk_then_execute_result<F, S, Future,
            Ts...>::type>
        bulk_then_execute(
            F&& f, S const& shape, Future&& predecessor, Ts&&... ts)
        {
            using func_result_type =
                typename detail::then_bulk_function_result<F, S, Future,
                    Ts...>::type;

            // std::vector<future<func_result_type>>
            using result_type = std::vector<hpx::future<func_result_type>>;

            auto&& func =
                detail::make_fused_bulk_async_execute_helper<result_type>(*this,
                    std::forward<F>(f), shape,
                    hpx::util::make_tuple(std::forward<Ts>(ts)...));

            // void or std::vector<func_result_type>
            using vector_result_type =
                typename detail::bulk_then_execute_result<F, S, Future,
                    Ts...>::type;

            // future<vector_result_type>
            using result_future_type = hpx::future<vector_result_type>;

            using shared_state_type =
                typename hpx::traits::detail::shared_state_ptr<
                    vector_result_type>::type;

            using future_type = typename std::decay<Future>::type;

            // vector<future<func_result_type>> -> vector<func_result_type>
            shared_state_type p =
                lcos::detail::make_continuation_alloc<vector_result_type>(
                    hpx::util::internal_allocator<>{},
                    std::forward<Future>(predecessor), hpx::launch::async,
                    [HPX_CAPTURE_MOVE(func)](future_type&& predecessor) mutable
                    -> vector_result_type {
                        // use unwrap directly (instead of lazily) to avoid
                        // having to pull in dataflow
                        return hpx::util::unwrap(func(std::move(predecessor)));
                    });

            return hpx::traits::future_access<result_future_type>::create(
                std::move(p));
            return hpx::make_ready_future();
        }
        /// \endcond

    private:
        threads::thread_pool_base* pool_;

        //  TODO: Actually use these
        threads::thread_priority priority_;
        threads::thread_stacksize stacksize_;
        threads::thread_schedule_hint schedulehint_;

        std::size_t first_thread_;
        std::size_t num_threads_;
        std::atomic<std::size_t> os_thread_;
    };

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    using local_queue_attached_executor = restricted_thread_pool_executor;
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
    using static_queue_attached_executor = restricted_thread_pool_executor;
#endif

    using local_priority_queue_attached_executor =
        restricted_thread_pool_executor;

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    using static_priority_queue_attached_executor =
        restricted_thread_pool_executor;
#endif
}}}    // namespace hpx::parallel::execution

namespace hpx { namespace parallel { namespace execution {
    /// \cond NOINTERNAL
    template <>
    struct is_one_way_executor<
        parallel::execution::restricted_thread_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<
        parallel::execution::restricted_thread_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<
        parallel::execution::restricted_thread_pool_executor> : std::true_type
    {
    };
    /// \endcond
}}}    // namespace hpx::parallel::execution

#endif
