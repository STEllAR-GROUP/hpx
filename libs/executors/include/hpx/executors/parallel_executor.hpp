//  Copyright (c) 2007-2019 Hartmut Kaiser
//  Copyright (c) 2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/parallel_executor.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_PARALLEL_EXECUTOR_MAY_13_2015_1057AM)
#define HPX_PARALLEL_EXECUTORS_PARALLEL_EXECUTOR_MAY_13_2015_1057AM

#include <hpx/config.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/assertion.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/execution/detail/async_launch_policy_dispatch.hpp>
#include <hpx/execution/detail/post_policy_dispatch.hpp>
#include <hpx/execution/executors/fused_bulk_execute.hpp>
#include <hpx/execution/executors/static_chunk_size.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/one_shot.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/pack_traversal/unwrap.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/synchronization/latch.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>
#include <hpx/traits/future_traits.hpp>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { namespace execution {
    namespace detail {
        template <typename Policy>
        struct get_default_policy
        {
            static constexpr Policy call()
            {
                return Policy{};
            }
        };

        template <>
        struct get_default_policy<hpx::launch>
        {
            static constexpr hpx::launch::async_policy call()
            {
                return hpx::launch::async_policy{};
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Shape, typename... Ts>
        struct bulk_function_result;

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Shape, typename Future, typename... Ts>
        struct bulk_then_execute_result;

        template <typename F, typename Shape, typename Future, typename... Ts>
        struct then_bulk_function_result;
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    /// A \a parallel_executor creates groups of parallel execution agents
    /// which execute in threads implicitly created by the executor. This
    /// executor prefers continuing with the creating thread first before
    /// executing newly created threads.
    ///
    /// This executor conforms to the concepts of a TwoWayExecutor,
    /// and a BulkTwoWayExecutor
    template <typename Policy>
    struct parallel_policy_executor
    {
        /// Associate the parallel_execution_tag executor tag type as a default
        /// with this executor.
        typedef parallel_execution_tag execution_category;

        /// Associate the static_chunk_size executor parameters type as a default
        /// with this executor.
        typedef static_chunk_size executor_parameters_type;

        /// Create a new parallel executor
        constexpr explicit parallel_policy_executor(
            threads::thread_priority priority =
                threads::thread_priority_default,
            threads::thread_stacksize stacksize =
                threads::thread_stacksize_default,
            threads::thread_schedule_hint schedulehint = {},
            Policy l = detail::get_default_policy<Policy>::call(),
            std::size_t spread = 4, std::size_t tasks = std::size_t(-1))
          : priority_(priority)
          , stacksize_(stacksize)
          , schedulehint_(schedulehint)
          , policy_(l)
          , num_spread_(spread)
          , num_tasks_(tasks)
        {
        }

        constexpr explicit parallel_policy_executor(
            threads::thread_stacksize stacksize,
            threads::thread_schedule_hint schedulehint = {},
            Policy l = detail::get_default_policy<Policy>::call(),
            std::size_t spread = 4, std::size_t tasks = std::size_t(-1))
          : priority_(l.priority())
          , stacksize_(stacksize)
          , schedulehint_(schedulehint)
          , policy_(l)
          , num_spread_(spread)
          , num_tasks_(tasks)
        {
        }

        constexpr explicit parallel_policy_executor(
            threads::thread_schedule_hint schedulehint,
            Policy l = detail::get_default_policy<Policy>::call(),
            std::size_t spread = 4, std::size_t tasks = std::size_t(-1))
          : priority_(l.priority())
          , stacksize_(threads::thread_stacksize_default)
          , schedulehint_(schedulehint)
          , policy_(l)
          , num_spread_(spread)
          , num_tasks_(tasks)
        {
        }

        constexpr explicit parallel_policy_executor(Policy l,
            std::size_t spread = 4, std::size_t tasks = std::size_t(-1))
          : priority_(l.priority())
          , stacksize_(threads::thread_stacksize_default)
          , schedulehint_()
          , policy_(l)
          , num_spread_(spread)
          , num_tasks_(tasks)
        {
        }

        /// \cond NOINTERNAL
        bool operator==(parallel_policy_executor const& rhs) const noexcept
        {
            return policy_ == rhs.policy_ && num_spread_ == rhs.num_spread_ &&
                num_tasks_ == rhs.num_tasks_;
        }

        bool operator!=(parallel_policy_executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        parallel_policy_executor const& context() const noexcept
        {
            return *this;
        }
        /// \endcond

        /// \cond NOINTERNAL

        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        hpx::future<
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F&& f, Ts&&... ts) const
        {
            return hpx::detail::async_launch_policy_dispatch<Policy>::call(
                policy_, priority_, stacksize_, schedulehint_,
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename F, typename Future, typename... Ts>
        HPX_FORCEINLINE
            hpx::future<typename hpx::util::detail::invoke_deferred_result<F,
                Future, Ts...>::type>
            then_execute(F&& f, Future&& predecessor, Ts&&... ts) const
        {
            using result_type =
                typename hpx::util::detail::invoke_deferred_result<F, Future,
                    Ts...>::type;

            auto&& func = hpx::util::one_shot(hpx::util::bind_back(
                std::forward<F>(f), std::forward<Ts>(ts)...));

            typename hpx::traits::detail::shared_state_ptr<result_type>::type
                p = lcos::detail::make_continuation_alloc_nounwrap<result_type>(
                    hpx::util::internal_allocator<>{},
                    std::forward<Future>(predecessor), policy_,
                    std::move(func));

            return hpx::traits::future_access<hpx::future<result_type>>::create(
                std::move(p));
        }

        // NonBlockingOneWayExecutor (adapted) interface
        template <typename F, typename... Ts>
        void post(F&& f, Ts&&... ts) const
        {
            char const* annotation =
                hpx::traits::get_function_annotation<F>::call(f);
            hpx::util::thread_description desc(f, annotation);

            detail::post_policy_dispatch<Policy>::call(policy_, desc, priority_,
                stacksize_, schedulehint_, std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

        // BulkTwoWayExecutor interface
        template <typename F, typename S, typename... Ts>
        std::vector<hpx::future<
            typename detail::bulk_function_result<F, S, Ts...>::type>>
        bulk_async_execute(F&& f, S const& shape, Ts&&... ts) const
        {
            std::size_t num_tasks = num_tasks_;
            if (num_tasks == std::size_t(-1))
            {
                auto pool = threads::detail::get_self_or_default_pool();
                num_tasks =
                    (std::min)(std::size_t(128), pool->get_os_thread_count());
            }

            typedef std::vector<hpx::future<
                typename detail::bulk_function_result<F, S, Ts...>::type>>
                result_type;

            result_type results;
            std::size_t size = hpx::util::size(shape);
            results.resize(size);

            lcos::local::latch l(size);
            if (hpx::detail::has_async_policy(policy_))
            {
                spawn_hierarchical(results, l, 0, size, num_tasks, f,
                    hpx::util::begin(shape), ts...);
            }
            else
            {
                spawn_sequential(
                    results, l, 0, size, f, hpx::util::begin(shape), ts...);
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
                    std::forward<Future>(predecessor), policy_,
                    [func = std::move(func)](future_type&& predecessor) mutable
                    -> vector_result_type {
                        // use unwrap directly (instead of lazily) to avoid
                        // having to pull in dataflow
                        return hpx::util::unwrap(func(std::move(predecessor)));
                    });

            return hpx::traits::future_access<result_future_type>::create(
                std::move(p));
        }
        /// \endcond

    protected:
        /// \cond NOINTERNAL
        template <typename Result, typename F, typename Iter, typename... Ts>
        void spawn_sequential(std::vector<hpx::future<Result>>& results,
            lcos::local::latch& l, std::size_t base, std::size_t size, F&& func,
            Iter it, Ts&&... ts) const
        {
            // spawn tasks sequentially
            HPX_ASSERT(base + size <= results.size());

            for (std::size_t i = 0; i != size; ++i, ++it)
            {
                results[base + i] = async_execute(func, *it, ts...);
            }

            l.count_down(size);
        }

        template <typename Result, typename F, typename Iter, typename... Ts>
        void spawn_hierarchical(std::vector<hpx::future<Result>>& results,
            lcos::local::latch& l, std::size_t base, std::size_t size,
            std::size_t num_tasks, F&& func, Iter it, Ts&&... ts) const
        {
            if (size > num_tasks)
            {
                // spawn hierarchical tasks
                std::size_t chunk_size = (size + num_spread_) / num_spread_ - 1;
                chunk_size = (std::max)(chunk_size, num_tasks);

                while (size > chunk_size)
                {
                    post([&, base, chunk_size, num_tasks, it] {
                        spawn_hierarchical(results, l, base, chunk_size,
                            num_tasks, func, it, ts...);
                    });

                    base += chunk_size;
                    it = hpx::parallel::v1::detail::next(it, chunk_size);
                    size -= chunk_size;
                }
            }

            // spawn remaining tasks sequentially
            spawn_sequential(results, l, base, size, func, it, ts...);
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int version)
        {
            // clang-format off
            ar & policy_ & num_spread_ & num_tasks_;
            // clang-format on
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        threads::thread_priority priority_;
        threads::thread_stacksize stacksize_;
        threads::thread_schedule_hint schedulehint_;
        Policy policy_;
        std::size_t num_spread_;
        std::size_t num_tasks_;
        /// \endcond
    };

    using parallel_executor = parallel_policy_executor<hpx::launch>;
}}}    // namespace hpx::parallel::execution

namespace hpx { namespace parallel { namespace execution {
    /// \cond NOINTERNAL
    template <typename Policy>
    struct is_one_way_executor<
        parallel::execution::parallel_policy_executor<Policy>> : std::true_type
    {
    };

    template <typename Policy>
    struct is_two_way_executor<
        parallel::execution::parallel_policy_executor<Policy>> : std::true_type
    {
    };

    template <typename Policy>
    struct is_bulk_two_way_executor<
        parallel::execution::parallel_policy_executor<Policy>> : std::true_type
    {
    };
    /// \endcond
}}}    // namespace hpx::parallel::execution

#endif
