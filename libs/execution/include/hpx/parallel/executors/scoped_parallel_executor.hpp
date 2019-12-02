//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/scoped_parallel_executor.hpp

#if !defined(HPX_EXECUTORS_SCOPED_PARALLEL_EXECUTOR_DEC_01_2019_0951AM)
#define HPX_EXECUTORS_SCOPED_PARALLEL_EXECUTOR_DEC_01_2019_0951AM

#include <hpx/config.hpp>

#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/assertion.hpp>
#include <hpx/async_launch_policy_dispatch.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/one_shot.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/executors/fused_bulk_execute.hpp>
#include <hpx/parallel/executors/post_policy_dispatch.hpp>
#include <hpx/parallel/executors/static_chunk_size.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/synchronization/latch.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/util/unwrap.hpp>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { namespace execution {

    ///////////////////////////////////////////////////////////////////////////
    /// A \a scoped_parallel_executor creates groups of parallel execution
    /// agents which execute in threads implicitly created by the executor. This
    /// executor assumes that all created tasks are fully scoped by the calling
    /// parent task.
    ///
    /// This executor conforms to the concepts of a TwoWayExecutor,
    /// and a BulkTwoWayExecutor
    template <typename Policy>
    struct scoped_parallel_policy_executor
    {
        /// Associate the parallel_execution_tag executor tag type as a default
        /// with this executor.
        typedef parallel_execution_tag execution_category;

        /// Associate the static_chunk_size executor parameters type as a default
        /// with this executor.
        typedef static_chunk_size executor_parameters_type;

        /// Create a new parallel executor
        HPX_CONSTEXPR explicit scoped_parallel_policy_executor(
            Policy l = detail::get_default_policy<Policy>::call(),
            std::size_t spread = 4, std::size_t tasks = std::size_t(-1))
          : policy_(l)
          , num_spread_(spread)
          , num_tasks_(tasks)
        {
        }

        /// \cond NOINTERNAL
        bool operator==(scoped_parallel_policy_executor const& rhs) const
            noexcept
        {
            return policy_ == rhs.policy_ && num_spread_ == rhs.num_spread_ &&
                num_tasks_ == rhs.num_tasks_;
        }

        bool operator!=(scoped_parallel_policy_executor const& rhs) const
            noexcept
        {
            return !(*this == rhs);
        }

        scoped_parallel_policy_executor const& context() const noexcept
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
            threads::thread_schedule_hint hint;
            hint.runs_as_child = true;

            return hpx::detail::async_launch_policy_dispatch<Policy>::call(
                policy_, hint, std::forward<F>(f), std::forward<Ts>(ts)...);
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
        /// \endcond

    protected:
        /// \cond NOINTERNAL
        template <typename Result, typename F, typename Iter, typename... Ts>
        void spawn_sequential(std::vector<hpx::future<Result>>& results,
            lcos::local::latch& l, std::size_t base, std::size_t size,
            F const& func, Iter it, Ts const&... ts) const
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
            std::size_t num_tasks, F const& func, Iter it,
            Ts const&... ts) const
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
        Policy policy_;
        std::size_t num_spread_;
        std::size_t num_tasks_;
        /// \endcond
    };

    using scoped_parallel_executor =
        scoped_parallel_policy_executor<hpx::launch>;

}}}    // namespace hpx::parallel::execution

namespace hpx { namespace parallel { namespace execution {

    /// \cond NOINTERNAL
    template <typename Policy>
    struct is_two_way_executor<
        parallel::execution::scoped_parallel_policy_executor<Policy>>
      : std::true_type
    {
    };

    template <typename Policy>
    struct is_bulk_two_way_executor<
        parallel::execution::scoped_parallel_policy_executor<Policy>>
      : std::true_type
    {
    };
    /// \endcond
}}}    // namespace hpx::parallel::execution

#endif
