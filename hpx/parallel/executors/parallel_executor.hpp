//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/parallel_executor.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_PARALLEL_EXECUTOR_MAY_13_2015_1057AM)
#define HPX_PARALLEL_EXECUTORS_PARALLEL_EXECUTOR_MAY_13_2015_1057AM

#include <hpx/config.hpp>
#include <hpx/async_launch_policy_dispatch.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/when_all_fwd.hpp>
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/executors/post_policy_dispatch.hpp>
#include <hpx/parallel/executors/static_chunk_size.hpp>
#include <hpx/runtime/get_worker_thread_num.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/range.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { namespace execution
{
    namespace detail
    {
        template <typename Policy>
        struct get_default_policy
        {
            static HPX_CONSTEXPR Policy call()
            {
                return Policy{};
            }
        };

        template <>
        struct get_default_policy<hpx::launch>
        {
            static HPX_CONSTEXPR hpx::launch::async_policy call()
            {
                return hpx::launch::async_policy{};
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Shape, typename ... Ts>
        struct bulk_function_result;
    }

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
        HPX_CONSTEXPR explicit parallel_policy_executor(
                Policy l = detail::get_default_policy<Policy>::call(),
                std::size_t spread = 4, std::size_t tasks = std::size_t(-1))
          : l_(l), num_spread_(spread), num_tasks_(tasks)
        {}

        /// \cond NOINTERNAL
        bool operator==(parallel_policy_executor const& rhs) const noexcept
        {
            return l_ == rhs.l_ &&
                num_spread_ == rhs.num_spread_ &&
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
        template <typename F, typename ... Ts>
        hpx::future<
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type
        >
        async_execute(F && f, Ts &&... ts) const
        {
            return hpx::detail::async_launch_policy_dispatch<Policy>::call(
                l_, std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        // NonBlockingOneWayExecutor (adapted) interface
        template <typename F, typename ... Ts>
        void post(F && f, Ts &&... ts)
        {
            hpx::util::thread_description desc(f,
                "hpx::parallel::execution::parallel_executor::post");

            detail::post_policy_dispatch<Policy>::call(
                desc, l_, std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        // BulkTwoWayExecutor interface
        template <typename F, typename S, typename ... Ts>
        std::vector<hpx::future<
            typename detail::bulk_function_result<F, S, Ts...>::type
        > >
        bulk_async_execute(F && f, S const& shape, Ts &&... ts) const
        {
            // lazily initialize once
            static std::size_t global_num_tasks =
                (std::min)(std::size_t(128), hpx::get_os_thread_count());

            std::size_t num_tasks =
                (num_tasks_ == std::size_t(-1)) ? global_num_tasks : num_tasks_;

            typedef std::vector<hpx::future<
                    typename detail::bulk_function_result<
                        F, S, Ts...
                    >::type
                > > result_type;

            result_type results;
            std::size_t size = hpx::util::size(shape);
            results.resize(size);

            spawn(results, 0, size, num_tasks, f,
                hpx::util::begin(shape), ts...).get();

            return results;
        }
        /// \endcond

    protected:
        /// \cond NOINTERNAL
        template <typename Result, typename F, typename Iter, typename ... Ts>
        hpx::future<void> spawn(std::vector<hpx::future<Result> >& results,
            std::size_t base, std::size_t size, std::size_t num_tasks,
            F const& func, Iter it, Ts const&... ts) const
        {
            if (size > num_tasks)
            {
                // spawn hierarchical tasks
                std::size_t chunk_size = (size + num_spread_) / num_spread_ - 1;
                chunk_size = (std::max)(chunk_size, num_tasks);

                std::vector<hpx::future<void> > tasks;
                tasks.reserve(num_spread_);

                hpx::future<void> (parallel_policy_executor::*spawn_func)(
                        std::vector<hpx::future<Result> >&, std::size_t,
                        std::size_t, std::size_t, F const&, Iter, Ts const&...
                    ) const = &parallel_policy_executor::spawn;

                while (size != 0)
                {
                    std::size_t curr_chunk_size = (std::min)(chunk_size, size);

                    hpx::future<void> f = async_execute(
                        spawn_func, this, std::ref(results), base,
                        curr_chunk_size, num_tasks, std::ref(func), it,
                        std::ref(ts)...);
                    tasks.push_back(std::move(f));

                    base += curr_chunk_size;
                    it = hpx::parallel::v1::detail::next(it, curr_chunk_size);
                    size -= curr_chunk_size;
                }

                HPX_ASSERT(size == 0);

                return hpx::lcos::when_all_fwd(std::move(tasks));
            }

            // spawn all tasks sequentially
            HPX_ASSERT(base + size <= results.size());

            for (std::size_t i = 0; i != size; ++i, ++it)
            {
                results[base + i] = async_execute(func, *it, ts...);
            }

            return hpx::make_ready_future();
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & l_ & num_spread_ & num_tasks_;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        Policy l_;
        std::size_t num_spread_;
        std::size_t num_tasks_;
        /// \endcond
    };

    using parallel_executor = parallel_policy_executor<hpx::launch>;
}}}

namespace hpx { namespace parallel { namespace execution
{
    /// \cond NOINTERNAL
    template <typename Policy>
    struct is_one_way_executor<
            parallel::execution::parallel_policy_executor<Policy> >
      : std::true_type
    {};

    template <typename Policy>
    struct is_two_way_executor<
            parallel::execution::parallel_policy_executor<Policy> >
      : std::true_type
    {};

    template <typename Policy>
    struct is_bulk_two_way_executor<
            parallel::execution::parallel_policy_executor<Policy> >
      : std::true_type
    {};
    /// \endcond
}}}

#endif
