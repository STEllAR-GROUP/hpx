//  Copyright (c) 2007-2019 Hartmut Kaiser
//  Copyright (c) 2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/parallel_executor_aggregated.hpp

#if !defined(                                                                  \
    HPX_PARALLEL_EXECUTORS_PARALLEL_EXECUTOR_AGGREGATED_DEC_20_2018_0624PM)
#define HPX_PARALLEL_EXECUTORS_PARALLEL_EXECUTOR_AGGREGATED_DEC_20_2018_0624PM

#include <hpx/config.hpp>
#include <hpx/async_launch_policy_dispatch.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/latch.hpp>
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/executors/post_policy_dispatch.hpp>
#include <hpx/parallel/executors/static_chunk_size.hpp>
#include <hpx/runtime/get_worker_thread_num.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/traits/is_executor.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    /// A \a parallel_executor_aggregated creates groups of parallel execution
    /// agents that execute in threads implicitly created by the executor. This
    /// executor prefers continuing with the creating thread first before
    /// executing newly created threads.
    ///
    /// This executor conforms to the concepts of a TwoWayExecutor,
    /// and a BulkTwoWayExecutor
    template <typename Policy = hpx::launch::async_policy>
    struct parallel_policy_executor_aggregated
    {
        /// Associate the parallel_execution_tag executor tag type as a default
        /// with this executor.
        using execution_category = parallel_execution_tag;

        /// Associate the static_chunk_size executor parameters type as a default
        /// with this executor.
        using executor_parameters_type = static_chunk_size;

        /// Create a new parallel executor
        HPX_CONSTEXPR explicit parallel_policy_executor_aggregated(
            std::size_t spread = 4, std::size_t tasks = std::size_t(-1))
          : num_spread_(spread)
          , num_tasks_(tasks)
        {
        }

        /// \cond NOINTERNAL
        bool operator==(parallel_policy_executor_aggregated const& rhs) const
            noexcept
        {
            return num_spread_ == rhs.num_spread_ &&
                num_tasks_ == rhs.num_tasks_;
        }

        bool operator!=(parallel_policy_executor_aggregated const& rhs) const
            noexcept
        {
            return !(*this == rhs);
        }

        parallel_policy_executor_aggregated const& context() const noexcept
        {
            return *this;
        }
        /// \endcond

        /// \cond NOINTERNAL

        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        static hpx::future<void> async_execute(F&& f, Ts&&... ts)
        {
            return hpx::detail::async_launch_policy_dispatch<Policy>::call(
                Policy{}, std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        // NonBlockingOneWayExecutor (adapted) interface
        template <typename F, typename... Ts>
        static void post(F&& f, Ts&&... ts)
        {
            hpx::util::thread_description desc(
                f, "parallel_executor_aggregated::post");

            detail::post_policy_dispatch<Policy>::call(
                desc, Policy{}, std::forward<F>(f), std::forward<Ts>(ts)...);
        }

    private:
        /// \cond NOINTERNAL
        struct sync_exec
        {
            template <typename F, typename S, typename... Ts>
            void operator()(F&& f, S const& shape, Ts&&... ts) const
            {
                // lazily initialize once
                static std::size_t global_num_tasks =
                    (std::min)(std::size_t(128), hpx::get_os_thread_count());

                std::size_t num_tasks = (num_tasks_ == std::size_t(-1)) ?
                    global_num_tasks :
                    num_tasks_;

                std::exception_ptr e;
                lcos::local::spinlock mtx_e;

                std::size_t size = hpx::util::size(shape);
                lcos::local::latch l(size);
                if (hpx::detail::has_async_policy(Policy{}))
                {
                    spawn_hierarchical(l, size, num_tasks, std::forward<F>(f),
                        hpx::util::begin(shape), e, mtx_e, ts...);
                }
                else
                {
                    spawn_sequential(l, size, std::forward<F>(f),
                        hpx::util::begin(shape), e, mtx_e, ts...);
                }
                l.wait();

                // rethrow any exceptions caught during processing the
                // bulk_execute, note that we don't need to acquire the lock
                // at this point as no other threads may access the exception
                // concurrently
                if (e)
                {
                    std::rethrow_exception(std::move(e));
                }
            }

            template <typename F, typename Iter, typename... Ts>
            void spawn_sequential(lcos::local::latch& l, std::size_t size,
                F&& f, Iter it, std::exception_ptr& e,
                lcos::local::spinlock& mtx_e, Ts const&... ts) const
            {
                // spawn tasks sequentially
                for (std::size_t i = 0; i != size; ++i, ++it)
                {
                    post([&, it] {
                        // properly handle all exceptions thrown from 'f'
                        try
                        {
                            hpx::util::invoke(f, *it, ts...);
                        }
                        catch (...)
                        {
                            // store the first caught exception only
                            std::lock_guard<lcos::local::spinlock> l(mtx_e);
                            if (!e)
                                e = std::current_exception();
                        }
                        // count down the latch in any case
                        l.count_down(1);
                    });
                }
            }

            template <typename F, typename Iter, typename... Ts>
            void spawn_hierarchical(lcos::local::latch& l, std::size_t size,
                std::size_t num_tasks, F&& f, Iter it, std::exception_ptr& e,
                lcos::local::spinlock& mtx_e, Ts const&... ts) const
            {
                if (size > num_tasks)
                {
                    // spawn hierarchical tasks
                    std::size_t chunk_size =
                        (size + num_spread_) / num_spread_ - 1;
                    chunk_size = (std::max)(chunk_size, num_tasks);

                    while (size > chunk_size)
                    {
                        post([&, chunk_size, num_tasks, it] {
                            spawn_hierarchical(l, chunk_size, num_tasks, f, it,
                                e, mtx_e, ts...);
                        });

                        it = hpx::parallel::v1::detail::next(it, chunk_size);
                        size -= chunk_size;
                    }
                }

                // spawn remaining tasks sequentially
                spawn_sequential(
                    l, size, std::forward<F>(f), it, e, mtx_e, ts...);
            }

            std::size_t const num_spread_;
            std::size_t const num_tasks_;
        };
        /// \endcond

    public:
        // BulkTwoWayExecutor interface
        template <typename F, typename S, typename... Ts>
        std::vector<hpx::future<void>> bulk_async_execute(
            F&& f, S const& shape, Ts&&... ts) const
        {
            // for now, wrap single future in a vector to avoid having to
            // change the executor and algorithm infrastructure
            std::vector<hpx::future<void>> result;
            result.push_back(async_execute(sync_exec{num_spread_, num_tasks_},
                std::forward<F>(f), shape, std::forward<Ts>(ts)...));
            return result;
        }

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int version)
        {
            ar& num_spread_& num_tasks_;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        std::size_t const num_spread_;
        std::size_t const num_tasks_;
        /// \endcond
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    struct parallel_policy_executor_aggregated<hpx::launch>
    {
        /// Associate the parallel_execution_tag executor tag type as a default
        /// with this executor.
        using execution_category = parallel_execution_tag;

        /// Associate the static_chunk_size executor parameters type as a default
        /// with this executor.
        using executor_parameters_type = static_chunk_size;

        /// Create a new parallel executor
        HPX_CONSTEXPR explicit parallel_policy_executor_aggregated(
            hpx::launch l = hpx::launch::async_policy{}, std::size_t spread = 4,
            std::size_t tasks = std::size_t(-1))
          : policy_(l)
          , num_spread_(spread)
          , num_tasks_(tasks)
        {
        }

        /// \cond NOINTERNAL
        bool operator==(parallel_policy_executor_aggregated const& rhs) const
            noexcept
        {
            return policy_ == rhs.policy_ && num_spread_ == rhs.num_spread_ &&
                num_tasks_ == rhs.num_tasks_;
        }

        bool operator!=(parallel_policy_executor_aggregated const& rhs) const
            noexcept
        {
            return !(*this == rhs);
        }

        parallel_policy_executor_aggregated const& context() const noexcept
        {
            return *this;
        }
        /// \endcond

        /// \cond NOINTERNAL

        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        hpx::future<void> async_execute(F&& f, Ts&&... ts) const
        {
            return hpx::detail::async_launch_policy_dispatch<hpx::launch>::call(
                policy_, std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        // NonBlockingOneWayExecutor (adapted) interface
        template <typename F, typename... Ts>
        void post(F&& f, Ts&&... ts) const
        {
            hpx::util::thread_description desc(
                f, "parallel_executor_aggregated::post");

            detail::post_policy_dispatch<hpx::launch>::call(
                desc, policy_, std::forward<F>(f), std::forward<Ts>(ts)...);
        }

    private:
        /// \cond NOINTERNAL
        struct sync_exec
        {
            template <typename F, typename S, typename... Ts>
            void operator()(F&& f, S const& shape, Ts&&... ts) const
            {
                // lazily initialize once
                static std::size_t global_num_tasks =
                    (std::min)(std::size_t(128), hpx::get_os_thread_count());

                std::size_t num_tasks = (num_tasks_ == std::size_t(-1)) ?
                    global_num_tasks :
                    num_tasks_;

                std::exception_ptr e;
                lcos::local::spinlock mtx_e;

                std::size_t size = hpx::util::size(shape);
                lcos::local::latch l(size);
                if (hpx::detail::has_async_policy(policy_))
                {
                    spawn_hierarchical(l, size, num_tasks, std::forward<F>(f),
                        hpx::util::begin(shape), e, mtx_e, ts...);
                }
                else
                {
                    spawn_sequential(l, size, std::forward<F>(f),
                        hpx::util::begin(shape), e, mtx_e, ts...);
                }
                l.wait();

                // rethrow any exceptions caught during processing the
                // bulk_execute, note that we don't need to acquire the lock
                // at this point as no other threads may access the exception
                // concurrently
                if (e)
                {
                    std::rethrow_exception(std::move(e));
                }
            }

            template <typename F, typename Iter, typename... Ts>
            void spawn_sequential(lcos::local::latch& l, std::size_t size,
                F&& f, Iter it, std::exception_ptr& e,
                lcos::local::spinlock& mtx_e, Ts const&... ts) const
            {
                // spawn tasks sequentially
                hpx::util::thread_description desc(
                    f, "parallel_executor_aggregated::spawn_sequential");

                for (std::size_t i = 0; i != size; ++i, ++it)
                {
                    detail::post_policy_dispatch<hpx::launch>::call(
                        desc, policy_, [&, it]() -> void {
                            // properly handle all exceptions thrown from 'f'
                            try
                            {
                                hpx::util::invoke(f, *it, ts...);
                            }
                            catch (...)
                            {
                                // store the first caught exception only
                                std::lock_guard<lcos::local::spinlock> l(mtx_e);
                                if (!e)
                                    e = std::current_exception();
                            }
                            // count down the latch in any case
                            l.count_down(1);
                        });
                }
            }

            template <typename F, typename Iter, typename... Ts>
            void spawn_hierarchical(lcos::local::latch& l, std::size_t size,
                std::size_t num_tasks, F&& f, Iter it, std::exception_ptr& e,
                lcos::local::spinlock& mtx_e, Ts const&... ts) const
            {
                if (size > num_tasks)
                {
                    // spawn hierarchical tasks
                    hpx::util::thread_description desc(
                        f, "parallel_executor_aggregated::spawn_hierarchical");

                    std::size_t chunk_size =
                        (size + num_spread_) / num_spread_ - 1;
                    chunk_size = (std::max)(chunk_size, num_tasks);

                    while (size > chunk_size)
                    {
                        detail::post_policy_dispatch<hpx::launch>::call(
                            desc, policy_, [&, chunk_size, num_tasks, it] {
                                spawn_hierarchical(l, chunk_size, num_tasks, f,
                                    it, e, mtx_e, ts...);
                            });

                        it = hpx::parallel::v1::detail::next(it, chunk_size);
                        size -= chunk_size;
                    }
                }

                // spawn remaining tasks sequentially
                spawn_sequential(
                    l, size, std::forward<F>(f), it, e, mtx_e, ts...);
            }

            hpx::launch const policy_;
            std::size_t const num_spread_;
            std::size_t const num_tasks_;
        };
        /// \endcond

    public:
        // BulkTwoWayExecutor interface
        template <typename F, typename S, typename... Ts>
        std::vector<hpx::future<void>> bulk_async_execute(
            F&& f, S const& shape, Ts&&... ts) const
        {
            // for now, wrap single future in a vector to avoid having to
            // change the executor and algorithm infrastructure
            std::vector<hpx::future<void>> result;
            result.push_back(
                async_execute(sync_exec{policy_, num_spread_, num_tasks_},
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...));
            return result;
        }

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int version)
        {
            ar& policy_& num_spread_& num_tasks_;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        hpx::launch const policy_;
        std::size_t const num_spread_;
        std::size_t const num_tasks_;
        /// \endcond
    };

    ///////////////////////////////////////////////////////////////////////////
    using parallel_executor_aggregated =
        parallel_policy_executor_aggregated<hpx::launch::async_policy>;
}}}    // namespace hpx::parallel::execution

namespace hpx { namespace parallel { namespace execution {
    /// \cond NOINTERNAL
    template <typename Policy>
    struct is_one_way_executor<
        parallel::execution::parallel_policy_executor_aggregated<Policy>>
      : std::true_type
    {
    };

    template <typename Policy>
    struct is_two_way_executor<
        parallel::execution::parallel_policy_executor_aggregated<Policy>>
      : std::true_type
    {
    };

    template <typename Policy>
    struct is_bulk_two_way_executor<
        parallel::execution::parallel_policy_executor_aggregated<Policy>>
      : std::true_type
    {
    };
    /// \endcond
}}}    // namespace hpx::parallel::execution

#endif
