//  Copyright (c) 2007-2019 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/parallel_executor_aggregated.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_PARALLEL_EXECUTOR_AGGREGATED_DEC_20_2018_0624PM)
#define HPX_PARALLEL_EXECUTORS_PARALLEL_EXECUTOR_AGGREGATED_DEC_20_2018_0624PM

#include <hpx/config.hpp>
#include <hpx/async_launch_policy_dispatch.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/counting_semaphore.hpp>
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/executors/parallel_executor.hpp>
#include <hpx/parallel/executors/post_policy_dispatch.hpp>
#include <hpx/parallel/executors/static_chunk_size.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/range.hpp>
#include <hpx/util/tuple.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { namespace execution
{
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
        typedef parallel_execution_tag execution_category;

        /// Associate the static_chunk_size executor parameters type as a default
        /// with this executor.
        typedef static_chunk_size executor_parameters_type;

        /// Create a new parallel executor
        HPX_CONSTEXPR explicit parallel_policy_executor_aggregated() {}

        /// \cond NOINTERNAL
        bool operator==(parallel_policy_executor_aggregated const& rhs) const
            noexcept
        {
            return true;
        }

        bool operator!=(parallel_policy_executor_aggregated const& rhs) const
            noexcept
        {
            return false;
        }

        parallel_policy_executor_aggregated const& context() const noexcept
        {
            return *this;
        }
        /// \endcond

        /// \cond NOINTERNAL

        // TwoWayExecutor interface
        template <typename F, typename ... Ts>
        hpx::future<void>
        async_execute(F && f, Ts &&... ts) const
        {
            return hpx::detail::async_launch_policy_dispatch<Policy>::call(
                Policy{}, std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        // NonBlockingOneWayExecutor (adapted) interface
        template <typename F, typename ... Ts>
        static void post(F && f, Ts &&... ts)
        {
            hpx::util::thread_description desc(f,
                "hpx::parallel::execution::parallel_executor_aggregated::post");

            detail::post_policy_dispatch<Policy>::call(
                desc, Policy{}, std::forward<F>(f), std::forward<Ts>(ts)...);
        }

    private:
        /// \cond NOINTERNAL
        template <typename F, typename ... Ts>
        struct wrapped_func
        {
            template <typename Value>
            void operator()(
                lcos::local::counting_semaphore& sem, Value const& value)
            {
                // invoke user function
                hpx::util::invoke_fused(f_,
                    hpx::util::tuple_cat(
                        hpx::util::forward_as_tuple(value), ts_));

                // make sure to signal the semaphore before exiting
                sem.signal(1);
            }

            F f_;
            hpx::util::tuple<Ts const&...> ts_;
        };

        template <typename F, typename ... Ts>
        HPX_FORCEINLINE static
        wrapped_func<F, Ts...> make_wrapped_func(F && f, Ts &&... ts)
        {
            return wrapped_func<F, Ts...>{std::forward<F>(f),
                hpx::util::forward_as_tuple(std::forward<Ts>(ts)...)};
        }

        struct sync_exec
        {
            template <typename F, typename S, typename... Ts>
            void operator()(F&& f, S const& shape, Ts&&... ts) const
            {
                // Simple sequential spawning of tasks, one task for each
                // requested iteration.
                lcos::local::counting_semaphore sem(0);

                auto wrapped = make_wrapped_func(
                    std::forward<F>(f), std::forward<Ts>(ts)...);

                auto end = hpx::util::end(shape);
                for (auto it = hpx::util::begin(shape); it != end; ++it)
                {
                    post(wrapped, std::ref(sem), *it);
                }

                sem.wait(static_cast<std::int64_t>(hpx::util::size(shape)));
            }
        };
        /// \endcond

    public:
        // BulkTwoWayExecutor interface
        template <typename F, typename S, typename... Ts>
        std::vector<hpx::future<void>>
        bulk_async_execute(F&& f, S const& shape, Ts&&... ts) const
        {
            // for now, wrap single future in a vector to avoid having to change
            // the executor and algorithm infrastructure
            std::vector<hpx::future<void>> result;
            result.emplace_back(
                async_execute(sync_exec{}, std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...));

            return result;
        }

        template <typename F, typename S, typename... Ts>
        void bulk_sync_execute(F&& f, S const& shape, Ts&&... ts) const
        {
            sync_exec{}(std::forward<F>(f), shape, std::forward<Ts>(ts)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    struct parallel_policy_executor_aggregated<hpx::launch>
    {
        /// Associate the parallel_execution_tag executor tag type as a default
        /// with this executor.
        typedef parallel_execution_tag execution_category;

        /// Associate the static_chunk_size executor parameters type as a default
        /// with this executor.
        typedef static_chunk_size executor_parameters_type;

        /// Create a new parallel executor
        HPX_CONSTEXPR explicit parallel_policy_executor_aggregated(
                hpx::launch l = hpx::launch::async_policy{})
          : l_(l)
        {}

        /// \cond NOINTERNAL
        bool operator==(parallel_policy_executor_aggregated const& rhs) const
            noexcept
        {
            return l_ == rhs.l_;
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
        template <typename F, typename ... Ts>
        hpx::future<void>
        async_execute(F && f, Ts &&... ts) const
        {
            return hpx::detail::async_launch_policy_dispatch<Policy>::call(
                l_, std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        // NonBlockingOneWayExecutor (adapted) interface
        template <typename F, typename ... Ts>
        void post(F && f, Ts &&... ts) const
        {
            hpx::util::thread_description desc(f,
                "hpx::parallel::execution::parallel_executor_aggregated::post");

            detail::post_policy_dispatch<Policy>::call(
                desc, l_, std::forward<F>(f), std::forward<Ts>(ts)...);
        }

    private:
        /// \cond NOINTERNAL
        template <typename F, typename ...Ts>
        struct wrapped_func
        {
            template <typename Value>
            void operator()(
                lcos::local::counting_semaphore& sem, Value const& value)
            {
                // invoke user function
                hpx::util::invoke_fused(f_,
                    hpx::util::tuple_cat(
                        hpx::util::forward_as_tuple(value), ts_));

                // make sure to signal the semaphore before exiting
                sem.signal(1);
            }

            F f_;
            hpx::util::tuple<Ts const&...> ts_;
        };

        template <typename F, typename ... Ts>
        HPX_FORCEINLINE static
        wrapped_func<F, Ts...> make_wrapped_func(F && f, Ts &&... ts)
        {
            return wrapped_func<F, Ts...>{std::forward<F>(f),
                hpx::util::forward_as_tuple(std::forward<Ts>(ts)...)};
        }

        struct sync_exec
        {
            template <typename F, typename S, typename... Ts>
            void operator()(F&& f, S const& shape, Ts&&... ts) const
            {
                // Simple sequential spawning of tasks, one task for each
                // requested iteration.
                lcos::local::counting_semaphore sem(0);

                auto wrapped = make_wrapped_func(
                    std::forward<F>(f), std::forward<Ts>(ts)...);

                for (auto const& elem : shape)
                {
                    post(wrapped, std::ref(sem), std::ref(elem));
                }

                sem.wait(static_cast<std::int64_t>(hpx::util::size(shape)));
            }

            parallel_policy_executor_aggregated const& this_;
        };
        /// \endcond

    public:
        // BulkTwoWayExecutor interface
        template <typename F, typename S, typename... Ts>
        std::vector<hpx::future<void>>
        bulk_async_execute(F&& f, S const& shape, Ts&&... ts) const
        {
            // for now, wrap single future in a vector to avoid having to change
            // the executor and algorithm infrastructure
            std::vector<hpx::future<void>> result;
            result.emplace_back(
                async_execute(sync_exec{*this}, std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...));

            return result;
        }

        template <typename F, typename S, typename... Ts>
        void bulk_sync_execute(F&& f, S const& shape, Ts&&... ts) const
        {
            sync_exec{*this}(
                std::forward<F>(f), shape, std::forward<Ts>(ts)...);
        }

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & l_;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        hpx::launch l_;
        /// \endcond
    };

    ///////////////////////////////////////////////////////////////////////////
    using parallel_executor_aggregated =
        parallel_policy_executor_aggregated<hpx::launch::async_policy>;
}}}

namespace hpx { namespace parallel { namespace execution
{
    /// \cond NOINTERNAL
    template <typename Policy>
    struct is_one_way_executor<
            parallel::execution::parallel_policy_executor_aggregated<Policy> >
      : std::true_type
    {};

    template <typename Policy>
    struct is_two_way_executor<
            parallel::execution::parallel_policy_executor_aggregated<Policy> >
      : std::true_type
    {};

    template <typename Policy>
    struct is_bulk_two_way_executor<
            parallel::execution::parallel_policy_executor_aggregated<Policy> >
      : std::true_type
    {};
    /// \endcond
}}}

#endif
