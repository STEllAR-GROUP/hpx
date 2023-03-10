//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_combinators/wait_all.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/async_local/dataflow.hpp>
#endif
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/util/detail/scoped_executor_parameters.hpp>
#include <hpx/parallel/util/detail/select_partitioner.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/type_support/unused.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::parallel::util {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        // The static partitioner with cleanup spawns several chunks of
        // iterations for each available core. The number of iterations is
        // determined automatically based on the measured runtime of the
        // iterations.
        template <typename ExPolicy, typename R, typename Result>
        struct static_partitioner_with_cleanup
        {
            using parameters_type = typename ExPolicy::executor_parameters_type;
            using executor_type = typename ExPolicy::executor_type;

            using handle_local_exceptions =
                detail::handle_local_exceptions<ExPolicy>;

            template <typename ExPolicy_, typename FwdIter, typename F1,
                typename F2, typename Cleanup>
            static R call(ExPolicy_&& policy, FwdIter first, std::size_t count,
                F1&& f1, F2&& f2, Cleanup&& cleanup)
            {
                // inform parameter traits
                using scoped_executor_parameters =
                    detail::scoped_executor_parameters_ref<parameters_type,
                        typename std::decay_t<ExPolicy_>::executor_type>;

                scoped_executor_parameters scoped_params(
                    policy.parameters(), policy.executor());

                try
                {
                    auto&& items = detail::partition<Result>(
                        HPX_FORWARD(ExPolicy_, policy), first, count,
                        HPX_FORWARD(F1, f1));

                    scoped_params.mark_end_of_scheduling();

                    return reduce(HPX_MOVE(items), HPX_FORWARD(F2, f2),
                        HPX_FORWARD(Cleanup, cleanup));
                }
                catch (...)
                {
                    handle_local_exceptions::call(std::current_exception());
                }

                HPX_UNREACHABLE;    //-V779
            }

        private:
            template <typename Items, typename F, typename Cleanup>
            static R reduce(Items&& workitems, F&& f, Cleanup&& cleanup)
            {
                // wait for all tasks to finish
                if (hpx::wait_all_nothrow(workitems))
                {
                    // always rethrow if 'errors' is not empty or workitems has
                    // exceptional future
                    handle_local_exceptions::call_with_cleanup(
                        workitems, HPX_FORWARD(Cleanup, cleanup));
                }
                return f(HPX_MOVE(workitems));
            }

            template <typename Items1, typename Items2, typename F,
                typename Cleanup>
            static R reduce(
                std::pair<Items1, Items2>&& items, F&& f, Cleanup&& cleanup)
            {
                if (items.first.empty())
                {
                    return reduce(HPX_MOVE(items.second), HPX_FORWARD(F, f),
                        HPX_FORWARD(Cleanup, cleanup));
                }

                items.first.insert(items.first.end(),
                    std::make_move_iterator(items.second.begin()),
                    std::make_move_iterator(items.second.end()));
                return reduce(HPX_MOVE(items.first), HPX_FORWARD(F, f),
                    HPX_FORWARD(Cleanup, cleanup));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename R, typename Result>
        struct task_static_partitioner_with_cleanup
        {
            using parameters_type = typename ExPolicy::executor_parameters_type;
            using executor_type = typename ExPolicy::executor_type;

            using scoped_executor_parameters =
                detail::scoped_executor_parameters<parameters_type,
                    executor_type>;

            using handle_local_exceptions =
                detail::handle_local_exceptions<ExPolicy>;

            template <typename ExPolicy_, typename FwdIter, typename F1,
                typename F2, typename Cleanup>
            static hpx::future<R> call(ExPolicy_&& policy, FwdIter first,
                std::size_t count, F1&& f1, F2&& f2, Cleanup&& cleanup)
            {
                // inform parameter traits
                std::shared_ptr<scoped_executor_parameters> scoped_params =
                    std::make_shared<scoped_executor_parameters>(
                        policy.parameters(), policy.executor());

                try
                {
                    auto&& items = detail::partition<Result>(
                        HPX_FORWARD(ExPolicy_, policy), first, count,
                        HPX_FORWARD(F1, f1));

                    scoped_params->mark_end_of_scheduling();

                    return reduce(HPX_MOVE(scoped_params), HPX_MOVE(items),
                        HPX_FORWARD(F2, f2), HPX_FORWARD(Cleanup, cleanup));
                }
                catch (std::bad_alloc const&)
                {
                    return hpx::make_exceptional_future<R>(
                        std::current_exception());
                }
            }

        private:
            template <typename Items1, typename Items2, typename F,
                typename Cleanup>
            static hpx::future<R> reduce(
                std::shared_ptr<scoped_executor_parameters>&& scoped_params,
                std::pair<Items1, Items2>&& items, F&& f, Cleanup&& cleanup)
            {
                if (items.first.empty())
                {
                    return reduce(HPX_MOVE(scoped_params),
                        HPX_MOVE(items.second), HPX_FORWARD(F, f),
                        HPX_FORWARD(Cleanup, cleanup));
                }

                items.first.insert(items.first.end(),
                    std::make_move_iterator(items.second.begin()),
                    std::make_move_iterator(items.second.end()));
                return reduce(HPX_MOVE(scoped_params), HPX_MOVE(items.first),
                    HPX_FORWARD(F, f), HPX_FORWARD(Cleanup, cleanup));
            }

            template <typename Items, typename F, typename Cleanup>
            static hpx::future<R> reduce(
                [[maybe_unused]] std::shared_ptr<scoped_executor_parameters>&&
                    scoped_params,
                [[maybe_unused]] Items&& workitems, [[maybe_unused]] F&& f,
                [[maybe_unused]] Cleanup&& cleanup)
            {
                // wait for all tasks to finish
#if defined(HPX_COMPUTE_DEVICE_CODE)
                HPX_ASSERT(false);
                return hpx::future<R>{};
#else
                return hpx::dataflow(
                    hpx::launch::sync,
                    [scoped_params = HPX_MOVE(scoped_params),
                        f = HPX_FORWARD(F, f),
                        cleanup = HPX_FORWARD(Cleanup, cleanup)](
                        auto&& r) mutable -> R {
                        HPX_UNUSED(scoped_params);

                        handle_local_exceptions::call_with_cleanup(
                            r, HPX_FORWARD(Cleanup, cleanup));

                        return f(HPX_MOVE(r));
                    },
                    HPX_MOVE(workitems));
#endif
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // ExPolicy: execution policy
    // R:        overall result type
    // Result:   intermediate result type of first step
    template <typename ExPolicy, typename R = void, typename Result = R>
    struct partitioner_with_cleanup
      : detail::select_partitioner<std::decay_t<ExPolicy>,
            detail::static_partitioner_with_cleanup,
            detail::task_static_partitioner_with_cleanup>::template apply<R,
            Result>
    {
    };
}    // namespace hpx::parallel::util
