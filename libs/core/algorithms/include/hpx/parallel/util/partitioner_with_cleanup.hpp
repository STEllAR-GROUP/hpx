//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_combinators/wait_all.hpp>
#include <hpx/modules/errors.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/async_local/dataflow.hpp>
#endif
#include <hpx/type_support/unused.hpp>

#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/util/detail/chunk_size.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/util/detail/partitioner_iteration.hpp>
#include <hpx/parallel/util/detail/scoped_executor_parameters.hpp>
#include <hpx/parallel/util/detail/select_partitioner.hpp>
#include <hpx/parallel/util/partitioner.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util {
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

            using scoped_executor_parameters =
                detail::scoped_executor_parameters_ref<parameters_type,
                    executor_type>;

            using handle_local_exceptions =
                detail::handle_local_exceptions<ExPolicy>;

            template <typename ExPolicy_, typename FwdIter, typename F1,
                typename F2, typename Cleanup>
            static R call(ExPolicy_&& policy, FwdIter first, std::size_t count,
                F1&& f1, F2&& f2, Cleanup&& cleanup)
            {
                // inform parameter traits
                scoped_executor_parameters scoped_params(
                    policy.parameters(), policy.executor());

                std::vector<hpx::future<Result>> workitems;
                std::list<std::exception_ptr> errors;
                try
                {
                    workitems = detail::partition<Result>(
                        std::forward<ExPolicy_>(policy), first, count,
                        std::forward<F1>(f1));

                    scoped_params.mark_end_of_scheduling();
                }
                catch (...)
                {
                    handle_local_exceptions::call(
                        std::current_exception(), errors);
                }
                return reduce(std::move(workitems), std::move(errors),
                    std::forward<F2>(f2), std::forward<Cleanup>(cleanup));
            }

        private:
            template <typename F, typename Cleanup>
            static R reduce(std::vector<hpx::future<Result>>&& workitems,
                std::list<std::exception_ptr>&& errors, F&& f,
                Cleanup&& cleanup)
            {
                // wait for all tasks to finish
                hpx::wait_all(workitems);

                // always rethrow if 'errors' is not empty or workitems has
                // exceptional future
                handle_local_exceptions::call_with_cleanup(
                    workitems, errors, std::forward<Cleanup>(cleanup));

                try
                {
                    return f(std::move(workitems));
                }
                catch (...)
                {
                    // rethrow either bad_alloc or exception_list
                    handle_local_exceptions::call(std::current_exception());
                    HPX_ASSERT(false);
                    return f(std::move(workitems));
                }
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

                std::vector<hpx::future<Result>> workitems;
                std::list<std::exception_ptr> errors;
                try
                {
                    workitems = detail::partition<Result>(
                        std::forward<ExPolicy_>(policy), first, count,
                        std::forward<F1>(f1));

                    scoped_params->mark_end_of_scheduling();
                }
                catch (std::bad_alloc const&)
                {
                    return hpx::make_exceptional_future<R>(
                        std::current_exception());
                }
                catch (...)
                {
                    handle_local_exceptions::call(
                        std::current_exception(), errors);
                }
                return reduce(std::move(scoped_params), std::move(workitems),
                    std::move(errors), std::forward<F2>(f2),
                    std::forward<Cleanup>(cleanup));
            }

        private:
            template <typename F, typename Cleanup>
            static hpx::future<R> reduce(
                std::shared_ptr<scoped_executor_parameters>&& scoped_params,
                std::vector<hpx::future<Result>>&& workitems,
                std::list<std::exception_ptr>&& errors, F&& f,
                Cleanup&& cleanup)
            {
                // wait for all tasks to finish
#if defined(HPX_COMPUTE_DEVICE_CODE)
                HPX_UNUSED(scoped_params);
                HPX_UNUSED(workitems);
                HPX_UNUSED(errors);
                HPX_UNUSED(f);
                HPX_UNUSED(cleanup);
                HPX_ASSERT(false);
                return hpx::future<R>{};
#else
                return hpx::dataflow(
                    [errors = std::move(errors),
                        scoped_params = std::move(scoped_params),
                        f = std::forward<F>(f),
                        cleanup = std::forward<Cleanup>(cleanup)](
                        std::vector<hpx::future<Result>>&& r) mutable -> R {
                        HPX_UNUSED(scoped_params);

                        handle_local_exceptions::call_with_cleanup(
                            r, errors, std::forward<Cleanup>(cleanup));
                        return f(std::move(r));
                    },
                    std::move(workitems));
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
      : detail::select_partitioner<typename std::decay<ExPolicy>::type,
            detail::static_partitioner_with_cleanup,
            detail::task_static_partitioner_with_cleanup>::template apply<R,
            Result>
    {
    };
}}}    // namespace hpx::parallel::util
