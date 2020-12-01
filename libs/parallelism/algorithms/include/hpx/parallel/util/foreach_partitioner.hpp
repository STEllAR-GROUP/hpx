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

#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/util/detail/chunk_size.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/util/detail/partitioner_iteration.hpp>
#include <hpx/parallel/util/detail/scoped_executor_parameters.hpp>
#include <hpx/parallel/util/detail/select_partitioner.hpp>

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
    namespace detail {
        template <typename Result, typename ExPolicy, typename FwdIter,
            typename F>
        std::pair<std::vector<hpx::future<Result>>,
            std::vector<hpx::future<Result>>>
        foreach_partition(
            ExPolicy&& policy, FwdIter first, std::size_t count, F&& f)
        {
            // estimate a chunk size based on number of cores used
            using parameters_type =
                typename std::decay<ExPolicy>::type::executor_parameters_type;
            using has_variable_chunk_size =
                typename execution::extract_has_variable_chunk_size<
                    parameters_type>::type;

            std::vector<hpx::future<Result>> inititems;
            auto shape = detail::get_bulk_iteration_shape_idx(
                has_variable_chunk_size{}, std::forward<ExPolicy>(policy),
                inititems, f, first, count, 1);

            std::vector<hpx::future<Result>> workitems =
                execution::bulk_async_execute(policy.executor(),
                    partitioner_iteration<Result, F>{std::forward<F>(f)},
                    std::move(shape));
            return std::make_pair(std::move(inititems), std::move(workitems));
        }

        ///////////////////////////////////////////////////////////////////////
        // The static partitioner simply spawns one chunk of iterations for
        // each available core.
        template <typename ExPolicy, typename Result>
        struct foreach_static_partitioner
        {
            using parameters_type = typename ExPolicy::executor_parameters_type;
            using executor_type = typename ExPolicy::executor_type;

            using scoped_executor_parameters =
                detail::scoped_executor_parameters_ref<parameters_type,
                    executor_type>;

            using handle_local_exceptions =
                detail::handle_local_exceptions<ExPolicy>;

            template <typename ExPolicy_, typename FwdIter, typename F1,
                typename F2>
            static FwdIter call(ExPolicy_&& policy, FwdIter first,
                std::size_t count, F1&& f1, F2&& f2)
            {
                // inform parameter traits
                scoped_executor_parameters scoped_params(
                    policy.parameters(), policy.executor());

                FwdIter last = parallel::v1::detail::next(first, count);

                std::vector<hpx::future<Result>> inititems, workitems;
                std::list<std::exception_ptr> errors;
                try
                {
                    std::tie(inititems, workitems) =
                        detail::foreach_partition<Result>(
                            std::forward<ExPolicy_>(policy), first, count,
                            std::forward<F1>(f1));

                    scoped_params.mark_end_of_scheduling();
                }
                catch (...)
                {
                    handle_local_exceptions::call(
                        std::current_exception(), errors);
                }
                return reduce(std::move(inititems), std::move(workitems),
                    std::move(errors), std::forward<F2>(f2), std::move(last));
            }

        private:
            template <typename F, typename FwdIter>
            static FwdIter reduce(std::vector<hpx::future<Result>>&& inititems,
                std::vector<hpx::future<Result>>&& workitems,
                std::list<std::exception_ptr>&& errors, F&& f, FwdIter last)
            {
                // wait for all tasks to finish
                hpx::wait_all(workitems);

                // always rethrow if 'errors' is not empty or workitems has
                // exceptional future
                handle_local_exceptions::call(inititems, errors);
                handle_local_exceptions::call(workitems, errors);

                try
                {
                    return f(std::move(last));
                }
                catch (...)
                {
                    // rethrow either bad_alloc or exception_list
                    handle_local_exceptions::call(std::current_exception());
                }

                HPX_ASSERT(false);
                return last;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Result>
        struct foreach_task_static_partitioner
        {
            using parameters_type = typename ExPolicy::executor_parameters_type;
            using executor_type = typename ExPolicy::executor_type;

            using scoped_executor_parameters =
                detail::scoped_executor_parameters<parameters_type,
                    executor_type>;

            using handle_local_exceptions =
                detail::handle_local_exceptions<ExPolicy>;

            template <typename ExPolicy_, typename FwdIter, typename F1,
                typename F2>
            static hpx::future<FwdIter> call(ExPolicy_&& policy, FwdIter first,
                std::size_t count, F1&& f1, F2&& f2)
            {
                // inform parameter traits
                std::shared_ptr<scoped_executor_parameters> scoped_params =
                    std::make_shared<scoped_executor_parameters>(
                        policy.parameters(), policy.executor());

                FwdIter last = parallel::v1::detail::next(first, count);

                std::vector<hpx::future<Result>> inititems, workitems;
                std::list<std::exception_ptr> errors;
                try
                {
                    std::tie(inititems, workitems) =
                        detail::foreach_partition<Result>(
                            std::forward<ExPolicy_>(policy), first, count,
                            std::forward<F1>(f1));

                    scoped_params->mark_end_of_scheduling();
                }
                catch (std::bad_alloc const&)
                {
                    return hpx::make_exceptional_future<FwdIter>(
                        std::current_exception());
                }
                catch (...)
                {
                    handle_local_exceptions::call(
                        std::current_exception(), errors);
                }
                return reduce(std::move(scoped_params), std::move(inititems),
                    std::move(workitems), std::move(errors),
                    std::forward<F2>(f2), std::move(last));
            }

        private:
            template <typename F, typename FwdIter>
            static hpx::future<FwdIter> reduce(
                std::shared_ptr<scoped_executor_parameters>&& scoped_params,
                std::vector<hpx::future<Result>>&& inititems,
                std::vector<hpx::future<Result>>&& workitems,
                std::list<std::exception_ptr>&& errors, F&& f, FwdIter last)
            {
#if defined(HPX_COMPUTE_DEVICE_CODE)
                HPX_UNUSED(scoped_params);
                HPX_UNUSED(inititems);
                HPX_UNUSED(workitems);
                HPX_UNUSED(errors);
                HPX_UNUSED(f);
                HPX_UNUSED(last);
                HPX_ASSERT(false);
                return hpx::future<FwdIter>();
#else
                // wait for all tasks to finish
                return hpx::dataflow(
                    [last, errors = std::move(errors),
                        scoped_params = std::move(scoped_params),
                        f = std::forward<F>(f)](
                        std::vector<hpx::future<Result>>&& r1,
                        std::vector<hpx::future<Result>>&& r2) mutable
                    -> FwdIter {
                        HPX_UNUSED(scoped_params);

                        handle_local_exceptions::call(r1, errors);
                        handle_local_exceptions::call(r2, errors);
                        return f(std::move(last));
                    },
                    std::move(inititems), std::move(workitems));
#endif
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // ExPolicy: execution policy
    // Result:   intermediate result type of first step (default: void)
    template <typename ExPolicy, typename Result = void>
    struct foreach_partitioner
      : detail::select_partitioner<typename std::decay<ExPolicy>::type,
            detail::foreach_static_partitioner,
            detail::foreach_task_static_partitioner>::template apply<Result>
    {
    };
}}}    // namespace hpx::parallel::util
