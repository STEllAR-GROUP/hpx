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
#include <hpx/algorithms/traits/is_pair.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/execution/algorithms/then.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution_base/traits/is_executor_parameters.hpp>
#include <hpx/parallel/util/detail/chunk_size.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/util/detail/partitioner_iteration.hpp>
#include <hpx/parallel/util/detail/scoped_executor_parameters.hpp>
#include <hpx/parallel/util/detail/select_partitioner.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::parallel::util::detail {

    template <typename Result, typename ExPolicy, typename FwdIter, typename F>
    auto foreach_partition(
        ExPolicy&& policy, FwdIter first, std::size_t count, F&& f)
    {
        // estimate a chunk size based on number of cores used
        using parameters_type =
            execution::extract_executor_parameters_t<std::decay_t<ExPolicy>>;
        constexpr bool has_variable_chunk_size =
            execution::extract_has_variable_chunk_size_v<parameters_type>;
        constexpr bool invokes_testing_function =
            execution::extract_invokes_testing_function_v<parameters_type>;

        if constexpr (has_variable_chunk_size)
        {
            static_assert(!invokes_testing_function,
                "parameters object should not expose both, "
                "has_variable_chunk_size and invokes_testing_function");

            auto&& shape = detail::get_bulk_iteration_shape_idx_variable(
                HPX_FORWARD(ExPolicy, policy), first, count);

            return execution::bulk_async_execute(policy.executor(),
                partitioner_iteration<Result, F>{HPX_FORWARD(F, f)},
                HPX_MOVE(shape));
        }
        else if constexpr (!invokes_testing_function)
        {
            auto&& shape = detail::get_bulk_iteration_shape_idx(
                HPX_FORWARD(ExPolicy, policy), first, count);

            return execution::bulk_async_execute(policy.executor(),
                partitioner_iteration<Result, F>{HPX_FORWARD(F, f)},
                HPX_MOVE(shape));
        }
        else
        {
            std::vector<hpx::future<Result>> inititems;
            auto&& shape = detail::get_bulk_iteration_shape_idx(
                HPX_FORWARD(ExPolicy, policy), inititems, f, first, count);

            auto&& workitems = execution::bulk_async_execute(policy.executor(),
                partitioner_iteration<Result, F>{HPX_FORWARD(F, f)},
                HPX_MOVE(shape));

            return std::make_pair(HPX_MOVE(inititems), HPX_MOVE(workitems));
        }
    }

    ///////////////////////////////////////////////////////////////////////
    // The static partitioner simply spawns one chunk of iterations for
    // each available core.
    template <typename ExPolicy, typename Result>
    struct foreach_static_partitioner
    {
        using parameters_type = typename ExPolicy::executor_parameters_type;
        using executor_type = typename ExPolicy::executor_type;

        using handle_local_exceptions =
            detail::handle_local_exceptions<ExPolicy>;

        template <typename ExPolicy_, typename FwdIter, typename F1,
            typename F2>
        static decltype(auto) call(ExPolicy_&& policy, FwdIter first,
            std::size_t count, F1&& f1, F2&& f2)
        {
            // inform parameter traits
            using scoped_executor_parameters =
                detail::scoped_executor_parameters_ref<parameters_type,
                    typename std::decay_t<ExPolicy_>::executor_type>;

            scoped_executor_parameters scoped_params(
                policy.parameters(), policy.executor());

            FwdIter last = parallel::detail::next(first, count);
            try
            {
                auto&& items = detail::foreach_partition<Result>(
                    HPX_FORWARD(ExPolicy_, policy), first, count,
                    HPX_FORWARD(F1, f1));

                scoped_params.mark_end_of_scheduling();

                return reduce(
                    HPX_MOVE(items), HPX_FORWARD(F2, f2), HPX_MOVE(last));
            }
            catch (...)
            {
                handle_local_exceptions::call(std::current_exception());
            }
        }

    private:
        template <typename F, typename Items1, typename Items2,
            typename FwdIter>
        static auto reduce(
            std::pair<Items1, Items2>&& items, F&& f, FwdIter last)
        {
            // wait for all tasks to finish
            if (hpx::wait_all_nothrow(hpx::get<0>(items), hpx::get<1>(items)))
            {
                // always rethrow if inititems/workitems have at least one
                // exceptional future
                handle_local_exceptions::call(hpx::get<0>(items));
                handle_local_exceptions::call(hpx::get<1>(items));
            }

            // gcc 9 complains here if HPX_INVOKE is used.
            return hpx::invoke(f, HPX_MOVE(last));
        }

        template <typename F, typename Items, typename FwdIter,
            typename Enable =
                std::enable_if_t<!hpx::traits::is_pair_v<std::decay_t<Items>>>>
        static auto reduce(Items&& items, F&& f, FwdIter last)
        {
            namespace ex = hpx::execution::experimental;
            if constexpr (ex::is_sender_v<std::decay_t<Items>> &&
                !hpx::traits::is_future_v<std::decay_t<Items>>)
            {
                // the predecessor sender could be exposing zero or more
                // value types
                return ex::then(HPX_FORWARD(Items, items),
                    [f = HPX_FORWARD(F, f), last = HPX_MOVE(last)]() mutable {
                        return HPX_INVOKE(f, HPX_MOVE(last));
                    });
            }
            else
            {
                // wait for all tasks to finish
                if (hpx::wait_all_nothrow(items))
                {
                    // always rethrow if items has at least one exceptional
                    // future
                    handle_local_exceptions::call(items);
                }
                return HPX_INVOKE(f, HPX_MOVE(last));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Result>
    struct foreach_task_static_partitioner
    {
        using parameters_type = typename ExPolicy::executor_parameters_type;
        using executor_type = typename ExPolicy::executor_type;

        using scoped_executor_parameters =
            detail::scoped_executor_parameters<parameters_type, executor_type>;

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

            FwdIter last = parallel::detail::next(first, count);
            try
            {
                auto&& items = detail::foreach_partition<Result>(
                    HPX_FORWARD(ExPolicy_, policy), first, count,
                    HPX_FORWARD(F1, f1));

                scoped_params->mark_end_of_scheduling();

                return reduce(HPX_MOVE(scoped_params), HPX_MOVE(items),
                    HPX_FORWARD(F2, f2), HPX_MOVE(last));
            }
            catch (...)
            {
                return hpx::make_exceptional_future<FwdIter>(
                    std::current_exception());
            }
        }

    private:
        template <typename F, typename Items1, typename Items2,
            typename FwdIter>
        static hpx::future<FwdIter> reduce(
            [[maybe_unused]] std::shared_ptr<scoped_executor_parameters>&&
                scoped_params,
            [[maybe_unused]] std::pair<Items1, Items2>&& items,
            [[maybe_unused]] F&& f, [[maybe_unused]] FwdIter last)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_ASSERT(false);
            return hpx::future<FwdIter>();
#else
            // wait for all tasks to finish
            // Note: the lambda takes the vectors by value (dataflow
            //       moves those into the lambda) to ensure that they
            //       will be destroyed before the lambda exists.
            //       Otherwise the vectors stay alive in the dataflow's
            //       shared state and may reference data that has gone
            //       out of scope.
            return hpx::dataflow(
                hpx::launch::sync,
                [last, scoped_params = HPX_MOVE(scoped_params),
                    f = HPX_FORWARD(F, f)](
                    auto&& r1, auto&& r2) mutable -> FwdIter {
                    HPX_UNUSED(scoped_params);

                    handle_local_exceptions::call(r1);
                    handle_local_exceptions::call(r2);

                    return f(HPX_MOVE(last));
                },
                HPX_MOVE(hpx::get<0>(items)), HPX_MOVE(hpx::get<1>(items)));
#endif
        }

        template <typename F, typename Items, typename FwdIter,
            typename Enable =
                std::enable_if_t<!hpx::traits::is_pair_v<std::decay_t<Items>>>>
        static hpx::future<FwdIter> reduce(
            [[maybe_unused]] std::shared_ptr<scoped_executor_parameters>&&
                scoped_params,
            [[maybe_unused]] Items&& items, [[maybe_unused]] F&& f,
            [[maybe_unused]] FwdIter last)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_ASSERT(false);
            return hpx::future<FwdIter>();
#else
            // wait for all tasks to finish
            return hpx::dataflow(
                hpx::launch::sync,
                [last, scoped_params = HPX_MOVE(scoped_params),
                    f = HPX_FORWARD(F, f)](auto&& r) mutable -> FwdIter {
                    HPX_UNUSED(scoped_params);

                    handle_local_exceptions::call(r);

                    return f(HPX_MOVE(last));
                },
                HPX_MOVE(items));
#endif
        }

        template <typename F, typename FwdIter>
        static hpx::future<FwdIter> reduce(
            [[maybe_unused]] std::shared_ptr<scoped_executor_parameters>&&
                scoped_params,
            [[maybe_unused]] hpx::future<void>&& item, [[maybe_unused]] F&& f,
            [[maybe_unused]] FwdIter last)
        {
#if defined(HPX_COMPUTE_DEVICE_CODE)
            HPX_ASSERT(false);
            return hpx::future<FwdIter>();
#else
            // wait for the task to finish, invoke the given function before
            // returning
            return item.then(hpx::launch::sync,
                [last, scoped_params = HPX_MOVE(scoped_params),
                    f = HPX_FORWARD(F, f)](auto&& r) mutable -> FwdIter {
                    HPX_UNUSED(scoped_params);

                    handle_local_exceptions::call(HPX_MOVE(r));

                    return f(HPX_MOVE(last));
                });
#endif
        }
    };
}    // namespace hpx::parallel::util::detail

namespace hpx::parallel::util {

    ///////////////////////////////////////////////////////////////////////////
    // ExPolicy: execution policy
    // Result:   intermediate result type of first step (default: void)
    template <typename ExPolicy, typename Result = void>
    struct foreach_partitioner
      : detail::select_partitioner<std::decay_t<ExPolicy>,
            detail::foreach_static_partitioner,
            detail::foreach_task_static_partitioner>::template apply<Result>
    {
    };
}    // namespace hpx::parallel::util
