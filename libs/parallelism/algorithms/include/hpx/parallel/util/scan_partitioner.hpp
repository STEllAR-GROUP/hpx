//  Copyright (c) 2007-2018 Hartmut Kaiser
//  Copyright (c)      2015 Daniel Bourgeois
//  Copyright (c)      2017 Taeguk Kwon
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

#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/chunk_size.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
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
    struct scan_partitioner_normal_tag
    {
    };
    struct scan_partitioner_sequential_f3_tag
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        // The static partitioner simply spawns one chunk of iterations for
        // each available core.
        template <typename ExPolicy, typename ScanPartTag, typename R,
            typename Result1, typename Result2>
        struct scan_static_partitioner
        {
            using parameters_type = typename ExPolicy::executor_parameters_type;
            using executor_type = typename ExPolicy::executor_type;

            using scoped_executor_parameters =
                detail::scoped_executor_parameters_ref<parameters_type,
                    executor_type>;

            using handle_local_exceptions =
                detail::handle_local_exceptions<ExPolicy>;

            template <typename ExPolicy_, typename FwdIter, typename T,
                typename F1, typename F2, typename F3, typename F4>
            static R call(scan_partitioner_normal_tag, ExPolicy_ policy,
                FwdIter first, std::size_t count, T&& init, F1&& f1, F2&& f2,
                F3&& f3, F4&& f4)
            {
#if defined(HPX_COMPUTE_DEVICE_CODE)
                HPX_UNUSED(policy);
                HPX_UNUSED(first);
                HPX_UNUSED(count);
                HPX_UNUSED(init);
                HPX_UNUSED(f1);
                HPX_UNUSED(f2);
                HPX_UNUSED(f3);
                HPX_UNUSED(f4);
                HPX_ASSERT(false);
                return R();
#else
                // inform parameter traits
                scoped_executor_parameters scoped_params(
                    policy.parameters(), policy.executor());

                std::vector<hpx::shared_future<Result1>> workitems;
                std::vector<hpx::future<Result2>> finalitems;
                std::list<std::exception_ptr> errors;
                try
                {
                    // pre-initialize first intermediate result
                    workitems.push_back(
                        make_ready_future(std::forward<T>(init)));

                    HPX_ASSERT(count > 0);
                    FwdIter first_ = first;
                    std::size_t count_ = count;

                    // estimate a chunk size based on number of cores used
                    typedef typename execution::extract_has_variable_chunk_size<
                        parameters_type>::type has_variable_chunk_size;

                    auto shape = detail::get_bulk_iteration_shape(
                        has_variable_chunk_size(), policy, workitems, f1, first,
                        count, 1);

                    // schedule every chunk on a separate thread
                    std::size_t size = hpx::util::size(shape);

                    // If the size of count was enough to warrant testing for a
                    // chunk, pre-initialize second intermediate result and
                    // start f3.
                    if (workitems.size() == 2)
                    {
                        HPX_ASSERT(count_ > count);

                        workitems.reserve(size + 2);
                        finalitems.reserve(size + 1);

                        hpx::shared_future<Result1> curr = workitems[1];
                        finalitems.push_back(dataflow(hpx::launch::sync, f3,
                            first_, count_ - count, workitems[0], curr));

                        workitems[1] =
                            dataflow(hpx::launch::sync, f2, workitems[0], curr);
                    }
                    else
                    {
                        workitems.reserve(size + 1);
                        finalitems.reserve(size);
                    }

                    // Schedule first step of scan algorithm, step 2 is
                    // performed as soon as the current partition and the
                    // partition to the left is ready.
                    for (auto const& elem : shape)
                    {
                        FwdIter it = hpx::get<0>(elem);
                        std::size_t size = hpx::get<1>(elem);

                        hpx::shared_future<Result1> prev = workitems.back();
                        auto curr = execution::async_execute(
                            policy.executor(), f1, it, size)
                                        .share();

                        finalitems.push_back(dataflow(
                            hpx::launch::sync, f3, it, size, prev, curr));

                        workitems.push_back(
                            dataflow(hpx::launch::sync, f2, prev, curr));
                    }

                    scoped_params.mark_end_of_scheduling();
                }
                catch (...)
                {
                    handle_local_exceptions::call(
                        std::current_exception(), errors);
                }
                return reduce(std::move(workitems), std::move(finalitems),
                    std::move(errors), std::forward<F4>(f4));
#endif
            }

            template <typename ExPolicy_, typename FwdIter, typename T,
                typename F1, typename F2, typename F3, typename F4>
            static R call(scan_partitioner_sequential_f3_tag, ExPolicy_ policy,
                FwdIter first, std::size_t count, T&& init, F1&& f1, F2&& f2,
                F3&& f3, F4&& f4)
            {
#if defined(HPX_COMPUTE_DEVICE_CODE)
                HPX_UNUSED(policy);
                HPX_UNUSED(first);
                HPX_UNUSED(count);
                HPX_UNUSED(init);
                HPX_UNUSED(f1);
                HPX_UNUSED(f2);
                HPX_UNUSED(f3);
                HPX_UNUSED(f4);
                HPX_ASSERT(false);
                return R();
#else
                // inform parameter traits
                scoped_executor_parameters scoped_params(
                    policy.parameters(), policy.executor());

                std::vector<hpx::shared_future<Result1>> workitems;
                std::vector<hpx::future<Result2>> finalitems;
                std::list<std::exception_ptr> errors;
                try
                {
                    // pre-initialize first intermediate result
                    workitems.push_back(
                        make_ready_future(std::forward<T>(init)));

                    HPX_ASSERT(count > 0);
                    FwdIter first_ = first;
                    std::size_t count_ = count;
                    bool tested = false;

                    // estimate a chunk size based on number of cores used
                    typedef typename execution::extract_has_variable_chunk_size<
                        parameters_type>::type has_variable_chunk_size;

                    auto shape = detail::get_bulk_iteration_shape(
                        has_variable_chunk_size(), policy, workitems, f1, first,
                        count, 1);

                    // schedule every chunk on a separate thread
                    std::size_t size = hpx::util::size(shape);

                    // If the size of count was enough to warrant testing for a
                    // chunk, pre-initialize second intermediate result.
                    if (workitems.size() == 2)
                    {
                        workitems.reserve(size + 2);
                        finalitems.reserve(size + 1);

                        hpx::shared_future<Result1> curr = workitems[1];
                        workitems[1] =
                            dataflow(hpx::launch::sync, f2, workitems[0], curr);
                        tested = true;
                    }
                    else
                    {
                        workitems.reserve(size + 1);
                        finalitems.reserve(size);
                    }

                    // Schedule first step of scan algorithm, step 2 is
                    // performed as soon as the current partition and the
                    // partition to the left is ready.
                    for (auto const& elem : shape)
                    {
                        FwdIter it = hpx::get<0>(elem);
                        std::size_t size = hpx::get<1>(elem);

                        hpx::shared_future<Result1> prev = workitems.back();
                        auto curr = execution::async_execute(
                            policy.executor(), f1, it, size)
                                        .share();

                        workitems.push_back(
                            dataflow(hpx::launch::sync, f2, prev, curr));
                    }

                    // In the code below, performs step 3 sequentially.
                    auto shape_iter = std::begin(shape);

                    // First, perform f3 of the first partition.
                    if (tested)
                    {
                        HPX_ASSERT(count_ > count);

                        finalitems.push_back(
                            dataflow(hpx::launch::sync, f3, first_,
                                count_ - count, workitems[0], workitems[1]));
                    }
                    else
                    {
                        auto elem = *shape_iter++;
                        FwdIter it = hpx::get<0>(elem);
                        std::size_t size = hpx::get<1>(elem);

                        finalitems.push_back(dataflow(hpx::launch::sync, f3, it,
                            size, workitems[0], workitems[1]));
                    }

                    HPX_ASSERT(finalitems.size() >= 1);

                    // Perform f3 sequentially from the second to the end
                    // of partitions.
                    for (std::size_t widx = 1ul; shape_iter != std::end(shape);
                         ++shape_iter, ++widx)
                    {
                        auto elem = *shape_iter;
                        FwdIter it = hpx::get<0>(elem);
                        std::size_t size = hpx::get<1>(elem);

                        // Wait the completion of f3 on previous partition.
                        finalitems.back().wait();

                        finalitems.push_back(dataflow(hpx::launch::sync, f3, it,
                            size, workitems[widx], workitems[widx + 1]));
                    }

                    scoped_params.mark_end_of_scheduling();
                }
                catch (...)
                {
                    handle_local_exceptions::call(
                        std::current_exception(), errors);
                }
                return reduce(std::move(workitems), std::move(finalitems),
                    std::move(errors), std::forward<F4>(f4));
#endif
            }

            template <typename ExPolicy_, typename FwdIter, typename T,
                typename F1, typename F2, typename F3, typename F4>
            static R call(ExPolicy_&& policy, FwdIter first, std::size_t count,
                T&& init, F1&& f1, F2&& f2, F3&& f3, F4&& f4)
            {
                return call(ScanPartTag{}, std::forward<ExPolicy_>(policy),
                    first, count, std::forward<T>(init), std::forward<F1>(f1),
                    std::forward<F2>(f2), std::forward<F3>(f3),
                    std::forward<F4>(f4));
            }

        private:
            template <typename F>
            static R reduce(
                std::vector<hpx::shared_future<Result1>>&& workitems,
                std::vector<hpx::future<Result2>>&& finalitems,
                std::list<std::exception_ptr>&& errors, F&& f)
            {
#if defined(HPX_COMPUTE_DEVICE_CODE)
                HPX_UNUSED(workitems);
                HPX_UNUSED(finalitems);
                HPX_UNUSED(errors);
                HPX_UNUSED(f);
                HPX_ASSERT(false);
                return R();
#else
                // wait for all tasks to finish
                hpx::wait_all(workitems, finalitems);

                // always rethrow if 'errors' is not empty or 'workitems' or
                // 'finalitems' have an exceptional future
                handle_local_exceptions::call(workitems, errors);
                handle_local_exceptions::call(finalitems, errors);

                try
                {
                    return f(std::move(workitems), std::move(finalitems));
                }
                catch (...)
                {
                    // rethrow either bad_alloc or exception_list
                    handle_local_exceptions::call(std::current_exception());
                }
#endif
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename ScanPartTag, typename R,
            typename Result1, typename Result2>
        struct scan_task_static_partitioner
        {
            template <typename ExPolicy_, typename FwdIter, typename T,
                typename F1, typename F2, typename F3, typename F4>
            static hpx::future<R> call(ExPolicy_&& policy, FwdIter first,
                std::size_t count, T&& init, F1&& f1, F2&& f2, F3&& f3, F4&& f4)
            {
                return execution::async_execute(policy.executor(),
                    [first, count, policy = std::forward<ExPolicy_>(policy),
                        init = std::forward<T>(init), f1 = std::forward<F1>(f1),
                        f2 = std::forward<F2>(f2), f3 = std::forward<F3>(f3),
                        f4 = std::forward<F4>(f4)]() mutable -> R {
                        using partitioner_type =
                            scan_static_partitioner<ExPolicy, ScanPartTag, R,
                                Result1, Result2>;
                        return partitioner_type::call(ScanPartTag{},
                            std::forward<ExPolicy_>(policy), first, count,
                            std::move(init), f1, f2, f3, f4);
                    });
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // ExPolicy:    execution policy
    // R:           overall result type
    // Result1:     intermediate result type of first and second step
    // Result2:     intermediate result of the third step
    // ScanPartTag: select appropriate policy of scan partitioner
    template <typename ExPolicy, typename R = void, typename Result1 = R,
        typename Result2 = void,
        typename ScanPartTag = scan_partitioner_normal_tag>
    struct scan_partitioner
      : detail::select_partitioner<typename std::decay<ExPolicy>::type,
            detail::scan_static_partitioner,
            detail::scan_task_static_partitioner>::template apply<ScanPartTag,
            R, Result1, Result2>
    {
    };
}}}    // namespace hpx::parallel::util
