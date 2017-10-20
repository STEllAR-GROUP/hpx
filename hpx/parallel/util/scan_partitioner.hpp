//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2015 Daniel Bourgeois
//  Copyright (c)      2017 Taeguk Kwon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_SCAN_PARTITIONER_DEC_30_2014_0227PM)
#define HPX_PARALLEL_UTIL_SCAN_PARTITIONER_DEC_30_2014_0227PM

#include <hpx/config.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/unused.hpp>

#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/executors/execution.hpp>
#include <hpx/parallel/executors/execution_parameters.hpp>
#include <hpx/parallel/traits/extract_partitioner.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/chunk_size.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/util/detail/scoped_executor_parameters.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <list>
#include <memory>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util
{
    struct scan_partitioner_normal_tag {};
    struct scan_partitioner_sequential_f3_tag {};

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // The static partitioner simply spawns one chunk of iterations for
        // each available core.
        template <typename R, typename Result1, typename Result2,
            typename ScanPartTag>
        struct static_scan_partitioner_helper;

        template <typename R, typename Result1, typename Result2>
        struct static_scan_partitioner_helper<R, Result1, Result2,
            scan_partitioner_normal_tag>
        {
            template <typename ExPolicy, typename FwdIter, typename T,
                typename F1, typename F2, typename F3, typename F4>
            static R call(ExPolicy && policy, FwdIter first,
                std::size_t count, T && init, F1 && f1, F2 && f2, F3 && f3,
                F4 && f4)
            {
                typedef typename
                    hpx::util::decay<ExPolicy>::type::executor_parameters_type
                    parameters_type;
                typedef typename
                    hpx::util::decay<ExPolicy>::type::executor_type
                    executor_type;

                // inform parameter traits
                scoped_executor_parameters_ref<
                        parameters_type, executor_type
                    > scoped_param(policy.parameters(), policy.executor());

                std::vector<hpx::shared_future<Result1> > workitems;
                std::vector<hpx::future<Result2> > finalitems;
                std::list<std::exception_ptr> errors;

                try {
                    // pre-initialize first intermediate result
                    workitems.push_back(make_ready_future(std::forward<T>(init)));

                    HPX_ASSERT(count > 0);
                    FwdIter first_ = first;
                    std::size_t count_ = count;

                    // estimate a chunk size based on number of cores used
                    typedef typename execution::extract_has_variable_chunk_size<
                            parameters_type
                        >::type has_variable_chunk_size;

                    auto shape = get_bulk_iteration_shape(policy, workitems,
                        f1, first, count, 1, has_variable_chunk_size());

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
                        finalitems.push_back(dataflow(hpx::launch::sync,
                            f3, first_, count_ - count, workitems[0], curr));

                        workitems[1] = dataflow(hpx::launch::sync,
                            f2, workitems[0], curr);
                    }
                    else
                    {
                        workitems.reserve(size + 1);
                        finalitems.reserve(size);
                    }

                    // Schedule first step of scan algorithm, step 2 is
                    // performed as soon as the current partition and the
                    // partition to the left is ready.
                    for(auto const& elem: shape)
                    {
                        FwdIter it = hpx::util::get<0>(elem);
                        std::size_t size = hpx::util::get<1>(elem);

                        hpx::shared_future<Result1> prev = workitems.back();
                        auto curr = execution::async_execute(
                            policy.executor(), f1, it, size).share();

                        finalitems.push_back(dataflow(hpx::launch::sync,
                            f3, it, size, prev, curr));

                        workitems.push_back(dataflow(hpx::launch::sync,
                            f2, prev, curr));
                    }
                }
                catch (...) {
                    handle_local_exceptions<ExPolicy>::call(
                        std::current_exception(), errors);
                }

                // wait for all tasks to finish
                hpx::wait_all(workitems, finalitems);

                // always rethrow if 'errors' is not empty or 'workitems' or
                // 'finalitems' have an exceptional future
                handle_local_exceptions<ExPolicy>::call(workitems, errors);
                handle_local_exceptions<ExPolicy>::call(finalitems, errors);

                try {
                    return f4(std::move(workitems), std::move(finalitems));
                }
                catch (...) {
                    // rethrow either bad_alloc or exception_list
                    handle_local_exceptions<ExPolicy>::call(
                        std::current_exception());
                }
            }
        };

        template <typename R, typename Result1, typename Result2>
        struct static_scan_partitioner_helper<R, Result1, Result2,
            scan_partitioner_sequential_f3_tag>
        {
            template <typename ExPolicy, typename FwdIter, typename T,
                typename F1, typename F2, typename F3, typename F4>
            static R call(ExPolicy && policy, FwdIter first,
                std::size_t count, T && init, F1 && f1, F2 && f2, F3 && f3,
                F4 && f4)
            {
                typedef typename
                    hpx::util::decay<ExPolicy>::type::executor_parameters_type
                    parameters_type;
                typedef typename
                    hpx::util::decay<ExPolicy>::type::executor_type
                    executor_type;

                // inform parameter traits
                scoped_executor_parameters_ref<
                        parameters_type, executor_type
                    > scoped_param(policy.parameters(), policy.executor());

                std::vector<hpx::shared_future<Result1> > workitems;
                std::vector<hpx::future<Result2> > finalitems;
                std::list<std::exception_ptr> errors;

                try {
                    // pre-initialize first intermediate result
                    workitems.push_back(make_ready_future(std::forward<T>(init)));

                    HPX_ASSERT(count > 0);
                    FwdIter first_ = first;
                    std::size_t count_ = count;
                    bool tested = false;

                    // estimate a chunk size based on number of cores used
                    typedef typename execution::extract_has_variable_chunk_size<
                            parameters_type
                        >::type has_variable_chunk_size;

                    auto shape = get_bulk_iteration_shape(policy, workitems,
                        f1, first, count, 1, has_variable_chunk_size());

                    // schedule every chunk on a separate thread
                    std::size_t size = hpx::util::size(shape);

                    // If the size of count was enough to warrant testing for a
                    // chunk, pre-initialize second intermediate result.
                    if (workitems.size() == 2)
                    {
                        workitems.reserve(size + 2);
                        finalitems.reserve(size + 1);

                        hpx::shared_future<Result1> curr = workitems[1];
                        workitems[1] = dataflow(hpx::launch::sync,
                            f2, workitems[0], curr);
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
                        FwdIter it = hpx::util::get<0>(elem);
                        std::size_t size = hpx::util::get<1>(elem);

                        hpx::shared_future<Result1> prev = workitems.back();
                        auto curr = execution::async_execute(
                            policy.executor(), f1, it, size).share();

                        workitems.push_back(dataflow(hpx::launch::sync,
                            f2, prev, curr));
                    }

                    // In the code below, performs step 3 sequentially.
                    auto shape_iter = std::begin(shape);

                    // First, perform f3 of the first partition.
                    if (tested)
                    {
                        HPX_ASSERT(count_ > count);

                        finalitems.push_back(dataflow(hpx::launch::sync,
                            f3, first_, count_ - count,
                            workitems[0], workitems[1]));
                    }
                    else
                    {
                        auto elem = *shape_iter++;
                        FwdIter it = hpx::util::get<0>(elem);
                        std::size_t size = hpx::util::get<1>(elem);

                        finalitems.push_back(dataflow(hpx::launch::sync,
                            f3, it, size, workitems[0], workitems[1]));
                    }

                    HPX_ASSERT(finalitems.size() >= 1);

                    // Perform f3 sequentially from the second to the end
                    // of partitions.
                    for (std::size_t widx = 1ul;
                         shape_iter != std::end(shape);
                         ++shape_iter, ++widx)
                    {
                        auto elem = *shape_iter;
                        FwdIter it = hpx::util::get<0>(elem);
                        std::size_t size = hpx::util::get<1>(elem);

                        // Wait the completion of f3 on previous partition.
                        finalitems.back().wait();

                        finalitems.push_back(dataflow(hpx::launch::sync,
                            f3, it, size,
                            workitems[widx], workitems[widx + 1]));
                    }
                }
                catch (...) {
                    handle_local_exceptions<ExPolicy>::call(
                        std::current_exception(), errors);
                }

                // wait for all tasks to finish
                hpx::wait_all(workitems, finalitems);

                // always rethrow if 'errors' is not empty or 'workitems' or
                // 'finalitems' have an exceptional future
                handle_local_exceptions<ExPolicy>::call(workitems, errors);
                handle_local_exceptions<ExPolicy>::call(finalitems, errors);

                try {
                    return f4(std::move(workitems), std::move(finalitems));
                }
                catch (...) {
                    // rethrow either bad_alloc or exception_list
                    handle_local_exceptions<ExPolicy>::call(
                        std::current_exception());
                }
            }
        };

        template <typename ExPolicy_, typename R, typename Result1,
            typename Result2, typename ScanPartTag>
        struct static_scan_partitioner
        {
            template <typename ExPolicy, typename FwdIter, typename T,
                typename F1, typename F2, typename F3, typename F4>
            static R call(ExPolicy && policy, FwdIter first,
                std::size_t count, T && init, F1 && f1, F2 && f2, F3 && f3,
                F4 && f4)
            {
                return static_scan_partitioner_helper<
                        R, Result1, Result2, ScanPartTag
                    >::call(
                        std::forward<ExPolicy>(policy),
                        first, count, std::forward<T>(init),
                        std::forward<F1>(f1), std::forward<F2>(f2),
                        std::forward<F3>(f3), std::forward<F4>(f4));
            }
        };

        template <typename R, typename Result1, typename Result2,
            typename ScanPartTag>
        struct static_scan_partitioner<
            execution::parallel_task_policy, R, Result1, Result2, ScanPartTag>
        {
            template <typename ExPolicy, typename FwdIter, typename T,
                typename F1, typename F2, typename F3, typename F4>
            static hpx::future<R> call(ExPolicy && policy,
                FwdIter first, std::size_t count, T && init, F1 && f1,
                F2 && f2, F3 && f3, F4 && f4)
            {
                hpx::future<R> f = execution::async_execute(
                    policy.executor(),
                    [=]() mutable -> R
                    {
                        return static_scan_partitioner_helper<
                                R, Result1, Result2, ScanPartTag
                            >::call(policy, first, count, init,
                                f1, f2, f3, f4);
                    });

                if (f.has_exception())
                {
                    try {
                        std::rethrow_exception(f.get_exception_ptr());
                    }
                    catch (std::bad_alloc const& ba) {
                        throw ba;
                    }
                    catch (...) {
                        return hpx::make_exceptional_future<R>(std::current_exception());
                    }
                }

                return f;
            }
        };

        template <typename Executor, typename Parameters, typename R,
            typename Result1, typename Result2, typename ScanPartTag>
        struct static_scan_partitioner<
                execution::parallel_task_policy_shim<Executor, Parameters>,
                    R, Result1, Result2, ScanPartTag>
          : static_scan_partitioner<execution::parallel_task_policy, R,
              Result1, Result2, ScanPartTag>
        {};

        ///////////////////////////////////////////////////////////////////////
        // ExPolicy:    execution policy
        // R:           overall result type
        // Result1:     intermediate result type of first and second step
        // Result2:     intermediate result of the third step
        // ScanPartTag: select appropriate policy of scan partitioner
        // PartTag:     select appropriate partitioner
        template <typename ExPolicy, typename R, typename Result1,
            typename Result2, typename ScanPartTag, typename PartTag>
        struct scan_partitioner;

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy_, typename R, typename Result1,
            typename Result2, typename ScanPartTag>
        struct scan_partitioner<ExPolicy_, R, Result1, Result2, ScanPartTag,
            parallel::traits::static_partitioner_tag>
        {
            template <typename ExPolicy, typename FwdIter, typename T,
                typename F1, typename F2, typename F3, typename F4>
            static R call(ExPolicy && policy, FwdIter first,
                std::size_t count, T && init, F1 && f1, F2 && f2, F3 && f3,
                F4 && f4)
            {
                return static_scan_partitioner<
                        typename hpx::util::decay<ExPolicy>::type,
                        R, Result1, Result2, ScanPartTag
                    >::call(
                        std::forward<ExPolicy>(policy),
                        first, count, std::forward<T>(init),
                        std::forward<F1>(f1), std::forward<F2>(f2),
                        std::forward<F3>(f3), std::forward<F4>(f4));
            }
        };

        template <typename R, typename Result1, typename Result2,
            typename ScanPartTag>
        struct scan_partitioner<execution::parallel_task_policy, R, Result1,
            Result2, ScanPartTag, parallel::traits::static_partitioner_tag>
        {
            template <typename ExPolicy, typename FwdIter, typename T,
                typename F1, typename F2, typename F3, typename F4>
            static hpx::future<R> call(ExPolicy && policy, FwdIter first,
                std::size_t count, T && init, F1 && f1, F2 && f2, F3 && f3,
                F4 && f4)
            {
                return static_scan_partitioner<
                        typename hpx::util::decay<ExPolicy>::type,
                        R, Result1, Result2, ScanPartTag
                    >::call(
                        std::forward<ExPolicy>(policy),
                        first, count, std::forward<T>(init),
                        std::forward<F1>(f1), std::forward<F2>(f2),
                        std::forward<F3>(f3), std::forward<F4>(f4));
            }
        };

        template <typename Executor, typename Parameters, typename R,
            typename Result1, typename Result2, typename ScanPartTag>
        struct scan_partitioner<
                execution::parallel_task_policy_shim<Executor, Parameters>,
                R, Result1, Result2, ScanPartTag,
                parallel::traits::static_partitioner_tag>
          : scan_partitioner<execution::parallel_task_policy, R, Result1,
                Result2, ScanPartTag,
                parallel::traits::static_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename R,
            typename Result1, typename Result2, typename ScanPartTag>
        struct scan_partitioner<
                execution::parallel_task_policy_shim<Executor, Parameters>,
                R, Result1, Result2, ScanPartTag,
                parallel::traits::auto_partitioner_tag>
          : scan_partitioner<execution::parallel_task_policy, R, Result1,
                Result2, ScanPartTag,
                parallel::traits::auto_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename R,
            typename Result1, typename Result2, typename ScanPartTag>
        struct scan_partitioner<
                execution::parallel_task_policy_shim<Executor, Parameters>,
                R, Result1, Result2, ScanPartTag,
                parallel::traits::default_partitioner_tag>
          : scan_partitioner<execution::parallel_task_policy, R, Result1,
                Result2, ScanPartTag,
                parallel::traits::static_partitioner_tag>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename R, typename Result1,
            typename Result2, typename ScanPartTag>
        struct scan_partitioner<ExPolicy, R, Result1,
                Result2, ScanPartTag, parallel::traits::default_partitioner_tag>
          : scan_partitioner<ExPolicy, R, Result1,
                Result2, ScanPartTag, parallel::traits::static_partitioner_tag>
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename R = void, typename Result1 = R,
        typename Result2 = void,
        typename ScanPartTag = scan_partitioner_normal_tag,
        typename PartTag = typename parallel::traits::extract_partitioner<
            typename hpx::util::decay<ExPolicy>::type
        >::type>
    struct scan_partitioner
      : detail::scan_partitioner<
            typename hpx::util::decay<ExPolicy>::type, R, Result1,
            Result2, ScanPartTag, PartTag>
    {};
}}}

#endif
