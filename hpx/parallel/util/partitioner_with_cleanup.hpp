//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_PARTITIONER_WITH_CLEANUP_OCT_03_2014_0221PM)
#define HPX_PARALLEL_UTIL_PARTITIONER_WITH_CLEANUP_OCT_03_2014_0221PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/async.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/lcos/local/dataflow.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/deferred_call.hpp>

#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/chunk_size.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/traits/extract_partitioner.hpp>

#include <algorithm>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // The static partitioner with cleanup spawns several chunks of
        // iterations for each available core. The number of iterations is
        // determined automatically based on the measured runtime of the
        // iterations.
        template <typename ExPolicy, typename R, typename Result = void>
        struct static_partitioner_with_cleanup
        {
            template <typename FwdIter, typename F1, typename F2, typename F3>
            static R call(ExPolicy policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2, F3 && f3,
                std::size_t chunk_size)
            {
                typedef typename ExPolicy::executor_type executor_type;
                typedef typename hpx::parallel::executor_traits<executor_type>
                    executor_traits;
                typedef typename hpx::util::tuple<FwdIter, std::size_t>
                    tuple_type;

                std::vector<hpx::future<Result> > inititems;
                std::list<boost::exception_ptr> errors;
                std::vector<tuple_type> shape;

                try {
                    // estimate a chunk size based on number of cores used
                    shape = get_bulk_iteration_shape(policy, inititems, f1,
                        first, count, chunk_size);

                    std::vector<hpx::future<Result> > workitems;
                    workitems.reserve(shape.size());

                    using hpx::util::bind;
                    using hpx::util::functional::invoke_fused;
                    using hpx::util::placeholders::_1;
                    workitems = executor_traits::async_execute(
                        policy.executor(),
                        bind(invoke_fused(), std::forward<F1>(f1), _1),
                        shape);

                    std::move(workitems.begin(), workitems.end(),
                        std::back_inserter(inititems));
                }
                catch (...) {
                    detail::handle_local_exceptions<ExPolicy>::call(
                        boost::current_exception(), errors);
                }

                // wait for all tasks to finish
                hpx::wait_all(inititems);
                detail::handle_local_exceptions<ExPolicy>::call(
                    inititems, errors, std::forward<F3>(f3));

                return f2(std::move(inititems));
            }

            template <typename FwdIter, typename F1, typename F2, typename F3>
            static R call_with_index(ExPolicy policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2, F3 && f3,
                std::size_t chunk_size)
            {
                typedef typename ExPolicy::executor_type executor_type;
                typedef typename hpx::parallel::executor_traits<executor_type>
                    executor_traits;
                typedef hpx::util::tuple<std::size_t, FwdIter, std::size_t>
                    tuple_type;

                std::vector<hpx::future<Result> > inititems;
                std::list<boost::exception_ptr> errors;
                std::vector<tuple_type> shape;

                try {
                    // estimate a chunk size based on number of cores used
                    std::size_t base_idx = 0;
                    shape = get_bulk_iteration_shape_idx(policy, inititems, f1,
                        base_idx, first, count, chunk_size);

                    std::vector<hpx::future<Result> > workitems;
                    workitems.reserve(shape.size());

                    using hpx::util::bind;
                    using hpx::util::functional::invoke_fused;
                    using hpx::util::placeholders::_1;
                    workitems = executor_traits::async_execute(
                        policy.executor(),
                        bind(invoke_fused(), std::forward<F1>(f1), _1),
                        shape);

                    std::move(workitems.begin(), workitems.end(),
                        std::back_inserter(inititems));
                }
                catch (...) {
                    detail::handle_local_exceptions<ExPolicy>::call(
                        boost::current_exception(), errors);
                }

                // wait for all tasks to finish
                hpx::wait_all(inititems);
                detail::handle_local_exceptions<ExPolicy>::call(
                    inititems, errors, std::forward<F3>(f3));

                return f2(std::move(inititems));
            }
        };

        template <typename R, typename Result>
        struct static_partitioner_with_cleanup<parallel_task_execution_policy,
            R, Result>
        {
            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2, typename F3>
            static hpx::future<R> call(ExPolicy policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2, F3 && f3,
                std::size_t chunk_size)
            {
                typedef typename ExPolicy::executor_type executor_type;
                typedef typename hpx::parallel::executor_traits<executor_type>
                    executor_traits;
                typedef typename hpx::util::tuple<FwdIter, std::size_t>
                    tuple_type;

                std::vector<hpx::future<Result> > inititems;
                std::list<boost::exception_ptr> errors;
                std::vector<tuple_type> shape;

                try {
                    // estimate a chunk size based on number of cores used
                    shape = get_bulk_iteration_shape(policy, inititems, f1,
                        first, count, chunk_size);

                    std::vector<hpx::future<Result> > workitems;
                    workitems.reserve(shape.size());

                    using hpx::util::bind;
                    using hpx::util::functional::invoke_fused;
                    using hpx::util::placeholders::_1;
                    workitems = executor_traits::async_execute(
                        policy.executor(),
                        bind(invoke_fused(), std::forward<F1>(f1), _1),
                        shape);

                    std::move(workitems.begin(), workitems.end(),
                        std::back_inserter(inititems));
                }
                catch (std::bad_alloc const&) {
                    return hpx::make_exceptional_future<R>(
                        boost::current_exception());
                }
                catch (...) {
                    errors.push_back(boost::current_exception());
                }

                // wait for all tasks to finish
                return hpx::lcos::local::dataflow(
                    [f2, f3, errors](
                        std::vector<hpx::future<Result> > && r) mutable -> R
                    {
                        detail::handle_local_exceptions<ExPolicy>::call(
                            r, errors, std::forward<F3>(f3));
                        return f2(std::move(r));
                    },
                    std::move(inititems));
            }

            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2, typename F3>
            static hpx::future<R> call_with_index(ExPolicy policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2, F3 && f3,
                std::size_t chunk_size)
            {
                typedef typename ExPolicy::executor_type executor_type;
                typedef typename hpx::parallel::executor_traits<executor_type>
                    executor_traits;
                typedef hpx::util::tuple<std::size_t, FwdIter, std::size_t>
                    tuple_type;

                std::vector<hpx::future<Result> > inititems;
                std::list<boost::exception_ptr> errors;
                std::vector<tuple_type> shape;

                try {
                    // estimate a chunk size based on number of cores used
                    std::size_t base_idx = 0;
                    shape = get_bulk_iteration_shape_idx(policy, inititems, f1,
                        base_idx, first, count, chunk_size);

                    std::vector<hpx::future<Result> > workitems;
                    workitems.reserve(shape.size());

                    using hpx::util::bind;
                    using hpx::util::functional::invoke_fused;
                    using hpx::util::placeholders::_1;
                    workitems = executor_traits::async_execute(
                        policy.executor(),
                        bind(invoke_fused(), std::forward<F1>(f1), _1),
                        shape);

                    std::move(workitems.begin(), workitems.end(),
                        std::back_inserter(inititems));
                }
                catch (std::bad_alloc const&) {
                    return hpx::make_exceptional_future<R>(
                        boost::current_exception());
                }
                catch (...) {
                    errors.push_back(boost::current_exception());
                }

                // wait for all tasks to finish
                return hpx::lcos::local::dataflow(
                    [f2, f3, errors](
                        std::vector<hpx::future<Result> > && r) mutable -> R
                    {
                        detail::handle_local_exceptions<ExPolicy>::call(
                            r, errors, std::forward<F3>(f3));
                        return f2(std::move(r));
                    },
                    std::move(inititems));
            }
        };

        template <typename Executor, typename Parameters, typename R,
            typename Result>
        struct static_partitioner_with_cleanup<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                R, Result>
          : static_partitioner_with_cleanup<
              parallel_task_execution_policy, R, Result>
        {};

        ///////////////////////////////////////////////////////////////////////
        // ExPolicy: execution policy
        // R:        overall result type
        // Result:   intermediate result type of first step
        // PartTag:  select appropriate partitioner
        template <typename ExPolicy, typename R, typename Result, typename Tag>
        struct partitioner_with_cleanup;

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename R, typename Result>
        struct partitioner_with_cleanup<ExPolicy, R, Result,
            parallel::traits::static_partitioner_tag>
        {
            template <typename FwdIter, typename F1, typename F2, typename F3>
            static R call(ExPolicy policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2, F3 && f3)
            {
                return static_partitioner_with_cleanup<ExPolicy, R, Result>::
                    call(
                        policy, first, count,
                        std::forward<F1>(f1), std::forward<F2>(f2),
                        std::forward<F3>(f3), 0);
            }

            template <typename FwdIter, typename F1, typename F2, typename F3>
            static R call_with_index(ExPolicy policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2, F3 && f3)
            {
                return static_partitioner_with_cleanup<ExPolicy, R, Result>::
                    call_with_index(
                        policy, first, count,
                        std::forward<F1>(f1), std::forward<F2>(f2),
                        std::forward<F3>(f3), 0);
            }
        };

        template <typename R, typename Result>
        struct partitioner_with_cleanup<parallel_task_execution_policy, R,
            Result, parallel::traits::static_partitioner_tag>
        {
            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2, typename F3>
            static hpx::future<R> call(ExPolicy policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2, F3 && f3)
            {
                return static_partitioner_with_cleanup<ExPolicy, R, Result>::
                    call(policy, first, count,
                        std::forward<F1>(f1), std::forward<F2>(f2),
                        std::forward<F3>(f3), 0);
            }

            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2, typename F3>
            static hpx::future<R> call_with_index(ExPolicy policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2, F3 && f3)
            {
                return static_partitioner_with_cleanup<ExPolicy, R, Result>::
                    call_with_index(policy, first, count,
                        std::forward<F1>(f1), std::forward<F2>(f2),
                        std::forward<F3>(f3), 0);
            }
        };

        template <typename Executor, typename Parameters, typename R,
            typename Result>
        struct partitioner_with_cleanup<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                R, Result, parallel::traits::static_partitioner_tag>
          : partitioner_with_cleanup<parallel_task_execution_policy, R, Result,
                parallel::traits::static_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename R,
            typename Result>
        struct partitioner_with_cleanup<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                R, Result, parallel::traits::auto_partitioner_tag>
          : partitioner_with_cleanup<parallel_task_execution_policy, R, Result,
                parallel::traits::auto_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename R,
            typename Result>
        struct partitioner_with_cleanup<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                R, Result, parallel::traits::default_partitioner_tag>
          : partitioner_with_cleanup<parallel_task_execution_policy, R, Result,
                parallel::traits::static_partitioner_tag>
        {};


        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename R, typename Result>
        struct partitioner_with_cleanup<ExPolicy, R, Result,
                parallel::traits::default_partitioner_tag>
          : partitioner_with_cleanup<ExPolicy, R, Result,
                parallel::traits::static_partitioner_tag>
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename R = void, typename Result = R,
        typename PartTag = typename parallel::traits::extract_partitioner<
            typename hpx::util::decay<ExPolicy>::type
        >::type>
    struct partitioner_with_cleanup
      : detail::partitioner_with_cleanup<
            typename hpx::util::decay<ExPolicy>::type, R, Result, PartTag>
    {};
}}}

#endif
