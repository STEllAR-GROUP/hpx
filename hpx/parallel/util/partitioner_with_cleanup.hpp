//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_PARTITIONER_WITH_CLEANUP_OCT_03_2014_0221PM)
#define HPX_PARALLEL_UTIL_PARTITIONER_WITH_CLEANUP_OCT_03_2014_0221PM

#include <hpx/config.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/deferred_call.hpp>

#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/executors/executor_parameter_traits.hpp>
#include <hpx/parallel/executors/execution.hpp>
#include <hpx/parallel/traits/extract_partitioner.hpp>
#include <hpx/parallel/util/detail/chunk_size.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/util/detail/partitioner_iteration.hpp>
#include <hpx/parallel/util/detail/scoped_executor_parameters.hpp>
#include <hpx/util/unused.hpp>

#include <boost/exception_ptr.hpp>

#include <algorithm>
#include <cstddef>
#include <list>
#include <memory>
#include <utility>
#include <vector>

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
        template <typename ExPolicy_, typename R, typename Result = void>
        struct static_partitioner_with_cleanup
        {
            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2, typename F3>
            static R call(ExPolicy && policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2, F3 && f3)
            {
                typedef typename
                    hpx::util::decay<ExPolicy>::type::executor_parameters_type
                    parameters_type;
                typedef executor_parameter_traits<parameters_type>
                    parameters_traits;

                // inform parameter traits
                scoped_executor_parameters<parameters_type> scoped_param(
                    policy.parameters());

                std::vector<hpx::future<Result> > inititems;
                std::list<boost::exception_ptr> errors;

                try {
                    // estimate a chunk size based on number of cores used
                    typedef typename parameters_traits::has_variable_chunk_size
                        has_variable_chunk_size;

                    auto shapes =
                        get_bulk_iteration_shape(policy, inititems, f1,
                            first, count, 1, has_variable_chunk_size());

                    std::vector<hpx::future<Result> > workitems =
                        execution::async_bulk_execute(
                            policy.executor(),
                            partitioner_iteration<Result, F1>{std::forward<F1>(f1)},
                            std::move(shapes));

                    inititems.reserve(inititems.size() + workitems.size());
                    std::move(workitems.begin(), workitems.end(),
                        std::back_inserter(inititems));
                }
                catch (...) {
                    handle_local_exceptions<ExPolicy>::call(
                        boost::current_exception(), errors);
                }

                // wait for all tasks to finish
                hpx::wait_all(inititems);

                // always rethrow if 'errors' is not empty or inititems has
                // exceptional future
                handle_local_exceptions<ExPolicy>::call(
                    inititems, errors, std::forward<F3>(f3));

                try {
                    return f2(std::move(inititems));
                }
                catch (...) {
                    // rethrow either bad_alloc or exception_list
                    handle_local_exceptions<ExPolicy>::call(
                        boost::current_exception());
                }
            }
        };

        template <typename R, typename Result>
        struct static_partitioner_with_cleanup<execution::parallel_task_policy,
            R, Result>
        {
            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2, typename F3>
            static hpx::future<R> call(ExPolicy && policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2, F3 && f3)
            {
                typedef typename
                    hpx::util::decay<ExPolicy>::type::executor_parameters_type
                    parameters_type;
                typedef executor_parameter_traits<parameters_type>
                    parameters_traits;

                typedef scoped_executor_parameters<parameters_type>
                    scoped_executor_parameters;

                // inform parameter traits
                std::shared_ptr<scoped_executor_parameters>
                    scoped_param(std::make_shared<
                            scoped_executor_parameters
                        >(policy.parameters()));

                std::vector<hpx::future<Result> > inititems;
                std::list<boost::exception_ptr> errors;

                try {
                    // estimate a chunk size based on number of cores used
                    typedef typename parameters_traits::has_variable_chunk_size
                        has_variable_chunk_size;

                    auto shapes =
                        get_bulk_iteration_shape(policy, inititems, f1,
                            first, count, 1, has_variable_chunk_size());

                    std::vector<hpx::future<Result> > workitems =
                        execution::async_bulk_execute(
                            policy.executor(),
                            partitioner_iteration<Result, F1>{std::forward<F1>(f1)},
                            std::move(shapes));

                    inititems.reserve(inititems.size() + workitems.size());
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
                return hpx::dataflow(
                    [f2, f3, errors, scoped_param](
                        std::vector<hpx::future<Result> > && r) mutable -> R
                    {
                        HPX_UNUSED(scoped_param);

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
                execution::parallel_task_policy_shim<Executor, Parameters>,
                R, Result>
          : static_partitioner_with_cleanup<
              execution::parallel_task_policy, R, Result>
        {};

        ///////////////////////////////////////////////////////////////////////
        // ExPolicy: execution policy
        // R:        overall result type
        // Result:   intermediate result type of first step
        // PartTag:  select appropriate partitioner
        template <typename ExPolicy, typename R, typename Result, typename Tag>
        struct partitioner_with_cleanup;

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy_, typename R, typename Result>
        struct partitioner_with_cleanup<ExPolicy_, R, Result,
            parallel::traits::static_partitioner_tag>
        {
            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2, typename F3>
            static R call(ExPolicy && policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2, F3 && f3)
            {
                return static_partitioner_with_cleanup<
                        typename hpx::util::decay<ExPolicy>::type, R, Result
                    >::call(
                        std::forward<ExPolicy>(policy), first, count,
                        std::forward<F1>(f1), std::forward<F2>(f2),
                        std::forward<F3>(f3));
            }
        };

        template <typename R, typename Result>
        struct partitioner_with_cleanup<execution::parallel_task_policy, R,
            Result, parallel::traits::static_partitioner_tag>
        {
            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2, typename F3>
            static hpx::future<R> call(ExPolicy && policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2, F3 && f3)
            {
                return static_partitioner_with_cleanup<
                        typename hpx::util::decay<ExPolicy>::type, R, Result
                    >::call(std::forward<ExPolicy>(policy), first, count,
                        std::forward<F1>(f1), std::forward<F2>(f2),
                        std::forward<F3>(f3));
            }
        };

        template <typename Executor, typename Parameters, typename R,
            typename Result>
        struct partitioner_with_cleanup<
                execution::parallel_task_policy_shim<Executor, Parameters>,
                R, Result, parallel::traits::static_partitioner_tag>
          : partitioner_with_cleanup<execution::parallel_task_policy, R, Result,
                parallel::traits::static_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename R,
            typename Result>
        struct partitioner_with_cleanup<
                execution::parallel_task_policy_shim<Executor, Parameters>,
                R, Result, parallel::traits::auto_partitioner_tag>
          : partitioner_with_cleanup<execution::parallel_task_policy, R, Result,
                parallel::traits::auto_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename R,
            typename Result>
        struct partitioner_with_cleanup<
                execution::parallel_task_policy_shim<Executor, Parameters>,
                R, Result, parallel::traits::default_partitioner_tag>
          : partitioner_with_cleanup<execution::parallel_task_policy, R, Result,
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
