//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_FOREACH_PARTITIONER_OCT_03_2014_0112PM)
#define HPX_PARALLEL_UTIL_FOREACH_PARTITIONER_OCT_03_2014_0112PM

#include <hpx/config.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/tuple.hpp>

#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/executors/executor_parameter_traits.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/traits/extract_partitioner.hpp>
#include <hpx/parallel/util/detail/chunk_size.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/util/detail/scoped_executor_parameters.hpp>

#include <boost/exception_ptr.hpp>

#include <algorithm>
#include <list>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // The static partitioner simply spawns one chunk of iterations for
        // each available core.
        template <typename ExPolicy_, typename Result = void>
        struct foreach_static_partitioner
        {
            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2>
            static FwdIter call(ExPolicy && policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2)
            {
                typedef typename hpx::util::decay<ExPolicy>::type::executor_type
                    executor_type;
                typedef hpx::parallel::executor_traits<executor_type>
                    executor_traits;

                typedef typename
                    hpx::util::decay<ExPolicy>::type::executor_parameters_type
                    parameters_type;

                typedef hpx::util::tuple<
                        std::size_t, FwdIter, std::size_t
                    > tuple_type;

                // inform parameter traits
                scoped_executor_parameters<parameters_type> scoped_param(
                    policy.parameters());

                FwdIter last = parallel::v1::detail::next(first, count);

                std::vector<hpx::future<Result> > inititems, workitems;
                std::list<boost::exception_ptr> errors;

                try {
                    // estimates a chunk size based on number of cores used
                    std::vector<tuple_type> shape =
                        get_bulk_iteration_shape_idx(
                            policy, inititems, f1, first, count, 1);

                    workitems.reserve(shape.size());

                    using hpx::util::functional::invoke_fused;
                    using hpx::util::placeholders::_1;
                    workitems = executor_traits::bulk_async_execute(
                        policy.executor(),
                        hpx::util::bind(invoke_fused(), std::forward<F1>(f1), _1),
                        std::move(shape));
                }
                catch (...) {
                    handle_local_exceptions<ExPolicy>::call(
                        boost::current_exception(), errors);
                }

                // wait for all tasks to finish
                hpx::wait_all(inititems, workitems);

                // handle exceptions
                handle_local_exceptions<ExPolicy>::call(inititems, errors);
                handle_local_exceptions<ExPolicy>::call(workitems, errors);

                try {
                    return f2(std::move(last));
                }
                catch (...) {
                    // rethrow either bad_alloc or exception_list
                    handle_local_exceptions<ExPolicy>::call(
                        boost::current_exception());
                }
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        struct foreach_static_partitioner<parallel_task_execution_policy, Result>
        {
            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2>
            static hpx::future<FwdIter> call(ExPolicy && policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2)
            {
                typedef typename hpx::util::decay<ExPolicy>::type::executor_type
                    executor_type;
                typedef hpx::parallel::executor_traits<executor_type>
                    executor_traits;

                typedef typename
                    hpx::util::decay<ExPolicy>::type::executor_parameters_type
                    parameters_type;
                typedef scoped_executor_parameters<parameters_type>
                    scoped_executor_parameters;

                typedef hpx::util::tuple<
                        std::size_t, FwdIter, std::size_t
                    > tuple_type;

                // inform parameter traits
                std::shared_ptr<scoped_executor_parameters>
                    scoped_param(std::make_shared<
                            scoped_executor_parameters
                        >(policy.parameters()));

                FwdIter last = parallel::v1::detail::next(first, count);

                std::vector<hpx::future<Result> > inititems, workitems;
                std::list<boost::exception_ptr> errors;

                try {
                    // estimates a chunk size based on number of cores used
                    std::vector<tuple_type> shape =
                        get_bulk_iteration_shape_idx(
                            policy, inititems, f1, first, count, 1);

                    workitems.reserve(shape.size());

                    using hpx::util::bind;
                    using hpx::util::functional::invoke_fused;
                    using hpx::util::placeholders::_1;
                    workitems = executor_traits::bulk_async_execute(
                        policy.executor(),
                        bind(invoke_fused(), std::forward<F1>(f1), _1),
                        std::move(shape));
                }
                catch (std::bad_alloc const&) {
                    return hpx::make_exceptional_future<FwdIter>(
                        boost::current_exception());
                }
                catch (...) {
                    errors.push_back(boost::current_exception());
                }

                // wait for all tasks to finish
                return hpx::dataflow(
                    [last, errors, scoped_param, f2](
                            std::vector<hpx::future<Result> > && r1,
                            std::vector<hpx::future<Result> > && r2) mutable
                    ->  FwdIter
                    {
                        handle_local_exceptions<ExPolicy>::call(r1, errors);
                        handle_local_exceptions<ExPolicy>::call(r2, errors);

                        return f2(std::move(last));
                    },
                    std::move(inititems), std::move(workitems));
            }
        };

        template <typename Executor, typename Parameters, typename Result>
        struct foreach_static_partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                Result>
          : foreach_static_partitioner<parallel_task_execution_policy, Result>
        {};

        ///////////////////////////////////////////////////////////////////////
        // ExPolicy: execution policy
        // Result:   intermediate result type of first step (default: void)
        // PartTag:  select appropriate partitioner
        template <typename ExPolicy, typename Result, typename PartTag>
        struct foreach_partitioner;

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy_, typename Result>
        struct foreach_partitioner<ExPolicy_, Result,
            parallel::traits::static_partitioner_tag>
        {
            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2>
            static FwdIter call(ExPolicy && policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2)
            {
                return foreach_static_partitioner<
                        typename hpx::util::decay<ExPolicy>::type, Result
                    >::call(
                        std::forward<ExPolicy>(policy), first, count,
                        std::forward<F1>(f1), std::forward<F2>(f2));
            }
        };

        template <typename Result>
        struct foreach_partitioner<parallel_task_execution_policy, Result,
                parallel::traits::static_partitioner_tag>
        {
            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2>
            static hpx::future<FwdIter> call(ExPolicy && policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2)
            {
                return foreach_static_partitioner<
                        typename hpx::util::decay<ExPolicy>::type, Result
                    >::call(
                        std::forward<ExPolicy>(policy), first, count,
                        std::forward<F1>(f1), std::forward<F2>(f2));
            }
        };

        template <typename Executor, typename Parameters, typename Result>
        struct foreach_partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                Result, parallel::traits::static_partitioner_tag>
          : foreach_partitioner<parallel_task_execution_policy, Result,
                parallel::traits::static_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename Result>
        struct foreach_partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                Result, parallel::traits::auto_partitioner_tag>
          : foreach_partitioner<parallel_task_execution_policy, Result,
                parallel::traits::auto_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename Result>
        struct foreach_partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                Result, parallel::traits::default_partitioner_tag>
          : foreach_partitioner<parallel_task_execution_policy, Result,
                parallel::traits::static_partitioner_tag>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Result>
        struct foreach_partitioner<ExPolicy, Result,
                parallel::traits::default_partitioner_tag>
          : foreach_partitioner<ExPolicy, Result,
                parallel::traits::static_partitioner_tag>
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Result = void,
        typename PartTag = typename parallel::traits::extract_partitioner<
            typename hpx::util::decay<ExPolicy>::type
        >::type>
    struct foreach_partitioner
      : detail::foreach_partitioner<
            typename hpx::util::decay<ExPolicy>::type, Result, PartTag>
    {};
}}}

#endif
