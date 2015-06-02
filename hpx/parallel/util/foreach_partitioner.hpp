//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_FOREACH_PARTITIONER_OCT_03_2014_0112PM)
#define HPX_PARALLEL_UTIL_FOREACH_PARTITIONER_OCT_03_2014_0112PM

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
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // The static partitioner simply spawns one chunk of iterations for
        // each available core.
        template <typename ExPolicy, typename Result = void>
        struct foreach_n_static_partitioner
        {
            template <typename FwdIter, typename F1>
            static FwdIter call(ExPolicy policy, FwdIter first,
                std::size_t count, F1 && f1, std::size_t chunk_size)
            {
                std::vector<hpx::future<Result> > workitems;
                std::list<boost::exception_ptr> errors;

                try {
                    // estimate a chunk size based on number of cores used
                    chunk_size = get_static_chunk_size(policy, workitems, f1,
                        first, count, chunk_size);

                    // schedule every chunk on a separate thread
                    workitems.reserve(count / chunk_size + 1);
                    while (count != 0)
                    {
                        std::size_t chunk = (std::min)(chunk_size, count);

                        typedef typename ExPolicy::executor_type executor_type;
                        workitems.push_back(
                            executor_traits<executor_type>::async_execute(
                                policy.executor(),
                                hpx::util::deferred_call(f1, first, chunk)
                            )
                        );

                        count -= chunk;
                        std::advance(first, chunk);
                    }
                }
                catch (...) {
                    detail::handle_local_exceptions<ExPolicy>::call(
                        boost::current_exception(), errors);
                }

                // wait for all tasks to finish
                hpx::wait_all(workitems);
                detail::handle_local_exceptions<ExPolicy>::call(
                    workitems, errors);

                return first;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        struct foreach_n_static_partitioner<parallel_task_execution_policy, Result>
        {
            template <typename ExPolicy, typename FwdIter, typename F1>
            static hpx::future<FwdIter> call(ExPolicy policy,
                FwdIter first, std::size_t count, F1 && f1,
                std::size_t chunk_size)
            {
                std::vector<hpx::future<Result> > workitems;
                std::list<boost::exception_ptr> errors;

                try {
                    // estimate a chunk size based on number of cores used
                    chunk_size = get_static_chunk_size(policy, workitems, f1,
                        first, count, chunk_size);

                    // schedule every chunk on a separate thread
                    workitems.reserve(count / chunk_size + 1);
                    while (count != 0)
                    {
                        std::size_t chunk = (std::min)(chunk_size, count);

                        typedef typename ExPolicy::executor_type executor_type;
                        workitems.push_back(
                            executor_traits<executor_type>::async_execute(
                                policy.executor(),
                                hpx::util::deferred_call(f1, first, chunk)
                            )
                        );

                        count -= chunk;
                        std::advance(first, chunk);
                    }
                }
                catch (std::bad_alloc const&) {
                    return hpx::make_exceptional_future<FwdIter>(
                        boost::current_exception());
                }
                catch (...) {
                    errors.push_back(boost::current_exception());
                }

                // wait for all tasks to finish
                return hpx::lcos::local::dataflow(
                    [first, errors](std::vector<hpx::future<Result> > && r)
                        mutable -> FwdIter
                    {
                        detail::handle_local_exceptions<ExPolicy>::call(r, errors);
                        return first;
                    },
                    std::move(workitems));
            }
        };

        template <typename Executor, typename Result>
        struct foreach_n_static_partitioner<
                parallel_task_execution_policy_shim<Executor>, Result>
          : foreach_n_static_partitioner<parallel_task_execution_policy, Result>
        {};

        ///////////////////////////////////////////////////////////////////////
        // ExPolicy: execution policy
        // Result:   intermediate result type of first step (default: void)
        // PartTag:  select appropriate partitioner
        template <typename ExPolicy, typename Result, typename PartTag>
        struct foreach_n_partitioner;

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Result>
        struct foreach_n_partitioner<ExPolicy, Result,
            parallel::traits::static_partitioner_tag>
        {
            template <typename FwdIter, typename F1>
            static FwdIter call(ExPolicy policy, FwdIter first,
                std::size_t count, F1 && f1, std::size_t chunk_size = 0)
            {
                return foreach_n_static_partitioner<ExPolicy, Result>::call(
                    policy, first, count, std::forward<F1>(f1), chunk_size);
            }
        };

        template <typename Result>
        struct foreach_n_partitioner<parallel_task_execution_policy, Result,
                parallel::traits::static_partitioner_tag>
        {
            template <typename ExPolicy, typename FwdIter, typename F1>
            static hpx::future<FwdIter> call(ExPolicy policy,
                FwdIter first, std::size_t count, F1 && f1,
                std::size_t chunk_size = 0)
            {
                return foreach_n_static_partitioner<ExPolicy, Result>::call(
                    policy, first, count, std::forward<F1>(f1), chunk_size);
            }
        };

        template <typename Executor, typename Result>
        struct foreach_n_partitioner<
                parallel_task_execution_policy_shim<Executor>, Result,
                parallel::traits::static_partitioner_tag>
          : foreach_n_partitioner<parallel_task_execution_policy, Result,
                parallel::traits::static_partitioner_tag>
        {};

        template <typename Executor, typename Result>
        struct foreach_n_partitioner<
                parallel_task_execution_policy_shim<Executor>, Result,
                parallel::traits::auto_partitioner_tag>
          : foreach_n_partitioner<parallel_task_execution_policy, Result,
                parallel::traits::auto_partitioner_tag>
        {};

        template <typename Executor, typename Result>
        struct foreach_n_partitioner<
                parallel_task_execution_policy_shim<Executor>, Result,
                parallel::traits::default_partitioner_tag>
          : foreach_n_partitioner<parallel_task_execution_policy, Result,
                parallel::traits::static_partitioner_tag>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Result>
        struct foreach_n_partitioner<ExPolicy, Result,
                parallel::traits::default_partitioner_tag>
          : foreach_n_partitioner<ExPolicy, Result,
                parallel::traits::static_partitioner_tag>
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Result = void,
        typename PartTag = typename parallel::traits::extract_partitioner<
            typename hpx::util::decay<ExPolicy>::type
        >::type>
    struct foreach_n_partitioner
      : detail::foreach_n_partitioner<
            typename hpx::util::decay<ExPolicy>::type, Result, PartTag>
    {};
}}}

#endif
