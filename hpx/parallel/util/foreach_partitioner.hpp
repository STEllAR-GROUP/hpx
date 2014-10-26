//  Copyright (c) 2007-2014 Hartmut Kaiser
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
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/chunk_size.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/traits/extract_partitioner.hpp>
#include <hpx/util/decay.hpp>

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
            static FwdIter call(ExPolicy const& policy, FwdIter first,
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

                    threads::executor exec = policy.get_executor();
                    while (count > chunk_size)
                    {
                        if (exec)
                        {
                            workitems.push_back(hpx::async(exec, f1, first,
                                chunk_size));
                        }
                        else
                        {
                            workitems.push_back(hpx::async(hpx::launch::fork,
                                f1, first, chunk_size));
                        }
                        count -= chunk_size;
                        std::advance(first, chunk_size);
                    }

                    // execute last chunk directly
                    if (count != 0)
                    {
                        f1(first, count);
                        std::advance(first, count);
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
            template <typename FwdIter, typename F1>
            static hpx::future<FwdIter> call(
                parallel_task_execution_policy const& policy,
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

                    threads::executor exec = policy.get_executor();
                    while (count > chunk_size)
                    {
                        if (exec)
                        {
                            workitems.push_back(hpx::async(exec, f1, first,
                                chunk_size));
                        }
                        else
                        {
                            workitems.push_back(hpx::async(hpx::launch::fork, f1,
                                first, chunk_size));
                        }
                        count -= chunk_size;
                        std::advance(first, chunk_size);
                    }

                    // add last chunk
                    if (count != 0)
                    {
                        if (exec)
                        {
                            workitems.push_back(hpx::async(exec, f1, first, count));
                        }
                        else
                        {
                            workitems.push_back(hpx::async(hpx::launch::fork, f1,
                                first, count));
                        }
                        std::advance(first, count);
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
                    [first, errors](std::vector<hpx::future<Result> > && r) mutable -> FwdIter
                    {
                        detail::handle_local_exceptions<
                                parallel_task_execution_policy
                            >::call(r, errors);
                        return first;
                    },
                    std::move(workitems));
            }
        };

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
            static FwdIter call(ExPolicy const& policy, FwdIter first,
                std::size_t count, F1 && f1)
            {
                return foreach_n_static_partitioner<ExPolicy, Result>::call(
                    policy, first, count, std::forward<F1>(f1), 0);
            }
        };

        template <typename Result>
        struct foreach_n_partitioner<
            parallel_task_execution_policy, Result,
                parallel::traits::static_partitioner_tag>
        {
            template <typename FwdIter, typename F1>
            static hpx::future<FwdIter> call(
                parallel_task_execution_policy const& policy,
                FwdIter first, std::size_t count, F1 && f1)
            {
                return foreach_n_static_partitioner<
                        parallel_task_execution_policy, Result
                    >::call(policy, first, count, std::forward<F1>(f1), 0);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Result>
        struct foreach_n_partitioner<ExPolicy, Result,
                parallel::traits::default_partitioner_tag>
          : foreach_n_partitioner<ExPolicy, Result,
                parallel::traits::static_partitioner_tag>
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Iter, typename Result = void,
        typename PartTag = typename parallel::traits::extract_partitioner<
            typename hpx::util::decay<ExPolicy>::type, Iter
        >::type>
    struct foreach_n_partitioner
      : detail::foreach_n_partitioner<
            typename hpx::util::decay<ExPolicy>::type, Result, PartTag>
    {};
}}}

#endif
