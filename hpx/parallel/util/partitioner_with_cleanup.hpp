//  Copyright (c) 2007-2014 Hartmut Kaiser
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

#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/chunk_size.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/traits/extract_partitioner.hpp>

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
            static R call(ExPolicy const& policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2, F3 && f3,
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
                            workitems.push_back(hpx::async(hpx::launch::fork,
                                f1, first, chunk_size));
                        }
                        count -= chunk_size;
                        std::advance(first, chunk_size);
                    }

                    // execute last chunk directly
                    if (count != 0)
                    {
                        workitems.push_back(hpx::async(hpx::launch::sync,
                            std::forward<F1>(f1), first, count));
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
                    workitems, errors, std::forward<F3>(f3));

                return f2(std::move(workitems));
            }

            template <typename FwdIter, typename F1, typename F2, typename F3>
            static R call_with_index(ExPolicy const& policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2, F3 && f3,
                std::size_t chunk_size)
            {
                std::vector<hpx::future<Result> > workitems;
                std::list<boost::exception_ptr> errors;

                try {
                    // estimate a chunk size based on number of cores used
                    std::size_t base_idx = 0;
                    chunk_size = get_static_chunk_size_idx(policy, workitems,
                        f1, base_idx, first, count, chunk_size);

                    // schedule every chunk on a separate thread
                    workitems.reserve(count / chunk_size + 1);

                    threads::executor exec = policy.get_executor();
                    while (count > chunk_size)
                    {
                        if (exec)
                        {
                            workitems.push_back(hpx::async(exec, f1, base_idx,
                                first, chunk_size));
                        }
                        else
                        {
                            workitems.push_back(hpx::async(hpx::launch::fork,
                                f1, base_idx, first, chunk_size));
                        }
                        count -= chunk_size;
                        std::advance(first, chunk_size);
                        base_idx += chunk_size;
                    }

                    // execute last chunk directly
                    if (count != 0)
                    {
                        workitems.push_back(hpx::async(hpx::launch::sync,
                            std::forward<F1>(f1), base_idx, first, count));
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
                    workitems, errors, std::forward<F3>(f3));

                return f2(std::move(workitems));
            }
        };

        template <typename R, typename Result>
        struct static_partitioner_with_cleanup<
            parallel_task_execution_policy, R, Result>
        {
            template <typename FwdIter, typename F1, typename F2, typename F3>
            static hpx::future<R> call(
                parallel_task_execution_policy const& policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2, F3 && f3,
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
                            workitems.push_back(hpx::async(hpx::launch::fork,
                                f1, first, chunk_size));
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
                            workitems.push_back(hpx::async(hpx::launch::fork,
                                f1, first, count));
                        }
                        std::advance(first, count);
                    }
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
                        detail::handle_local_exceptions<
                                parallel_task_execution_policy
                            >::call(r, errors, std::forward<F3>(f3));
                        return f2(std::move(r));
                    },
                    std::move(workitems));
            }

            template <typename FwdIter, typename F1, typename F2, typename F3>
            static hpx::future<R> call_with_index(
                parallel_task_execution_policy const& policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2, F3 && f3,
                std::size_t chunk_size)
            {
                std::vector<hpx::future<Result> > workitems;
                std::list<boost::exception_ptr> errors;

                try {
                    // estimate a chunk size based on number of cores used
                    std::size_t base_idx = 0;
                    chunk_size = get_static_chunk_size_idx(policy, workitems,
                        f1, base_idx, first, count, chunk_size);

                    // schedule every chunk on a separate thread
                    workitems.reserve(count / chunk_size + 1);

                    threads::executor exec = policy.get_executor();
                    while (count > chunk_size)
                    {
                        if (exec)
                        {
                            workitems.push_back(hpx::async(exec, f1, base_idx,
                                first, chunk_size));
                        }
                        else
                        {
                            workitems.push_back(hpx::async(hpx::launch::fork,
                                f1, base_idx, first, chunk_size));
                        }
                        count -= chunk_size;
                        std::advance(first, chunk_size);
                        base_idx += chunk_size;
                    }

                    // add last chunk
                    if (count != 0)
                    {
                        if (exec)
                        {
                            workitems.push_back(hpx::async(exec, f1, base_idx,
                                first, count));
                        }
                        else
                        {
                            workitems.push_back(hpx::async(hpx::launch::fork,
                                f1, base_idx, first, count));
                        }
                        std::advance(first, count);
                    }
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
                        detail::handle_local_exceptions<
                                parallel_task_execution_policy
                            >::call(r, errors, std::forward<F3>(f3));
                        return f2(std::move(r));
                    },
                    std::move(workitems));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // ExPolicy: execution policy
        // R:        overall result type
        // Result:   intermediate result type of first step
        // PartTag:  select appropriate partitioner
        template <typename ExPolicy, typename R, typename Result, typename PartTag>
        struct partitioner_with_cleanup;

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename R, typename Result>
        struct partitioner_with_cleanup<ExPolicy, R, Result,
            parallel::traits::static_partitioner_tag>
        {
            template <typename FwdIter, typename F1, typename F2, typename F3>
            static R call(ExPolicy const& policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2, F3 && f3)
            {
                return static_partitioner_with_cleanup<ExPolicy, R, Result>::
                    call(
                        policy, first, count,
                        std::forward<F1>(f1), std::forward<F2>(f2),
                        std::forward<F3>(f3), 0);
            }

            template <typename FwdIter, typename F1, typename F2, typename F3>
            static R call_with_index(ExPolicy const& policy, FwdIter first,
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
        struct partitioner_with_cleanup<parallel_task_execution_policy, R, Result,
            parallel::traits::static_partitioner_tag>
        {
            template <typename FwdIter, typename F1, typename F2, typename F3>
            static hpx::future<R> call(
                parallel_task_execution_policy const& policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2, F3 && f3)
            {
                return static_partitioner_with_cleanup<
                        parallel_task_execution_policy, R, Result
                    >::call(policy, first, count,
                        std::forward<F1>(f1), std::forward<F2>(f2),
                        std::forward<F3>(f3), 0);
            }

            template <typename FwdIter, typename F1, typename F2, typename F3>
            static hpx::future<R> call_with_index(
                parallel_task_execution_policy const& policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2, F3 && f3)
            {
                return static_partitioner_with_cleanup<
                        parallel_task_execution_policy, R, Result
                    >::call_with_index(policy, first, count,
                        std::forward<F1>(f1), std::forward<F2>(f2),
                        std::forward<F3>(f3), 0);
            }
        };

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
