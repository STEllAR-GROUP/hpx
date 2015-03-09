//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_SCAN_PARTITIONER_DEC_30_2014_0227PM)
#define HPX_PARALLEL_UTIL_SCAN_PARTITIONER_DEC_30_2014_0227PM

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

#include <algorithm>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // The static partitioner simply spawns one chunk of iterations for
        // each available core.
        template <typename ExPolicy, typename R, typename Result = void>
        struct static_scan_partitioner
        {
            template <typename FwdIter, typename T,
                typename F1, typename F2, typename F3>
            static R call(ExPolicy const& policy, FwdIter first,
                std::size_t count_, T && init, F1 && f1, F2 && f2, F3 && f3,
                std::size_t chunk_size)
            {
                std::vector<hpx::shared_future<Result> > workitems;
                std::vector<std::size_t> chunk_sizes;

                std::list<boost::exception_ptr> errors;

                try {
                    // pre-initialize first intermediate result
                    workitems.push_back(make_ready_future(std::forward<T>(init)));

                    // estimate a chunk size based on number of cores used
                    std::size_t count = count_;
                    HPX_ASSERT(count > 0);

                    chunk_size = get_static_chunk_size(policy, workitems, f1,
                        first, count, chunk_size);

                    // schedule every chunk on a separate thread
                    workitems.reserve(count_ / chunk_size + 2);
                    chunk_sizes.reserve(workitems.capacity());

                    // If the size of count was enough to warrant testing for a
                    // chunk_size, add to chunk_sizes and pre-initialize second
                    // intermediate result.
                    if (workitems.size() == 2)
                    {
                        chunk_sizes.push_back(count_ - count);
                        workitems[1] = lcos::local::dataflow(hpx::launch::sync,
                            f2, workitems[0], workitems[1]);
                    }

                    std::size_t parts = 0;

                    // Schedule first step of scan algorithm, step 2 is
                    // performed as soon as the current partition and the
                    // partition to the left is ready.
                    threads::executor exec = policy.get_executor();
                    while (count != 0)
                    {
                        std::size_t chunk = (std::min)(chunk_size, count);
                        BOOST_SCOPED_ENUM(hpx::launch) p = (++parts & 0x7) ?
                            hpx::launch::sync : hpx::launch::async;

                        if (exec)
                        {
                            workitems.push_back(
                                lcos::local::dataflow(
                                    p, f2, workitems.back(),
                                    hpx::async(exec, f1, first, chunk)
                                ));
                        }
                        else
                        {
                            workitems.push_back(
                                lcos::local::dataflow(
                                    p, f2, workitems.back(),
                                    hpx::async(hpx::launch::fork,
                                        f1, first, chunk)
                                ));
                        }

                        chunk_sizes.push_back(chunk);
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
                detail::handle_local_exceptions<
                    ExPolicy>::call(workitems, errors);

                // Execute step 3 of the scan algorithm
                return f3(std::move(workitems), chunk_sizes);
            }
        };

        template <typename R, typename Result>
        struct static_scan_partitioner<parallel_task_execution_policy, R, Result>
        {
            template <typename FwdIter, typename T,
                typename F1, typename F2, typename F3>
            static hpx::future<R> call(
                parallel_task_execution_policy const& policy,
                FwdIter first, std::size_t count_, T && init,
                F1 && f1, F2 && f2, F3 && f3, std::size_t chunk_size)
            {
                std::vector<hpx::shared_future<Result> > workitems;
                std::vector<std::size_t> chunk_sizes;

                std::list<boost::exception_ptr> errors;

                try {
                    // pre-initialize first intermediate result
                    workitems.push_back(make_ready_future(std::forward<T>(init)));

                    // estimate a chunk size based on number of cores used
                    std::size_t count = count_;
                    HPX_ASSERT(count > 0);

                    chunk_size = get_static_chunk_size(policy, workitems, f1,
                        first, count, chunk_size);

                    // schedule every chunk on a separate thread
                    workitems.reserve(count_ / chunk_size + 2);
                    chunk_sizes.reserve(workitems.capacity());

                    // If the size of count was enough to warrant testing for a
                    // chunk_size, add to chunk_sizes and pre-initialize second
                    // intermediate result.
                    if (workitems.size() == 2)
                    {
                        chunk_sizes.push_back(count_ - count);
                        workitems[1] = lcos::local::dataflow(hpx::launch::sync,
                            f2, workitems[0], workitems[1]);
                    }

                    std::size_t parts = 0;

                    // Schedule first step of scan algorithm, step 2 is
                    // performed as soon as the current partition and the
                    // partition to the left is ready.
                    threads::executor exec = policy.get_executor();
                    while (count != 0)
                    {
                        std::size_t chunk = (std::min)(chunk_size, count);
                        BOOST_SCOPED_ENUM(hpx::launch) p = (++parts & 0x7) ?
                            hpx::launch::sync : hpx::launch::async;

                        if (exec)
                        {
                            workitems.push_back(
                                lcos::local::dataflow(
                                    p, f2, workitems.back(),
                                    hpx::async(exec, f1, first, chunk)
                                ));
                        }
                        else
                        {
                            workitems.push_back(
                                lcos::local::dataflow(
                                    p, f2, workitems.back(),
                                    hpx::async(hpx::launch::fork,
                                        f1, first, chunk)
                                ));
                        }

                        chunk_sizes.push_back(chunk);
                        count -= chunk;
                        std::advance(first, chunk);
                    }
                }
                catch (std::bad_alloc const&) {
                    return hpx::make_exceptional_future<R>(
                        boost::current_exception());
                }
                catch (...) {
                    errors.push_back(boost::current_exception());
                }

                typedef typename parallel::detail::algorithm_result<
                        parallel_task_execution_policy, R
                    >::type result_type;

                // wait for all tasks to finish
                return lcos::local::dataflow(
                    [=](std::vector<hpx::shared_future<Result> >&& r) mutable
                      -> result_type
                    {
                        detail::handle_local_exceptions<
                                parallel_task_execution_policy
                            >::call(r, errors);

                        // Execute step 3 of the scan algorithm
                        return f3(std::move(r), chunk_sizes);
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
        struct scan_partitioner;

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename R, typename Result>
        struct scan_partitioner<ExPolicy, R, Result,
            parallel::traits::static_partitioner_tag>
        {
            template <typename FwdIter, typename T,
                typename F1, typename F2, typename F3>
            static R call(ExPolicy const& policy, FwdIter first,
                std::size_t count, T && init, F1 && f1, F2 && f2, F3 && f3,
                std::size_t chunk_size = 0)
            {
                return static_scan_partitioner<ExPolicy, R, Result>::call(
                    policy, first, count, std::forward<T>(init),
                    std::forward<F1>(f1), std::forward<F2>(f2),
                    std::forward<F3>(f3), chunk_size);
            }
        };

        template <typename R, typename Result>
        struct scan_partitioner<parallel_task_execution_policy, R, Result,
            parallel::traits::static_partitioner_tag>
        {
            template <typename FwdIter, typename T,
                typename F1, typename F2, typename F3>
            static hpx::future<R> call(
                parallel_task_execution_policy const& policy, FwdIter first,
                std::size_t count, T && init, F1 && f1, F2 && f2, F3 && f3,
                std::size_t chunk_size = 0)
            {
                return static_scan_partitioner<
                        parallel_task_execution_policy, R, Result
                    >::call(policy, first, count, std::forward<T>(init),
                        std::forward<F1>(f1), std::forward<F2>(f2),
                        std::forward<F3>(f3), chunk_size);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename R, typename Result>
        struct scan_partitioner<ExPolicy, R, Result,
                parallel::traits::default_partitioner_tag>
          : scan_partitioner<ExPolicy, R, Result,
                parallel::traits::static_partitioner_tag>
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename R = void, typename Result = R,
        typename PartTag = typename parallel::traits::extract_partitioner<
            typename hpx::util::decay<ExPolicy>::type
        >::type>
    struct scan_partitioner
      : detail::scan_partitioner<
            typename hpx::util::decay<ExPolicy>::type, R, Result, PartTag>
    {};
}}}

#endif
