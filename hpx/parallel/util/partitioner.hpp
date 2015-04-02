//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_PARTITIONER_MAY_27_2014_1040PM)
#define HPX_PARALLEL_UTIL_PARTITIONER_MAY_27_2014_1040PM

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
        // The static partitioner simply spawns one chunk of iterations for
        // each available core.
        template <typename ExPolicy, typename R, typename Result = void>
        struct static_partitioner
        {
            template <typename FwdIter, typename F1, typename F2>
            static R call(ExPolicy const& policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2, std::size_t chunk_size)
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
                    while(count != 0)
                    {
                        std::size_t chunk = (std::min)(count, chunk_size);
                        if (exec)
                        {
                            workitems.push_back(hpx::async(exec, f1, first,
                                chunk));
                        }
                        else
                        {
                            workitems.push_back(hpx::async(hpx::launch::fork, f1,
                                first, chunk));
                        }

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

                return f2(std::move(workitems));
            }

            template <typename FwdIter, typename F1, typename F2, typename Data>
                // requires is_container<Data>
            static R call_with_data(ExPolicy const& policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2,
                std::vector<std::size_t> const& chunk_sizes, Data && data)
            {
                HPX_ASSERT(boost::size(data) >= boost::size(chunk_sizes));

                typename hpx::util::decay<Data>::type::const_iterator data_it =
                    boost::begin(data);
                typename std::vector<std::size_t>::const_iterator chunk_size_it =
                    boost::begin(chunk_sizes);

                std::vector<hpx::future<Result> > workitems;
                std::list<boost::exception_ptr> errors;

                try {
                    // schedule every chunk on a separate thread
                    workitems.reserve(chunk_sizes.size());

                    threads::executor exec = policy.get_executor();
                    while(count != 0)
                    {
                        std::size_t chunk = (std::min)(count, *chunk_size_it);
                        HPX_ASSERT(chunk != 0);

                        if (exec)
                        {
                            workitems.push_back(hpx::async(exec,
                                f1, *data_it, first, chunk));
                        }
                        else
                        {
                            workitems.push_back(hpx::async(hpx::launch::fork,
                                f1, *data_it, first, chunk));
                        }

                        count -= chunk;
                        std::advance(first, chunk);

                        ++data_it;
                        ++chunk_size_it;
                    }

                    HPX_ASSERT(chunk_size_it == chunk_sizes.end());
                }
                catch (...) {
                    detail::handle_local_exceptions<ExPolicy>::call(
                        boost::current_exception(), errors);
                }

                // wait for all tasks to finish
                hpx::wait_all(workitems);
                detail::handle_local_exceptions<ExPolicy>::call(
                    workitems, errors);

                return f2(std::move(workitems));
            }

            template <typename FwdIter, typename F1, typename F2>
            static R call_with_index(ExPolicy const& policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2, std::size_t chunk_size)
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
                    while(count != 0)
                    {
                        std::size_t chunk = (std::min)(count, chunk_size);
                        if (exec)
                        {
                            workitems.push_back(hpx::async(exec, f1, base_idx,
                                first, chunk));
                        }
                        else
                        {
                            workitems.push_back(hpx::async(hpx::launch::fork,
                                f1, base_idx, first, chunk));
                        }

                        count -= chunk;
                        std::advance(first, chunk);
                        base_idx += chunk;
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

                return f2(std::move(workitems));
            }
        };

        template <typename R, typename Result>
        struct static_partitioner<parallel_task_execution_policy, R, Result>
        {
            template <typename FwdIter, typename F1, typename F2>
            static hpx::future<R> call(
                parallel_task_execution_policy const& policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2,
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
                    while(count != 0)
                    {
                        std::size_t chunk = (std::min)(count, chunk_size);
                        if (exec)
                        {
                            workitems.push_back(hpx::async(exec, f1, first,
                                chunk));
                        }
                        else
                        {
                            workitems.push_back(hpx::async(hpx::launch::fork,
                                f1, first, chunk));
                        }

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

                // wait for all tasks to finish
                return hpx::lcos::local::dataflow(
                    [f2, errors](std::vector<hpx::future<Result> > && r) mutable -> R
                    {
                        detail::handle_local_exceptions<
                                parallel_task_execution_policy
                            >::call(r, errors);
                        return f2(std::move(r));
                    },
                    std::move(workitems));
            }

            template <typename FwdIter, typename F1, typename F2,
                typename Data>
                // requires is_container<Data>
            static hpx::future<R> call_with_data(
                parallel_task_execution_policy const& policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2,
                std::vector<std::size_t> const& chunk_sizes, Data && data)
            {
                HPX_ASSERT(boost::size(data) >= boost::size(chunk_sizes));

                typename hpx::util::decay<Data>::type::const_iterator data_it =
                    boost::begin(data);
                typename std::vector<std::size_t>::const_iterator chunk_size_it =
                    boost::begin(chunk_sizes);

                std::vector<hpx::future<Result> > workitems;
                std::list<boost::exception_ptr> errors;

                try {
                    // schedule every chunk on a separate thread
                    workitems.reserve(chunk_sizes.size());

                    threads::executor exec = policy.get_executor();
                    while (count != 0)
                    {
                        std::size_t chunk = *chunk_size_it;
                        HPX_ASSERT(chunk != 0 && count >= chunk);

                        if (exec)
                        {
                            workitems.push_back(hpx::async(exec,
                                f1, *data_it, first, chunk));
                        }
                        else
                        {
                            workitems.push_back(hpx::async(hpx::launch::fork,
                                f1, *data_it, first, chunk));
                        }

                        count -= chunk;
                        std::advance(first, chunk);

                        ++data_it;
                        ++chunk_size_it;
                    }

                    HPX_ASSERT(chunk_size_it == chunk_sizes.end());
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
                    [f2, errors](std::vector<hpx::future<Result> > && r) mutable -> R
                    {
                        detail::handle_local_exceptions<
                                parallel_task_execution_policy
                            >::call(r, errors);
                        return f2(std::move(r));
                    },
                    std::move(workitems));
            }

            template <typename FwdIter, typename F1, typename F2>
            static hpx::future<R> call_with_index(
                parallel_task_execution_policy const& policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2,
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
                    while(count != 0)
                    {
                        std::size_t chunk = (std::min)(count, chunk_size);
                        if (exec)
                        {
                            workitems.push_back(hpx::async(exec, f1, base_idx,
                                first, chunk));
                        }
                        else
                        {
                            workitems.push_back(hpx::async(hpx::launch::fork,
                                f1, base_idx, first, chunk));
                        }

                        count -= chunk;
                        std::advance(first, chunk);
                        base_idx += chunk;
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
                    [f2, errors](std::vector<hpx::future<Result> > && r)
                        mutable -> R
                    {
                        detail::handle_local_exceptions<
                                parallel_task_execution_policy
                            >::call(r, errors);
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
        struct partitioner;

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename R, typename Result>
        struct partitioner<ExPolicy, R, Result,
            parallel::traits::static_partitioner_tag>
        {
            template <typename FwdIter, typename F1, typename F2>
            static R call(ExPolicy const& policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2,
                std::size_t chunk_size = 0)
            {
                return static_partitioner<ExPolicy, R, Result>::call(
                    policy, first, count,
                    std::forward<F1>(f1), std::forward<F2>(f2), chunk_size);
            }

            template <typename FwdIter, typename F1, typename F2,
                typename Data>
            static R call_with_data(ExPolicy const& policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2,
                std::vector<std::size_t> const& chunk_sizes, Data && data)
            {
                return static_partitioner<ExPolicy, R, Result>::call_with_data(
                    policy, first, count,
                    std::forward<F1>(f1), std::forward<F2>(f2), chunk_sizes,
                    std::forward<Data>(data));
            }

            template <typename FwdIter, typename F1, typename F2>
            static R call_with_index(ExPolicy const& policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2,
                std::size_t chunk_size = 0)
            {
                return static_partitioner<ExPolicy, R, Result>::call_with_index(
                    policy, first, count,
                    std::forward<F1>(f1), std::forward<F2>(f2), chunk_size);
            }
        };

        template <typename R, typename Result>
        struct partitioner<parallel_task_execution_policy, R, Result,
            parallel::traits::static_partitioner_tag>
        {
            template <typename FwdIter, typename F1, typename F2>
            static hpx::future<R> call(
                parallel_task_execution_policy const& policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2,
                std::size_t chunk_size = 0)
            {
                return static_partitioner<
                        parallel_task_execution_policy, R, Result
                    >::call(policy, first, count,
                        std::forward<F1>(f1), std::forward<F2>(f2), chunk_size);
            }

            template <typename FwdIter, typename F1, typename F2,
                typename Data>
            static hpx::future<R> call_with_data(
                parallel_task_execution_policy const& policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2,
                std::vector<std::size_t> const& chunk_sizes, Data && data)
            {
                return static_partitioner<
                    parallel_task_execution_policy, R, Result
                >::call_with_data(policy, first, count,
                    std::forward<F1>(f1), std::forward<F2>(f2),
                    chunk_sizes, std::forward<Data>(data));
            }

            template <typename FwdIter, typename F1, typename F2>
            static hpx::future<R> call_with_index(
                parallel_task_execution_policy const& policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2,
                std::size_t chunk_size = 0)
            {
                return static_partitioner<
                        parallel_task_execution_policy, R, Result
                    >::call_with_index(policy, first, count,
                        std::forward<F1>(f1), std::forward<F2>(f2), chunk_size);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename R, typename Result>
        struct partitioner<ExPolicy, R, Result,
                parallel::traits::default_partitioner_tag>
          : partitioner<ExPolicy, R, Result,
                parallel::traits::static_partitioner_tag>
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename R = void, typename Result = R,
        typename PartTag = typename parallel::traits::extract_partitioner<
            typename hpx::util::decay<ExPolicy>::type
        >::type>
    struct partitioner
      : detail::partitioner<
            typename hpx::util::decay<ExPolicy>::type, R, Result, PartTag>
    {};
}}}

#endif
