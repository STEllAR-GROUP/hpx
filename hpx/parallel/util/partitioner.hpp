//  Copyright (c) 2007-2014 Hartmut Kaiser
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
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/detail/algorithm_result.hpp>
#include <hpx/util/decay.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util
{
    struct static_partitioner_tag {};
    struct auto_partitioner_tag {};
    struct default_partitioner_tag {};
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    template <typename ExPolicy, typename Enable = void>
    struct extract_partitioner
    {
        typedef parallel::util::default_partitioner_tag type;
    };
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy>
        struct handle_local_exceptions
        {
            // std::bad_alloc has to be handled separately
            static void call(boost::exception_ptr const& e,
                std::list<boost::exception_ptr>& errors)
            {
                try {
                    boost::rethrow_exception(e);
                }
                catch (std::bad_alloc const& ba) {
                    boost::throw_exception(ba);
                }
                catch (...) {
                    errors.push_back(e);
                }
            }

            template <typename T>
            static void call(std::vector<hpx::future<T> > const& workitems,
                std::list<boost::exception_ptr>& errors)
            {
                for (hpx::future<T> const& f: workitems)
                {
                    if (f.has_exception())
                        call(f.get_exception_ptr(), errors);
                }

                if (!errors.empty())
                    boost::throw_exception(exception_list(std::move(errors)));
            }
        };

        template <>
        struct handle_local_exceptions<parallel_vector_execution_policy>
        {
            HPX_ATTRIBUTE_NORETURN static void call(
                boost::exception_ptr const&, std::list<boost::exception_ptr>&)
            {
                std::terminate();
            }

            template <typename T>
            static void call(std::vector<hpx::future<T> > const& workitems,
                std::list<boost::exception_ptr>&)
            {
                for (hpx::future<T> const& f: workitems)
                {
                    if (f.has_exception())
                        hpx::terminate();
                }
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename R, typename F, typename FwdIter>
        void add_ready_future(std::vector<hpx::future<R> >& workitems,
            F && f, FwdIter first, std::size_t count)
        {
            workitems.push_back(hpx::make_ready_future(f(first, count)));
        }

        template <typename F, typename FwdIter>
        void add_ready_future(std::vector<hpx::future<void> >&,
            F && f, FwdIter first, std::size_t count)
        {
            f(first, count);
        }

        // estimate a chunk size based on number of cores used
        template <typename Result, typename F1, typename FwdIter>
        std::size_t auto_chunk_size(
            std::vector<hpx::future<Result> >& workitems,
            F1 && f1, FwdIter& first, std::size_t& count)
        {
            std::size_t test_chunk_size = count / 100;
            if (0 == test_chunk_size) return 0;

            boost::uint64_t t = hpx::util::high_resolution_clock::now();
            add_ready_future(workitems, f1, first, test_chunk_size);

            t = (hpx::util::high_resolution_clock::now() - t) / test_chunk_size;

            std::advance(first, test_chunk_size);
            count -= test_chunk_size;

            // return chunk size which will create 80 microseconds of work
            return t == 0 ? 0 : (std::min)(count, 80000 / t);
        }

        template <typename ExPolicy, typename Result, typename F1,
            typename FwdIter>
        std::size_t get_static_chunk_size(ExPolicy const& policy,
            std::vector<hpx::future<Result> >& workitems,
            F1 && f1, FwdIter& first, std::size_t& count,
            std::size_t chunk_size)
        {
            threads::executor exec = policy.get_executor();
            if (chunk_size == 0)
            {
                chunk_size = policy.get_chunk_size();
                if (chunk_size == 0)
                {
                    std::size_t const cores = hpx::get_os_thread_count(exec);
                    if (count > 100*cores)
                        chunk_size = auto_chunk_size(workitems, f1, first, count);

                    if (chunk_size == 0)
                        chunk_size = (count + cores - 1) / cores;
                }
            }
            return chunk_size;
        }

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
        struct foreach_n_static_partitioner<task_execution_policy, Result>
        {
            template <typename FwdIter, typename F1>
            static hpx::future<FwdIter> call(
                task_execution_policy const& policy,
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
                    return hpx::make_error_future<FwdIter>(
                        boost::current_exception());
                }
                catch (...) {
                    errors.push_back(boost::current_exception());
                }

                // wait for all tasks to finish
                return hpx::lcos::local::dataflow(
                    [first, errors](std::vector<hpx::future<Result> > && r) mutable
                    {
                        detail::handle_local_exceptions<task_execution_policy>
                            ::call(r, errors);
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
        struct foreach_n_partitioner<ExPolicy, Result, static_partitioner_tag>
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
            task_execution_policy, Result, static_partitioner_tag>
        {
            template <typename FwdIter, typename F1>
            static hpx::future<FwdIter> call(
                task_execution_policy const& policy,
                FwdIter first, std::size_t count, F1 && f1)
            {
                return foreach_n_static_partitioner<
                        task_execution_policy, Result
                    >::call(policy, first, count, std::forward<F1>(f1), 0);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Result>
        struct foreach_n_partitioner<ExPolicy, Result, default_partitioner_tag>
          : foreach_n_partitioner<ExPolicy, Result, static_partitioner_tag>
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

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename R, typename F, typename FwdIter>
        void add_ready_future_idx(std::vector<hpx::future<R> >& workitems,
            F && f, std::size_t base_idx, FwdIter first, std::size_t count)
        {
            workitems.push_back(
                hpx::make_ready_future(f(base_idx, first, count)));
        }

        template <typename F, typename FwdIter>
        void add_ready_future_idx(std::vector<hpx::future<void> >&,
            F && f, std::size_t base_idx, FwdIter first, std::size_t count)
        {
            f(base_idx, first, count);
        }

        // estimate a chunk size based on number of cores used, take into
        // account base index
        template <typename Result, typename F1, typename FwdIter>
        std::size_t auto_chunk_size_idx(
            std::vector<hpx::future<Result> >& workitems, F1 && f1,
            std::size_t& base_idx, FwdIter& first, std::size_t& count)
        {
            std::size_t test_chunk_size = count / 100;
            if (0 == test_chunk_size) return 0;

            boost::uint64_t t = hpx::util::high_resolution_clock::now();
            add_ready_future_idx(workitems, f1, base_idx, first, test_chunk_size);

            t = (hpx::util::high_resolution_clock::now() - t) / test_chunk_size;

            base_idx += test_chunk_size;
            std::advance(first, test_chunk_size);
            count -= test_chunk_size;

            // return chunk size which will create 80 microseconds of work
            return t == 0 ? 0 : (std::min)(count, 80000 / t);
        }

        template <typename ExPolicy, typename Result, typename F1,
            typename FwdIter>
        std::size_t get_static_chunk_size_idx(ExPolicy const& policy,
            std::vector<hpx::future<Result> >& workitems,
            F1 && f1, std::size_t& base_idx, FwdIter& first,
            std::size_t& count, std::size_t chunk_size)
        {
            threads::executor exec = policy.get_executor();
            if (chunk_size == 0)
            {
                chunk_size = policy.get_chunk_size();
                if (chunk_size == 0)
                {
                    std::size_t const cores = hpx::get_os_thread_count(exec);
                    if (count > 100*cores)
                        chunk_size = auto_chunk_size_idx(workitems, f1,
                            base_idx, first, count);

                    if (chunk_size == 0)
                        chunk_size = (count + cores - 1) / cores;
                }
            }
            return chunk_size;
        }

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
                    workitems, errors);

                return f2(std::move(workitems));
            }
        };

        template <typename R, typename Result>
        struct static_partitioner<task_execution_policy, R, Result>
        {
            template <typename FwdIter, typename F1, typename F2>
            static hpx::future<R> call(task_execution_policy const& policy,
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
                    return hpx::make_error_future<R>(
                        boost::current_exception());
                }
                catch (...) {
                    errors.push_back(boost::current_exception());
                }

                // wait for all tasks to finish
                return hpx::lcos::local::dataflow(
                    [f2, errors](std::vector<hpx::future<Result> > && r) mutable
                    {
                        detail::handle_local_exceptions<task_execution_policy>
                            ::call(r, errors);
                        return f2(std::move(r));
                    },
                    std::move(workitems));
            }

            template <typename FwdIter, typename F1, typename F2>
            static hpx::future<R> call_with_index(
                task_execution_policy const& policy,
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
                    return hpx::make_error_future<R>(
                        boost::current_exception());
                }
                catch (...) {
                    errors.push_back(boost::current_exception());
                }

                // wait for all tasks to finish
                return hpx::lcos::local::dataflow(
                    [f2, errors](std::vector<hpx::future<Result> > && r) mutable
                    {
                        detail::handle_local_exceptions<task_execution_policy>
                            ::call(r, errors);
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
        struct partitioner<ExPolicy, R, Result, static_partitioner_tag>
        {
            template <typename FwdIter, typename F1, typename F2>
            static R call(ExPolicy const& policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2)
            {
                return static_partitioner<ExPolicy, R, Result>::call(
                    policy, first, count,
                    std::forward<F1>(f1), std::forward<F2>(f2), 0);
            }

            template <typename FwdIter, typename F1, typename F2>
            static R call_with_index(ExPolicy const& policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2)
            {
                return static_partitioner<ExPolicy, R, Result>::call_with_index(
                    policy, first, count,
                    std::forward<F1>(f1), std::forward<F2>(f2), 0);
            }
        };

        template <typename R, typename Result>
        struct partitioner<task_execution_policy, R, Result, static_partitioner_tag>
        {
            template <typename FwdIter, typename F1, typename F2>
            static hpx::future<R> call(task_execution_policy const& policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2)
            {
                return static_partitioner<
                        task_execution_policy, R, Result
                    >::call(policy, first, count,
                        std::forward<F1>(f1), std::forward<F2>(f2), 0);
            }

            template <typename FwdIter, typename F1, typename F2>
            static hpx::future<R> call_with_index(
                task_execution_policy const& policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2)
            {
                return static_partitioner<
                        task_execution_policy, R, Result
                    >::call_with_index(policy, first, count,
                        std::forward<F1>(f1), std::forward<F2>(f2), 0);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename R, typename Result>
        struct partitioner<ExPolicy, R, Result, default_partitioner_tag>
          : partitioner<ExPolicy, R, Result, static_partitioner_tag>
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
