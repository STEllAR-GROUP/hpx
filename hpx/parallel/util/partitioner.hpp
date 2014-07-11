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

        ///////////////////////////////////////////////////////////////////////
        template <>
        struct handle_local_exceptions<parallel_vector_execution_policy>
        {
            static void call(boost::exception_ptr const&,
                std::list<boost::exception_ptr>&)
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

        template <typename F, typename FwdIter, typename R>
        R handle_step_two(F && f, FwdIter,
            std::vector<hpx::future<R> >&& workitems)
        {
            return f(std::move(workitems));
        }

        template <typename F, typename FwdIter>
        FwdIter handle_step_two(F &&, FwdIter first,
            std::vector<hpx::future<void> >&&)
        {
            return first;
        }

        ///////////////////////////////////////////////////////////////////////
        // The static partitioner simply spawns one chunk of iterations for
        // each available core.
        template <typename ExPolicy, typename R, typename Result = void>
        struct static_partitioner
        {
            template <typename FwdIter, typename F1, typename F2>
            static R call(ExPolicy const& policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2,
                std::size_t chunk_size = 0)
            {
                // estimate a chunk size based on number of cores used
                threads::executor exec = policy.get_executor();
                if (chunk_size == 0)
                {
                    chunk_size = policy.get_chunk_size();
                    if (chunk_size == 0)
                    {
                        std::size_t const cores = hpx::get_os_thread_count(exec);
                        chunk_size = (count + cores - 1) / cores;
                    }
                }

                // schedule every chunk on a separate thread
                std::vector<hpx::future<Result> > workitems;
                workitems.reserve(count / chunk_size + 1);

                while (count > chunk_size)
                {
                    workitems.push_back(hpx::async(exec, f1, first, chunk_size));
                    count -= chunk_size;
                    std::advance(first, chunk_size);
                }

                std::list<boost::exception_ptr> errors;

                // execute last chunk directly
                if (count != 0)
                {
                    try {
                        add_ready_future(workitems, std::forward<F1>(f1),
                            first, count);
                    }
                    catch (...) {
                        detail::handle_local_exceptions<ExPolicy>::call(
                            boost::current_exception(), errors);
                    }
                    std::advance(first, count);
                }

                // wait for all tasks to finish
                hpx::wait_all(workitems);
                detail::handle_local_exceptions<ExPolicy>::call(
                    workitems, errors);
                return handle_step_two(std::forward<F2>(f2), first,
                    std::move(workitems));
            }
        };

        template <typename R, typename Result>
        struct static_partitioner<task_execution_policy, R, Result>
        {
            template <typename FwdIter, typename F1, typename F2>
            static hpx::future<R> call(task_execution_policy const& policy,
                FwdIter first, std::size_t count,
                F1 && f1, F2 && f2, std::size_t chunk_size = 0)
            {
                // estimate a chunk size based on number of cores used
                threads::executor exec = policy.get_executor();
                if (chunk_size == 0)
                {
                    chunk_size = policy.get_chunk_size();
                    if (chunk_size == 0)
                    {
                        std::size_t const cores = hpx::get_os_thread_count(exec);
                        chunk_size = (count + cores - 1) / cores;
                    }
                }

                // schedule every chunk on a separate thread
                std::vector<hpx::future<Result> > workitems;
                workitems.reserve(count / chunk_size + 1);

                while (count > chunk_size)
                {
                    workitems.push_back(hpx::async(exec, f1, first, chunk_size));
                    count -= chunk_size;
                    std::advance(first, chunk_size);
                }

                // add last chunk
                if (count != 0)
                {
                    workitems.push_back(hpx::async(exec, f1, first, count));
                    std::advance(first, count);
                }

                // wait for all tasks to finish
                return hpx::lcos::local::dataflow(
                    [first, f2](std::vector<hpx::future<Result> > && r) mutable
                    {
                        std::list<boost::exception_ptr> errors;
                        detail::handle_local_exceptions<task_execution_policy>
                            ::call(r, errors);
                        return handle_step_two(f2, first, std::move(r));
                    },
                    std::move(workitems));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename R, typename PartTag>
        struct partitioner;

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy>
        struct partitioner<ExPolicy, void, static_partitioner_tag>
        {
            template <typename FwdIter, typename F>
            static FwdIter call(ExPolicy const& policy, FwdIter first,
                std::size_t count, F && f, std::size_t chunk_size = 0)
            {
                return static_partitioner<ExPolicy, FwdIter, void>::call(
                    policy, first, count, std::forward<F>(f), 0, chunk_size);
            }
        };

        template <typename ExPolicy, typename R>
        struct partitioner<ExPolicy, R, static_partitioner_tag>
        {
            template <typename FwdIter, typename F1, typename F2>
            static R call(ExPolicy const& policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2,
                std::size_t chunk_size = 0)
            {
                return static_partitioner<ExPolicy, R, R>::call(
                    policy, first, count,
                    std::forward<F1>(f1), std::forward<F2>(f2), chunk_size);
            }
        };

        template <>
        struct partitioner<task_execution_policy, void, static_partitioner_tag>
        {
            template <typename FwdIter, typename F>
            static hpx::future<FwdIter> call(task_execution_policy const& policy,
                FwdIter first, std::size_t count, F && f,
                std::size_t chunk_size = 0)
            {
                return static_partitioner<task_execution_policy, FwdIter>::call(
                    policy, first, count, std::forward<F>(f), 0, chunk_size);
            }
        };

        template <typename R>
        struct partitioner<task_execution_policy, R, static_partitioner_tag>
        {
            template <typename FwdIter, typename F1, typename F2>
            static hpx::future<R> call(task_execution_policy const& policy,
                FwdIter first, std::size_t count,
                F1 && f1, F2 && f2, std::size_t chunk_size = 0)
            {
                return static_partitioner<task_execution_policy, R, R>::call(
                    policy, first, count,
                    std::forward<F1>(f1), std::forward<F2>(f2), chunk_size);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename R>
        struct partitioner<ExPolicy, R, default_partitioner_tag>
          : partitioner<ExPolicy, R, static_partitioner_tag>
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename R = void,
        typename PartTag = typename parallel::traits::extract_partitioner<
            typename hpx::util::decay<ExPolicy>::type
        >::type>
    struct partitioner
      : detail::partitioner<typename hpx::util::decay<ExPolicy>::type, R, PartTag>
    {};
}}}

#endif
