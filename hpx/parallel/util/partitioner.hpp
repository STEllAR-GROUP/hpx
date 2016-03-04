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
#include <hpx/dataflow.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/tuple.hpp>

#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/chunk_size.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/traits/extract_partitioner.hpp>

#include <boost/range/functions.hpp>
#include <iterator>

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
            static R call(ExPolicy policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2)
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
                        first, count);

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
                    inititems, errors);

                return f2(std::move(inititems));
            }

            template <typename FwdIter, typename F1, typename F2, typename Data>
                // requires is_container<Data>
            static R call_with_data(ExPolicy policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2,
                std::vector<std::size_t> const& chunk_sizes, Data && data)
            {
                HPX_ASSERT(boost::size(data) >= boost::size(chunk_sizes));

                typedef typename ExPolicy::executor_type executor_type;
                typedef typename hpx::parallel::executor_traits<executor_type>
                    executor_traits;
                typedef typename hpx::util::decay<Data>::type data_type;

                typename data_type::const_iterator data_it = boost::begin(data);
                typename std::vector<std::size_t>::const_iterator chunk_size_it =
                    boost::begin(chunk_sizes);

                typedef typename hpx::util::tuple<
                        typename data_type::value_type, FwdIter, std::size_t
                    > tuple_type;

                std::vector<hpx::future<Result> > workitems;
                std::list<boost::exception_ptr> errors;
                std::vector<tuple_type> shape;

                try {
                    // schedule every chunk on a separate thread
                    shape.reserve(chunk_sizes.size());
                    while(count != 0)
                    {
                        std::size_t chunk = (std::min)(count, *chunk_size_it);
                        HPX_ASSERT(chunk != 0);

                        shape.push_back(hpx::util::make_tuple(
                            *data_it, first, chunk));

                        count -= chunk;
                        std::advance(first, chunk);

                        ++data_it;
                        ++chunk_size_it;
                    }

                    HPX_ASSERT(chunk_size_it == chunk_sizes.end());
                    workitems.reserve(chunk_sizes.size());

                    using hpx::util::bind;
                    using hpx::util::functional::invoke_fused;
                    using hpx::util::placeholders::_1;
                    workitems = executor_traits::async_execute(
                        policy.executor(),
                        bind(invoke_fused(), std::forward<F1>(f1), _1),
                        shape);
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
            static R call_with_index(ExPolicy policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2)
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
                    shape = get_bulk_iteration_shape_idx(policy, inititems, f1,
                        first, count);

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
                    inititems, errors);

                return f2(std::move(inititems));
            }
        };

        template <typename R, typename Result>
        struct static_partitioner<parallel_task_execution_policy, R, Result>
        {
            template <typename ExPolicy, typename FwdIter, typename F1, typename F2>
            static hpx::future<R> call(ExPolicy policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2)
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
                        first, count);

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
                return hpx::dataflow(
                    [f2, errors](std::vector<hpx::future<Result> > && r) mutable -> R
                    {
                        detail::handle_local_exceptions<ExPolicy>::call(r, errors);
                        return f2(std::move(r));
                    },
                    std::move(inititems));
            }

            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2, typename Data>
                // requires is_container<Data>
            static hpx::future<R> call_with_data(ExPolicy policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2,
                std::vector<std::size_t> const& chunk_sizes, Data && data)
            {
                HPX_ASSERT(boost::size(data) >= boost::size(chunk_sizes));

                typedef typename ExPolicy::executor_type executor_type;
                typedef typename hpx::parallel::executor_traits<executor_type>
                    executor_traits;
                typedef typename hpx::util::decay<Data>::type data_type;

                typename data_type::const_iterator data_it = boost::begin(data);
                typename std::vector<std::size_t>::const_iterator chunk_size_it =
                    boost::begin(chunk_sizes);

                typedef typename hpx::util::tuple<
                        typename data_type::value_type, FwdIter, std::size_t
                    > tuple_type;

                std::vector<hpx::future<Result> > workitems;
                std::list<boost::exception_ptr> errors;
                std::vector<tuple_type> shape;

                try {
                    // schedule every chunk on a separate thread
                    shape.reserve(chunk_sizes.size());
                    while(count != 0)
                    {
                        std::size_t chunk = (std::min)(count, *chunk_size_it);
                        HPX_ASSERT(chunk != 0);

                        shape.push_back(hpx::util::make_tuple(
                            *data_it, first, chunk));

                        count -= chunk;
                        std::advance(first, chunk);

                        ++data_it;
                        ++chunk_size_it;
                    }

                    HPX_ASSERT(chunk_size_it == chunk_sizes.end());
                    workitems.reserve(chunk_sizes.size());

                    using hpx::util::bind;
                    using hpx::util::functional::invoke_fused;
                    using hpx::util::placeholders::_1;
                    workitems = executor_traits::async_execute(
                        policy.executor(),
                        bind(invoke_fused(), std::forward<F1>(f1), _1),
                        shape);
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
                    [f2, errors](std::vector<hpx::future<Result> > && r) mutable -> R
                    {
                        detail::handle_local_exceptions<ExPolicy>::call(r, errors);
                        return f2(std::move(r));
                    },
                    std::move(workitems));
            }

            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2>
            static hpx::future<R> call_with_index(ExPolicy policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2)
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
                    shape = get_bulk_iteration_shape_idx(policy, inititems, f1,
                        first, count);

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
                return hpx::dataflow(
                    [f2, errors](std::vector<hpx::future<Result> > && r)
                        mutable -> R
                    {
                        detail::handle_local_exceptions<ExPolicy>::call(r, errors);
                        return f2(std::move(r));
                    },
                    std::move(inititems));
            }
        };

        template <typename Executor, typename Parameters, typename R,
            typename Result>
        struct static_partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                R, Result>
          : static_partitioner<parallel_task_execution_policy, R, Result>
        {};

        ///////////////////////////////////////////////////////////////////////
        // ExPolicy: execution policy
        // R:        overall result type
        // Result:   intermediate result type of first step
        // PartTag:  select appropriate partitioner
        template <typename ExPolicy, typename R, typename Result, typename Tag>
        struct partitioner;

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename R, typename Result>
        struct partitioner<ExPolicy, R, Result,
            parallel::traits::static_partitioner_tag>
        {
            template <typename FwdIter, typename F1, typename F2>
            static R call(ExPolicy policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2)
            {
                return static_partitioner<ExPolicy, R, Result>::call(
                    policy, first, count,
                    std::forward<F1>(f1), std::forward<F2>(f2));
            }

            template <typename FwdIter, typename F1, typename F2, typename Data>
            static R call_with_data(ExPolicy policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2,
                std::vector<std::size_t> const& chunk_sizes, Data && data)
            {
                return static_partitioner<ExPolicy, R, Result>::call_with_data(
                    policy, first, count,
                    std::forward<F1>(f1), std::forward<F2>(f2), chunk_sizes,
                    std::forward<Data>(data));
            }

            template <typename FwdIter, typename F1, typename F2>
            static R call_with_index(ExPolicy policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2)
            {
                return static_partitioner<ExPolicy, R, Result>::call_with_index(
                    policy, first, count,
                    std::forward<F1>(f1), std::forward<F2>(f2));
            }
        };

        template <typename R, typename Result>
        struct partitioner<parallel_task_execution_policy, R, Result,
            parallel::traits::static_partitioner_tag>
        {
            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2>
            static hpx::future<R> call(ExPolicy policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2)
            {
                return static_partitioner<ExPolicy, R, Result>::call(
                    policy, first, count,
                    std::forward<F1>(f1), std::forward<F2>(f2));
            }

            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2, typename Data>
            static hpx::future<R> call_with_data(ExPolicy policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2,
                std::vector<std::size_t> const& chunk_sizes, Data && data)
            {
                return static_partitioner<ExPolicy, R, Result>::call_with_data(
                    policy, first, count,
                    std::forward<F1>(f1), std::forward<F2>(f2),
                    chunk_sizes, std::forward<Data>(data));
            }

            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2>
            static hpx::future<R> call_with_index(ExPolicy policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2)
            {
                return static_partitioner<ExPolicy, R, Result>::call_with_index(
                    policy, first, count,
                    std::forward<F1>(f1), std::forward<F2>(f2));
            }
        };

        template <typename Executor, typename Parameters, typename R,
            typename Result>
        struct partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                R, Result, parallel::traits::static_partitioner_tag>
          : partitioner<parallel_task_execution_policy, R, Result,
                parallel::traits::static_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename R,
            typename Result>
        struct partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                R, Result, parallel::traits::auto_partitioner_tag>
          : partitioner<parallel_task_execution_policy, R, Result,
                parallel::traits::auto_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename R,
            typename Result>
        struct partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                R, Result, parallel::traits::default_partitioner_tag>
          : partitioner<parallel_task_execution_policy, R, Result,
                parallel::traits::static_partitioner_tag>
        {};

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
