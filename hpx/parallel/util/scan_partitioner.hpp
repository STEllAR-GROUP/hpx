//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_SCAN_PARTITIONER_DEC_30_2014_0227PM)
#define HPX_PARALLEL_UTIL_SCAN_PARTITIONER_DEC_30_2014_0227PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/async.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/deferred_call.hpp>

#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/chunk_size.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
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
        template <typename ExPolicy, typename R, typename Result1,
            typename Result2>
        struct static_scan_partitioner
        {
            template <typename FwdIter, typename T,
                typename F1, typename F2, typename F3, typename F4>
            static R call(ExPolicy policy, FwdIter first,
                std::size_t count, T && init, F1 && f1, F2 && f2, F3 && f3,
                F4 && f4)
            {
                typedef typename ExPolicy::executor_type executor_type;
                typedef typename hpx::parallel::executor_traits<executor_type>
                    executor_traits;
                typedef typename hpx::util::tuple<
                        FwdIter, std::size_t
                    > tuple_type;

                using hpx::util::get;
                using hpx::util::placeholders::_1;
                using hpx::util::deferred_call;

                std::vector<hpx::shared_future<Result1> > workitems;
                std::vector<hpx::future<Result2> > finalitems;
                std::vector<tuple_type> shape;

                std::list<boost::exception_ptr> errors;

                try {
                    // pre-initialize first intermediate result
                    workitems.push_back(make_ready_future(std::forward<T>(init)));

                    HPX_ASSERT(count > 0);
                    FwdIter first_ = first;
                    std::size_t test_chunk_size = count / 100;

                    // estimate a chunk size based on number of cores used
                    shape = get_bulk_iteration_shape(policy, workitems, f1,
                        first, count, 1);

                    // schedule every chunk on a separate thread
                    workitems.reserve(shape.size() + 1);
                    finalitems.reserve(shape.size());

                    // If the size of count was enough to warrant testing for a
                    // chunk, pre-initialize second intermediate result and
                    // start f3.
                    if (workitems.size() == 2)
                    {
                        HPX_ASSERT(workitems.size() < 3);

                        workitems[1] = dataflow(hpx::launch::sync,
                            f2, workitems[0], workitems[1]
                        );

                        finalitems.push_back(dataflow(
                            policy.executor(),
                            hpx::util::bind(
                                f3, first_, test_chunk_size, _1),
                            workitems[0], workitems[1])
                        );
                    }

                    std::size_t parts = workitems.size();

                    // Schedule first step of scan algorithm, step 2 is
                    // performed as soon as the current partition and the
                    // partition to the left is ready.
                    for(auto const& elem: shape)
                    {
                        BOOST_SCOPED_ENUM(hpx::launch) p = (parts & 0x7) ?
                            hpx::launch::sync : hpx::launch::async;

                        workitems.push_back(
                            dataflow(
                                p, f2, workitems.back(),
                                executor_traits::async_execute(
                                    policy.executor(),
                                    deferred_call(f1, get<0>(elem), get<1>(elem))
                                )
                            )
                        );

                        finalitems.push_back(
                            dataflow(
                                policy.executor(),
                                hpx::util::bind(
                                    f3, get<0>(elem), get<1>(elem), _1),
                                workitems[parts - 1], workitems[parts]
                            )
                        );

                        ++parts;
                    }
                }
                catch (...) {
                    detail::handle_local_exceptions<ExPolicy>::call(
                        boost::current_exception(), errors);
                }

                hpx::wait_all(finalitems, workitems);

                detail::handle_local_exceptions<
                    ExPolicy>::call(workitems, errors);
                detail::handle_local_exceptions<
                    ExPolicy>::call(finalitems, errors);

                return f4(std::move(workitems), std::move(finalitems));
            }
        };

        template <typename R, typename Result1, typename Result2>
        struct static_scan_partitioner<
            parallel_task_execution_policy, R, Result1, Result2>
        {
            template <typename ExPolicy, typename FwdIter, typename T,
                typename F1, typename F2, typename F3, typename F4>
            static hpx::future<R> call(ExPolicy policy,
                FwdIter first, std::size_t count, T && init, F1 && f1,
                F2 && f2, F3 && f3, F4 && f4)
            {
                typedef typename ExPolicy::executor_type executor_type;
                typedef typename hpx::parallel::executor_traits<executor_type>
                    executor_traits;
                typedef typename hpx::util::tuple<
                        FwdIter, std::size_t
                    > tuple_type;

                using hpx::util::get;
                using hpx::util::placeholders::_1;
                using hpx::util::deferred_call;

                std::vector<hpx::shared_future<Result1> > workitems;
                std::vector<hpx::future<Result2> > finalitems;
                std::vector<tuple_type> shape;

                std::list<boost::exception_ptr> errors;

                try {
                    // pre-initialize first intermediate result
                    workitems.push_back(make_ready_future(std::forward<T>(init)));

                    HPX_ASSERT(count > 0);
                    FwdIter first_ = first;
                    std::size_t test_chunk_size = count / 100;

                    // estimate a chunk size based on number of cores used
                    shape = get_bulk_iteration_shape(policy, workitems, f1,
                        first, count, 1);

                    // schedule every chunk on a separate thread
                    workitems.reserve(shape.size() + 1);
                    finalitems.reserve(shape.size());

                    // If the size of count was enough to warrant testing for a
                    // chunk, pre-initialize second intermediate result and
                    // start f3.
                    if (workitems.size() == 2)
                    {
                        HPX_ASSERT(workitems.size() < 3);

                        workitems[1] = dataflow(hpx::launch::sync,
                            f2, workitems[0], workitems[1]
                        );

                        finalitems.push_back(dataflow(
                            policy.executor(),
                            hpx::util::bind(
                                f3, first_, test_chunk_size, _1),
                            workitems[0], workitems[1])
                        );
                    }

                    std::size_t parts = workitems.size();

                    // Schedule first step of scan algorithm, step 2 is
                    // performed as soon as the current partition and the
                    // partition to the left is ready.
                    for(auto const& elem: shape)
                    {
                        BOOST_SCOPED_ENUM(hpx::launch) p = (parts & 0x7) ?
                            hpx::launch::sync : hpx::launch::async;

                        workitems.push_back(
                            dataflow(
                                p, f2, workitems.back(),
                                executor_traits::async_execute(
                                    policy.executor(),
                                    deferred_call(f1, get<0>(elem), get<1>(elem))
                                )
                            )
                        );

                        finalitems.push_back(
                            dataflow(
                                policy.executor(),
                                hpx::util::bind(
                                    f3, get<0>(elem), get<1>(elem), _1),
                                workitems[parts - 1], workitems[parts]
                            )
                        );

                        ++parts;
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
                return dataflow(
                    [=](std::vector<hpx::shared_future<Result1> >&& witems,
                        std::vector<hpx::future<Result2> >&& fitems) mutable
                      -> R
                    {
                        detail::handle_local_exceptions<ExPolicy
                            >::call(witems, errors);
                        detail::handle_local_exceptions<ExPolicy
                            >::call(fitems, errors);

                        return f4(std::move(witems), std::move(fitems));
                    },
                    std::move(workitems), std::move(finalitems));
            }
        };

        template <typename Executor, typename Parameters, typename R,
            typename Result1, typename Result2>
        struct static_scan_partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                    R, Result1, Result2>
          : static_scan_partitioner<parallel_task_execution_policy, R,
              Result1, Result2>
        {};

        ///////////////////////////////////////////////////////////////////////
        // ExPolicy: execution policy
        // R:        overall result type
        // Result1:  intermediate result type of first and second step
        // Result2:  intermediate result of the third step
        // PartTag:  select appropriate partitioner
        template <typename ExPolicy, typename R, typename Result1,
            typename Result2, typename Tag>
        struct scan_partitioner;

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename R, typename Result1,
            typename Result2>
        struct scan_partitioner<ExPolicy, R, Result1, Result2,
            parallel::traits::static_partitioner_tag>
        {
            template <typename FwdIter, typename T,
                typename F1, typename F2, typename F3, typename F4>
            static R call(ExPolicy policy, FwdIter first,
                std::size_t count, T && init, F1 && f1, F2 && f2, F3 && f3,
                F4 && f4)
            {
                return static_scan_partitioner<
                    ExPolicy, R, Result1, Result2>::call(
                        policy, first, count, std::forward<T>(init),
                        std::forward<F1>(f1), std::forward<F2>(f2),
                        std::forward<F3>(f3), std::forward<F4>(f4));
            }
        };

        template <typename R, typename Result1, typename Result2>
        struct scan_partitioner<parallel_task_execution_policy, R, Result1,
            Result2, parallel::traits::static_partitioner_tag>
        {
            template <typename ExPolicy, typename FwdIter, typename T,
                typename F1, typename F2, typename F3, typename F4>
            static hpx::future<R> call(ExPolicy policy, FwdIter first,
                std::size_t count, T && init, F1 && f1, F2 && f2, F3 && f3,
                F4 && f4)
            {
                return static_scan_partitioner<
                    ExPolicy, R, Result1, Result2>::call(
                        policy, first, count, std::forward<T>(init),
                        std::forward<F1>(f1), std::forward<F2>(f2),
                        std::forward<F3>(f3), std::forward<F4>(f4));
            }
        };

        template <typename Executor, typename Parameters, typename R,
            typename Result1, typename Result2>
        struct scan_partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                R, Result1, Result2, parallel::traits::static_partitioner_tag>
          : scan_partitioner<parallel_task_execution_policy, R, Result1,
                Result2, parallel::traits::static_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename R,
            typename Result1, typename Result2>
        struct scan_partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                R, Result1, Result2, parallel::traits::auto_partitioner_tag>
          : scan_partitioner<parallel_task_execution_policy, R, Result1,
                Result2, parallel::traits::auto_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename R,
            typename Result1, typename Result2>
        struct scan_partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                R, Result1, Result2, parallel::traits::default_partitioner_tag>
          : scan_partitioner<parallel_task_execution_policy, R, Result1,
                Result2, parallel::traits::static_partitioner_tag>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename R, typename Result1,
            typename Result2>
        struct scan_partitioner<ExPolicy, R, Result1,
                Result2, parallel::traits::default_partitioner_tag>
          : scan_partitioner<ExPolicy, R, Result1,
                Result2, parallel::traits::static_partitioner_tag>
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename R = void, typename Result1 = R,
        typename Result2 = void,
        typename PartTag = typename parallel::traits::extract_partitioner<
            typename hpx::util::decay<ExPolicy>::type
        >::type>
    struct scan_partitioner
      : detail::scan_partitioner<
            typename hpx::util::decay<ExPolicy>::type, R, Result1,
            Result2, PartTag>
    {};
}}}

#endif
