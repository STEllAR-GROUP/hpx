//  Copyright (c) 2016 Minh-Khanh Do
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/segmented_algorithms/detail/segmented_scan_partitioner.hpp

#if !defined(HPX_PARALLEL_SEGMENTED_SCAN_PARTITIONER)
#define HPX_PARALLEL_SEGMENTED_SCAN_PARTITIONER

#include <hpx/config.hpp>

#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // ExPolicy: execution policy
        // Result:   overall result type
        // Algo:     algorithm type (inclusive_scan or exclusive_scan)
        // PartTag:  select appropriate partitioner
        template <typename ExPolicy, typename Result, typename Algo, typename Tag>
        struct segmented_scan_partitioner;

        template <typename ExPolicy, typename Result, typename Algo>
        struct segmented_scan_partitioner<ExPolicy, Result, Algo,
            parallel::traits::static_partitioner_tag>
        {
            template <typename ExPolicy_, typename InIter, typename Op>
            static Result call(ExPolicy_ && policy,
                InIter first, InIter last, Op && op)
            {
                typedef typename std::iterator_traits<InIter>::value_type value_type;

                Result result(std::distance(first, last));

                if (result.size() != 0) {
                    result[0] = *first;

                    Algo::parallel(policy, first+1, last, result.begin()+1,
                        std::forward<value_type>(*first), op);
                }

                return result;
            }
        };

        template <typename Result, typename Algo>
        struct segmented_scan_partitioner<parallel_task_execution_policy, Result, Algo,
            parallel::traits::static_partitioner_tag>
        {
            template <typename ExPolicy, typename InIter, typename Op>
            static hpx::future<Result> call(ExPolicy && policy,
                InIter first, InIter last, Op && op)
            {
                typedef typename std::iterator_traits<InIter>::value_type value_type;

                Result result(std::distance(first, last));
                if (result.size() != 0) {
                    result[0] = *first;
                }

                return dataflow(
                    [=]() mutable
                    {
                        Algo::parallel(policy,
                            first+1, last, result.begin()+1,
                            std::forward<value_type>(*first),
                            std::forward<Op>(op)).wait();

                        return result;
                    });
            }
        };

        template <typename Executor, typename Parameters, typename Result,
            typename Algo>
        struct segmented_scan_partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                Result, Algo, parallel::traits::static_partitioner_tag>
          : segmented_scan_partitioner<parallel_task_execution_policy, Result,
                Algo, parallel::traits::static_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename Result,
            typename Algo>
        struct segmented_scan_partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                Result, Algo, parallel::traits::auto_partitioner_tag>
          : segmented_scan_partitioner<parallel_task_execution_policy, Result,
                Algo, parallel::traits::auto_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename Result,
            typename Algo>
        struct segmented_scan_partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                Result, Algo, parallel::traits::default_partitioner_tag>
          : segmented_scan_partitioner<parallel_task_execution_policy, Result,
                Algo, parallel::traits::static_partitioner_tag>
        {};


        ///////////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Result, typename Algo>
        struct segmented_scan_partitioner<ExPolicy, Result, Algo,
                parallel::traits::default_partitioner_tag>
            : segmented_scan_partitioner<ExPolicy, Result, Algo,
                parallel::traits::static_partitioner_tag>
        {};

    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Result, typename Algo,
        typename PartTag = typename parallel::traits::extract_partitioner<
            typename hpx::util::decay<ExPolicy>::type
        >::type>
    struct segmented_scan_partitioner
        : detail::segmented_scan_partitioner<
            typename hpx::util::decay<ExPolicy>::type, Result, Algo, PartTag>
    {};
}}}

#endif
