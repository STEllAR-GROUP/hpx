//  Copyright (c) 2016 Minh-Khanh Do
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/segmented_algorithms/exclusive_scan.hpp

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHMS_EXCLUSIVE_SCAN)
#define HPX_PARALLEL_SEGMENTED_ALGORITHMS_EXCLUSIVE_SCAN

#include <hpx/config.hpp>

#include <hpx/traits/segmented_iterator_traits.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/exclusive_scan.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/segmented_algorithms/detail/scan.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>

#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1
{
    ///////////////////////////////////////////////////////////////////////////
    // segmented exclusive_scan
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        struct merge_exclusive_scan
        {
            // adds init to each element except for the first one
            template <typename InIter, typename OutIter, typename T, typename Op>
            OutIter operator() (InIter first, InIter last,
                OutIter dest, T init, Op && op)
            {
                *dest = init;
                for (++first, ++dest; first != last; (void) ++first, ++dest)
                {
                    *dest = op(init, *first);
                }
                return dest;
            }
        };

        ///////////////////////////////////////////////////////////////////////

        // do exclusive scan returns result as vector
        // first element of result vector is last T of scan
        // (can be used to transfer to the next partition)

        // first element can be use because it will be
        // overwritten by the last T of the previous partition
        template <typename Value>
        struct segmented_exclusive_scan_vector
            : public detail::algorithm<
                segmented_exclusive_scan_vector<Value>, Value>
        {
            typedef Value vector_type;

            segmented_exclusive_scan_vector()
                : segmented_exclusive_scan_vector::algorithm(
                    "segmented_exclusive_scan_vector")
            {}

            template <typename ExPolicy, typename InIter, typename Op>
            static vector_type
            sequential(ExPolicy && policy, InIter first, InIter last, Op && op)
            {
                typedef typename std::iterator_traits<InIter>::value_type value_type;

                vector_type result(std::distance(first, last));

                // use first element to save the last T of scan
                if (result.size() != 0) {
                    exclusive_scan<typename vector_type::iterator>().sequential(
                        std::forward<ExPolicy>(policy), first+1, last, result.begin()+1,
                        std::forward<value_type>(*first), std::forward<Op>(op));
                    result[0] = op(result.back(), *(last-1));
                }
                return result;
            }

            template <typename ExPolicy, typename FwdIter, typename Op>
            static typename util::detail::algorithm_result<
                ExPolicy, vector_type
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last, Op && op)
            {
                typedef typename std::iterator_traits<FwdIter>::value_type value_type;

                typedef util::detail::algorithm_result<ExPolicy, vector_type> result;

                vector_type res(std::distance(first, last));

                // use first element to save last T of scan
                return result::get(
                    dataflow([=](vector_type r) {
                        exclusive_scan<typename vector_type::iterator>().parallel(
                            hpx::parallel::execution::par,
                            first+1, last, r.begin()+1,
                            std::forward<value_type>(*first), op);
                        r[0] = op(r.back(), *(last-1));
                        return r;

                    }, std::move(res)));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // sequential implementation

        // sequential segmented OutIter implementation
        template <typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_exclusive_scan_seq(ExPolicy && policy, SegIter first,
            SegIter last, OutIter dest, T const& init, Op && op, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<OutIter> traits_out;
            return segmented_scan_seq<
                exclusive_scan<typename traits_out::local_raw_iterator>>(
                std::forward<ExPolicy>(policy),
                first, last, dest, init, std::forward<Op>(op),
                std::true_type());
        }

        // sequential non segmented OutIter implementation
        template <typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_exclusive_scan_seq(ExPolicy && policy, SegIter first,
            SegIter last, OutIter dest, T const& init, Op && op, std::false_type)
        {
            typedef std::vector<T> vector_type;

            return segmented_scan_seq_non<
                segmented_exclusive_scan_vector<vector_type>>(
                    std::forward<ExPolicy>(policy),
                    first, last, dest, init, std::forward<Op>(op),
                    merge_exclusive_scan(),
                    // new init value is first element from
                    // segmented_excluisve_scan_vector + last init value
                    [op] (vector_type v, T val) {
                        return op(v.front(), val);
                    }
                );
        }

        ///////////////////////////////////////////////////////////////////////
        // parallel implementation

        // parallel segmented OutIter implementation
        template <typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_exclusive_scan_par(ExPolicy && policy, SegIter first,
            SegIter last, OutIter dest, T const& init, Op && op, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<OutIter> traits_out;

            return segmented_scan_par<
                exclusive_scan<typename traits_out::local_raw_iterator>>(
                std::forward<ExPolicy>(policy),
                first, last, dest, init, std::forward<Op>(op),
                std::true_type());
        }

        // parallel non-segmented OutIter implementation
        template <typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_exclusive_scan_par(ExPolicy && policy, SegIter first,
            SegIter last, OutIter dest, T const& init, Op && op, std::false_type)
        {
            typedef std::vector<T> vector_type;

            return segmented_scan_par_non<
                segmented_exclusive_scan_vector<vector_type>>(
                    std::forward<ExPolicy>(policy),
                    first, last, dest, init, std::forward<Op>(op),
                    merge_exclusive_scan(),
                    // last T of scan is on the front
                    // see segmented_exclusive_scan_vector
                    [] (vector_type v) {
                        return v.front();
                    }
                );
        }

        ///////////////////////////////////////////////////////////////////////
        // sequential remote implementation
        template <typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_exclusive_scan(ExPolicy && policy, SegIter first,
            SegIter last, OutIter dest, T const& init, Op && op, std::true_type)
        {
            typedef typename hpx::traits::segmented_iterator_traits<OutIter>
                ::is_segmented_iterator is_out_seg;

            // check if OutIter is segmented in the same way as SegIter
            if (is_segmented_the_same(first, last, dest, is_out_seg()))
            {
                return segmented_exclusive_scan_seq(
                    std::forward<ExPolicy>(policy),
                    first, last, dest, init, std::forward<Op>(op),
                    is_out_seg());
            }
            else
            {
                return segmented_exclusive_scan_seq(
                    std::forward<ExPolicy>(policy),
                    first, last, dest, init, std::forward<Op>(op),
                    std::false_type());
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // parallel remote implementation
        template <typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_exclusive_scan(ExPolicy && policy, SegIter first,
            SegIter last, OutIter dest, T const& init, Op && op, std::false_type)
        {
            typedef typename hpx::traits::segmented_iterator_traits<OutIter>
                ::is_segmented_iterator is_out_seg;

            if (is_segmented_the_same(first, last, dest, is_out_seg()))
            {
                return segmented_exclusive_scan_par(
                    std::forward<ExPolicy>(policy),
                    first, last, dest, init, std::forward<Op>(op),
                    is_out_seg());
            }
            else
            {
                return segmented_exclusive_scan_par(
                    std::forward<ExPolicy>(policy),
                    first, last, dest, init, std::forward<Op>(op),
                    std::false_type());
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename InIter, typename OutIter, typename T,
            typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        exclusive_scan_(ExPolicy&& policy, InIter first, InIter last, OutIter dest,
            T const& init, Op && op, std::true_type)
        {
            typedef parallel::execution::is_sequential_execution_policy<
                    ExPolicy
                > is_seq;

            if (first == last)
                return util::detail::algorithm_result<
                    ExPolicy, OutIter>::get(std::move(dest));

            return segmented_exclusive_scan(
                std::forward<ExPolicy>(policy),
                first, last, dest, init, std::forward<Op>(op), is_seq());
        }

        ///////////////////////////////////////////////////////////////////////
        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename InIter, typename OutIter, typename T,
            typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        exclusive_scan_(ExPolicy&& policy, InIter first, InIter last, OutIter dest,
            T const& init, Op && op, std::true_type);


        /// \endcond
    }
}}}
#endif
