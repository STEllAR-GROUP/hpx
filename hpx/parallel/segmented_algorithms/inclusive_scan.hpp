//  Copyright (c) 2016 Minh-Khanh Do
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/segmented_algorithms/inclusive_scan.hpp

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHMS_INCLUSIVE_SCAN)
#define HPX_PARALLEL_SEGMENTED_ALGORITHMS_INCLUSIVE_SCAN

#include <hpx/config.hpp>

#include <hpx/traits/segmented_iterator_traits.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/inclusive_scan.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/segmented_algorithms/detail/scan.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>

#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // segmented inclusive_scan
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL
<<<<<<< HEAD
        struct merge_inclusive_scan
        {
            // adds init to each element from first to dest 
=======
        
        struct merge_inclusive_scan
        {
>>>>>>> 0fc95c01481bc3f11b5ff4c14c4817850d6ec03f
            template <typename InIter, typename OutIter, typename T, typename Op>
            OutIter operator() (InIter first, InIter last,
                OutIter dest, T init, Op && op)
            {
<<<<<<< HEAD
=======
                // add init to each element
>>>>>>> 0fc95c01481bc3f11b5ff4c14c4817850d6ec03f
                for (/* */; first != last; (void) ++first, ++dest)
                {
                    *dest = op(init, *first);
                }
                return dest;
            }
        };

        ///////////////////////////////////////////////////////////////////////
<<<<<<< HEAD

        // do inclusive scan returns result as vector
        template <typename Value>
        struct segmented_inclusive_scan_vector
            : public detail::algorithm<
                segmented_inclusive_scan_vector<Value>, Value>
=======
        template <typename Value, typename Algo>
        struct segmented_inclusive_scan_vector
            : public detail::algorithm<
                segmented_inclusive_scan_vector<Value, Algo>, Value>
>>>>>>> 0fc95c01481bc3f11b5ff4c14c4817850d6ec03f
        {
            typedef Value vector_type;

            segmented_inclusive_scan_vector()
                : segmented_inclusive_scan_vector::algorithm(
                    "segmented_inclusive_scan_vector")
            {}

            template <typename ExPolicy, typename InIter, typename Op>
            static vector_type
            sequential(ExPolicy && policy, InIter first, InIter last, Op && op)
            {
                typedef typename std::iterator_traits<InIter>::value_type value_type;

                vector_type result(std::distance(first, last));

                // use first element as init value for inclusive_scan
                if (result.size() != 0) {
                    result[0] = *first;
<<<<<<< HEAD
                    inclusive_scan<typename vector_type::iterator>().sequential(
=======
                    Algo().sequential(
>>>>>>> 0fc95c01481bc3f11b5ff4c14c4817850d6ec03f
                        std::forward<ExPolicy>(policy), first+1, last, result.begin()+1,
                        std::forward<value_type>(*first), std::forward<Op>(op));
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
<<<<<<< HEAD
 
=======
                
>>>>>>> 0fc95c01481bc3f11b5ff4c14c4817850d6ec03f
                typedef util::detail::algorithm_result<ExPolicy, vector_type> result;

                vector_type res(std::distance(first, last));

<<<<<<< HEAD
                // use first element as the init value for inclusive_scan
                if (res.size() != 0) {
                    res[0] = *first;
                }
 
                return result::get(
                    dataflow([=, &op](vector_type r) {
                        inclusive_scan<typename vector_type::iterator>().parallel(
=======
                if (res.size() != 0) {
                    res[0] = *first;
                }
                
                return result::get(
                    dataflow([=, &op](vector_type r) {
                        Algo().parallel(
>>>>>>> 0fc95c01481bc3f11b5ff4c14c4817850d6ec03f
                            hpx::parallel::par, first+1, last, r.begin()+1,
                            std::forward<value_type>(*first), std::forward<Op>(op));
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
        segmented_inclusive_scan_seq(ExPolicy && policy, SegIter first,
            SegIter last, OutIter dest, T init, Op && op, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<OutIter> traits_out;

            return segmented_scan_seq<
                inclusive_scan<typename traits_out::local_raw_iterator>>(
                std::forward<ExPolicy>(policy),
                first, last, dest, std::move(init), std::forward<Op>(op),
                std::true_type());
        }

        // sequential non-segmented OutIter implementation
        template <typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_inclusive_scan_seq(ExPolicy && policy, SegIter first,
            SegIter last, OutIter dest, T init, Op && op, std::false_type)
        {
            typedef std::vector<T> vector_type;

            return segmented_scan_seq_non<
<<<<<<< HEAD
                segmented_inclusive_scan_vector<vector_type>>(
                    std::forward<ExPolicy>(policy),
                    first, last, dest, std::move(init), std::forward<Op>(op),
                    merge_inclusive_scan(),
                    // new init value is last element from 
                    // segmented_incluisve_scan_vector + last init value
=======
                segmented_inclusive_scan_vector<vector_type,
                    inclusive_scan<typename vector_type::iterator>>
                >(
                    std::forward<ExPolicy>(policy),
                    first, last, dest, std::move(init), std::forward<Op>(op),
                    merge_inclusive_scan(), 
>>>>>>> 0fc95c01481bc3f11b5ff4c14c4817850d6ec03f
                    [op] (vector_type v, T val) {
                        return op(v.back(), val);
                    }
                );
        }

        ///////////////////////////////////////////////////////////////////////
        // parallel implementation

        // parallel segmented OutIter implementation
        template <typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_inclusive_scan_par(ExPolicy && policy, SegIter first,
            SegIter last, OutIter dest, T init, Op && op, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<OutIter> traits_out;

            return segmented_scan_par<
                inclusive_scan<typename traits_out::local_raw_iterator>>(
                std::forward<ExPolicy>(policy),
                first, last, dest, std::move(init), std::forward<Op>(op),
                std::true_type());
        }


        // parallel non-segmented OutIter implementation
        template <typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_inclusive_scan_par(ExPolicy && policy, SegIter first,
            SegIter last, OutIter dest, T init, Op && op, std::false_type)
        {
            typedef std::vector<T> vector_type;

            return segmented_scan_par_non<
<<<<<<< HEAD
                segmented_inclusive_scan_vector<vector_type>>(
                    std::forward<ExPolicy>(policy),
                    first, last, dest, std::move(init), std::forward<Op>(op),
                    merge_inclusive_scan(),
                    // last T of scan is in the back
=======
                segmented_inclusive_scan_vector<vector_type,
                    inclusive_scan<typename vector_type::iterator>>
                >(
                    std::forward<ExPolicy>(policy),
                    first, last, dest, std::move(init), std::forward<Op>(op),
                    merge_inclusive_scan(),
>>>>>>> 0fc95c01481bc3f11b5ff4c14c4817850d6ec03f
                    [] (vector_type v) {
                        return v.back();
                    }
                );
        }

        ///////////////////////////////////////////////////////////////////////
        // sequential remote implementation
        template <typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_inclusive_scan(ExPolicy && policy, SegIter first,
            SegIter last, OutIter dest, T init, Op && op, std::true_type)
        {
            typedef typename hpx::traits::segmented_iterator_traits<OutIter>
                ::is_segmented_iterator is_out_seg;

            // check if OutIter is segmented in the same way as SegIter
            if (is_segmented_the_same(first, last, dest, is_out_seg()))
            {
                return segmented_inclusive_scan_seq(
                    std::forward<ExPolicy>(policy),
                    first, last, dest, std::move(init), std::forward<Op>(op),
                    is_out_seg());
            }
            else
            {
                return segmented_inclusive_scan_seq(
                    std::forward<ExPolicy>(policy),
                    first, last, dest, std::move(init), std::forward<Op>(op),
                    std::false_type());
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // parallel remote implementation
        template <typename ExPolicy, typename SegIter, typename OutIter,
            typename T, typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        segmented_inclusive_scan(ExPolicy && policy, SegIter first,
            SegIter last, OutIter dest, T init, Op && op, std::false_type)
        {
            typedef typename hpx::traits::segmented_iterator_traits<OutIter>
                ::is_segmented_iterator is_out_seg;

            // check if OutIter is segmented in the same way as SegIter
            if (is_segmented_the_same(first, last, dest, is_out_seg()))
            {
                return segmented_inclusive_scan_par(
                    std::forward<ExPolicy>(policy),
                    first, last, dest, std::move(init), std::forward<Op>(op),
                    is_out_seg());
            }
            else
            {
                return segmented_inclusive_scan_par(
                    std::forward<ExPolicy>(policy),
                    first, last, dest, std::move(init), std::forward<Op>(op),
                    std::false_type());
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename InIter, typename OutIter, typename T,
            typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        inclusive_scan_(ExPolicy&& policy, InIter first, InIter last, OutIter dest,
            T init, Op && op, std::true_type)
        {
            typedef parallel::is_sequential_execution_policy<ExPolicy> is_seq;

            if (first == last)
                return util::detail::algorithm_result<
                    ExPolicy, OutIter>::get(std::move(dest));

            return segmented_inclusive_scan(
                std::forward<ExPolicy>(policy),
                first, last, dest, std::move(init), std::forward<Op>(op), is_seq());
        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename InIter, typename OutIter, typename T,
            typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        inclusive_scan_(ExPolicy&& policy, InIter first, InIter last, OutIter dest,
            T init, Op && op, std::true_type);

        /// \endcond
    }
}}}
#endif
