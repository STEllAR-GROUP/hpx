//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/exclusive_scan.hpp

#if !defined(HPX_PARALLEL_ALGORITHM_PREFIX_SCAN_NOV_2015)
#define HPX_PARALLEL_ALGORITHM_PREFIX_SCAN_NOV_2015

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/unwrapped.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/inclusive_scan.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/lcos/when_all.hpp>

#include <algorithm>
#include <numeric>
#include <iterator>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx {
namespace parallel {
HPX_INLINE_NAMESPACE(v1)
    {
    ///////////////////////////////////////////////////////////////////////////
    // exclusive_scan
    namespace detail
    {
    /// \cond NOINTERNAL
    //---------------------------------------------------------------------
    // sequential exclusive_scan using iterators for start/end
    //---------------------------------------------------------------------
    template<typename InIter, typename OutIter, typename T, typename Op>
    OutIter sequential_scan(InIter first, InIter last,
        OutIter dest, T init, Op && op)
    {
        T temp = init;
        for (/* */; first != last; (void) ++first, ++dest) {
            init = op(init, *first);
            *dest = temp;
            temp = init;
        }
        return dest;
    }

    //---------------------------------------------------------------------
    // sequential exclusive_scan using count instead of end
    //---------------------------------------------------------------------
    template<typename InIter, typename T, typename Op>
    T sequential_scan_n(InIter first, std::size_t count,
        T init, Op && op)
    {
        T temp = init;
        for (/* */; count-- != 0; (void) ++first) {
            init = op(init, *first);
            temp = init;
        }
        return init;
    }

    //---------------------------------------------------------------------
    // sequential update, applies V to each value in the sequence
    //---------------------------------------------------------------------
    template<typename InIter, typename OutIter, typename T, typename Op>
    T sequential_update_n(InIter first, OutIter dest, std::size_t count,
        T value, Op && op)
    {
        T init = value;
        T temp = init;
        for (/* */; count-- != 0; (void) ++first, ++dest) {
            init = op(init, *first);
            *dest = temp;
            temp = init;
        }
        return init;
    }

    ///////////////////////////////////////////////////////////////////////
    template<typename OutIter>
    struct parallel_scan_struct
        : public hpx::parallel::v1::detail::algorithm<
          parallel_scan_struct<OutIter>, OutIter>
    {
        parallel_scan_struct()
        :
            parallel_scan_struct::algorithm("parallel_scan")
        {
        }

        template<typename ExPolicy, typename InIter, typename T, typename Op>
        static OutIter
        sequential(ExPolicy, InIter first, InIter last,
            OutIter dest, T && init, Op && op)
        {
            return detail::sequential_scan(first, last, dest,
                std::forward < T > (init), std::forward < Op > (op));
        }

        template<typename ExPolicy, typename FwdIter, typename T, typename Op>
        static typename hpx::parallel::util::detail::algorithm_result<
                ExPolicy, OutIter
        >::type
        parallel(ExPolicy policy, FwdIter first, FwdIter last,
                 OutIter dest, T && init, Op && op)
        {
            typedef util::detail::algorithm_result<ExPolicy, OutIter> result;
            typedef typename std::iterator_traits<FwdIter>::difference_type
                difference_type;

            if (first == last)
                return result::get(std::move(dest));

            // --------------------------------------------------------------------------
            // Parallel prefix sum /scan algorithm, after algorithm described in [1]
            // available here
            // http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
            //
            // 1. Hubert Nguyen. 2007. Gpu Gems 3 (First ed.).
            // Addison-Wesley Professional.
            // --------------------------------------------------------------------------

            difference_type count = std::distance(first, last);
            OutIter final_dest = dest;
            std::advance(final_dest, count);

            int cores = executor_information_traits<typename ExPolicy::executor_type>::
              processing_units_count(policy.executor(), policy.parameters());

            int n_chunks, log2N;
            std::size_t chunk_size;
            if (cores>1) {
              const std::size_t sequential_scan_limit = 32768;
              // we want 2^N chunks of data, so find a good N
              log2N = 0;
              while (cores >>= 1) ++log2N;
              n_chunks = (1 << log2N);
              chunk_size = std::max(1, int(count / n_chunks));
              while (n_chunks > 0 && chunk_size < sequential_scan_limit) {
                chunk_size <<= 1;
                n_chunks >>= 1;
                log2N -= 1;
              }
            }
            if (n_chunks < 2 || cores == 1) {
              return sequential(policy, first, last, dest, std::forward < T >(init),
                std::forward < Op >(op));
            }

            // --------------------------
            // Upsweep:
            // --------------------------
            // because our list might not be an exact power of 2 and we wish to run
            // 2^N tasks/chunks we will first create a list of iterator start/end points
            // for a 2^N list as indices into our original input data.
            // Then we will do the scan algorithm using the 2^N items as our base input.
            // This first loop spawns 2^N sequential scans on the initial lists
            typedef std::tuple<FwdIter, OutIter, std::size_t, T> chunk_info;
            std::vector < chunk_info > work_chunks;
            std::vector<hpx::future<T> > work_items;
            std::vector<hpx::future<void> > work_partials;
            work_chunks.reserve(n_chunks);
            work_items.reserve(n_chunks);
            FwdIter it2, it1 = first;
            for (int k = 0; k < n_chunks; ++k) {
                if (k < n_chunks - 1) {
                    it2 = it1 + chunk_size;
                }
                else { // last chunk might not be exactly the right size
                    chunk_size = std::distance(it2, last);
                    it2 = last;
                }
                // start and end of chunk of our input array
                work_chunks.push_back(
                    std::make_tuple(it1, dest, chunk_size, *(it1 + chunk_size - 1)));
                // spawn a task to do a sequential scan on this chunk
                work_items.push_back(
                    std::move(
                        hpx::async(&sequential_scan_n<FwdIter, T, Op>,
                            it1, chunk_size, (T()), std::forward < Op > (op))));
                it1 = it2;
                std::advance(dest, chunk_size);
            }
            //
            // do a tree of combine operations on the result of the 2^N sequence results
            //
            using hpx::lcos::local::dataflow;
            //
            hpx::when_all(std::move(work_items)).then(
                [=, &work_chunks](hpx::future<std::vector<hpx::future<T> > > fv) {
                // move the items into a local vector
                std::vector<hpx::future<T> > items(fv.get());
                for (int c = 0; c < n_chunks; ++c) {
                    std::get<3>(work_chunks[c]) = items[c].get();
                }
                for (int d = 0; d < log2N; ++d) {
                    int d_2 = (1 << d);
                    int dp1_2 = (2 << d);
                    for (int k = 0, f = 0; k < n_chunks; k += dp1_2, f += 2) {
                        int i1 = k + d_2 - 1;
                        int i2 = k + dp1_2 - 1;
                        T sum_left = std::get<3>(work_chunks[i1]);
                        T sum_right = std::get<3>(work_chunks[i2]);
                        std::get<3>(work_chunks[k + dp1_2 - 1]) = op(sum_left, sum_right);
                    }
                }
            }
            ).get();

            // --------------------------
            // Downsweep:
            // --------------------------
            //
            std::get < 3 > (work_chunks.back()) = T();
            for (int d = log2N - 1; d >= 0; --d) {
                int d_2 = (1 << d);
                int dp1_2 = (2 << d);
                int steps = n_chunks >> (d + 1);
                for (int k = 0; k < n_chunks - 1; k += dp1_2) {
                    T temp = std::get < 3 > (work_chunks[k + d_2 - 1]);
                    std::get < 3 > (work_chunks[k + d_2 - 1]) =
                        std::get < 3 > (work_chunks[k + dp1_2 - 1]);
                    std::get < 3 > (work_chunks[k + dp1_2 - 1]) =
                        op(std::get < 3 > (work_chunks[k + dp1_2 - 1]), temp);
                }
            }
            // now combine the partial sums back into the initial chunks
            for (int c = 0; c < n_chunks; ++c) {
                // spawn a task to do a sequential update on this chunk

                hpx::future<T> w1 = hpx::async(
                    &sequential_update_n<FwdIter, FwdIter, T, Op>,
                    std::get < 0 > (work_chunks[c]),
                    std::get < 1 > (work_chunks[c]),
                    std::get < 2 > (work_chunks[c]),
                    std::get < 3 > (work_chunks[c]), std::forward < Op > (op));
                work_items.push_back(std::move(w1));

            }
            hpx::wait_all(work_items);
            return final_dest;
        }
    };

    /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(binary_op, init, *first, ...,
    /// *(first + (i - result) - 1)).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a op.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    /// \tparam Op          The type of the binary function object used for
    ///                     the reduction operation.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param init         The initial value for the generalized sum.
    /// \param op           Specifies the function (or function object) which
    ///                     will be invoked for each of the values of the input
    ///                     sequence. This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it.
    ///                     The types \a Type1 and \a Ret must be
    ///                     such that an object of a type as given by the input
    ///                     sequence can be implicitly converted to any
    ///                     of those types.
    ///
    /// The reduce operations in the parallel \a exclusive_scan algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a exclusive_scan algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy_n algorithm returns a \a hpx::future<OutIter> if
    ///           the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a OutIter otherwise.
    ///           The \a exclusive_scan algorithm returns the output iterator
    ///           to the element in the destination range, one past the last
    ///           element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aK),
    ///           GENERALIZED_NONCOMMUTATIVE_SUM(op, aM, ..., aN))
    ///           where 1 < K+1 = M <= N.
    ///
    /// The difference between \a exclusive_scan and \a inclusive_scan is that
    /// \a inclusive_scan includes the ith input element in the ith sum. If
    /// \a op is not mathematically associative, the behavior of
    /// \a inclusive_scan may be non-deterministic.
    ///
    template<typename ExPolicy, typename InIter, typename OutIter, typename T,
    typename Op>
    inline typename boost::enable_if<
    is_execution_policy<ExPolicy>,
    typename util::detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    prefix_scan(ExPolicy&& policy, InIter first, InIter last, OutIter dest,
        T init, Op && op)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;
        typedef typename std::iterator_traits<OutIter>::iterator_category
            output_iterator_category;

        static_assert(
            (boost::is_base_of<std::input_iterator_tag, iterator_category>::value),
            "Requires at least input iterator.");

        static_assert(
            (boost::mpl::or_<
                boost::is_base_of<
                std::forward_iterator_tag, output_iterator_category>,
                boost::is_same<
                std::output_iterator_tag, output_iterator_category>
        >::value),
        "Requires at least output iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>,
            boost::is_same<std::output_iterator_tag, output_iterator_category>
        >::type is_seq;

        return detail::parallel_scan_struct<OutIter>().call(
            std::forward < ExPolicy > (policy), is_seq(),
            first, last, dest, std::move(init), std::forward < Op > (op));
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(+, init, *first, ..., *(first + (i - result) - 1))
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a std::plus<T>.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param init         The initial value for the generalized sum.
    ///
    /// The reduce operations in the parallel \a exclusive_scan algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a exclusive_scan algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy_n algorithm returns a \a hpx::future<OutIter> if
    ///           the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a OutIter otherwise.
    ///           The \a exclusive_scan algorithm returns the output iterator
    ///           to the element in the destination range, one past the last
    ///           element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(+, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * GENERALIZED_NONCOMMUTATIVE_SUM(+, a1, ..., aK)
    ///           + GENERALIZED_NONCOMMUTATIVE_SUM(+, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// The difference between \a exclusive_scan and \a inclusive_scan is that
    /// \a inclusive_scan includes the ith input element in the ith sum.
    ///
    template<typename ExPolicy, typename InIter, typename OutIter, typename T>
    inline typename boost::enable_if<
    is_execution_policy<ExPolicy>,
    typename util::detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    prefix_scan(ExPolicy&& policy, InIter first, InIter last, OutIter dest,
        T init)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;
        typedef typename std::iterator_traits<OutIter>::iterator_category
            output_iterator_category;

        static_assert(
            (boost::is_base_of<std::input_iterator_tag, iterator_category>::value),
            "Requires at least input iterator.");

        static_assert(
            (boost::mpl::or_<
                boost::is_base_of<
                std::forward_iterator_tag, output_iterator_category>,
                boost::is_same<
                std::output_iterator_tag, output_iterator_category>
        >::value),
        "Requires at least output iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>,
            boost::is_same<std::output_iterator_tag, output_iterator_category>
        >::type is_seq;

        return detail::parallel_scan_struct<OutIter>().call(
            std::forward < ExPolicy > (policy), is_seq(),
            first, last, dest, std::move(init), std::plus<T>());
    }
    }
}
}

#endif
