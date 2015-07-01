//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/is_partitioned.hpp

#if !defined(HPX_PARALLEL_ALGORITHMS_IS_PARTITIONED_FEB_11_2015_0331PM)
#define HPX_PARALLEL_ALGORITHMS_IS_PARTITIONED_FEB_11_2015_0331PM

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <algorithm>
#include <iterator>
#include <functional>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/type_traits/is_same.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ////////////////////////////////////////////////////////////////////////////
    // is_partitioned
    namespace detail
    {
        /// \cond NOINTERNAL

        inline bool sequential_is_partitioned(std::vector<hpx::future<bool > > && res)
        {
            std::vector<hpx::future<bool> >::iterator first = res.begin();
            std::vector<hpx::future<bool> >::iterator last = res.end();
            while (first!=last && first->get())
            {
                ++first;
            }
            if (first != last)
            {
                ++first;
                while(first != last)
                {
                    if(first->get()) return false;
                    ++first;
                }
            }
            return true;
        }

        template <typename Iter>
        struct is_partitioned:
            public detail::algorithm<is_partitioned<Iter>, bool>
        {
            is_partitioned()
                : is_partitioned::algorithm("is_partitioned")
            {}

            template<typename ExPolicy, typename Pred>
            static bool
            sequential(ExPolicy, Iter first, Iter last,
                Pred && pred)
            {
                return std::is_partitioned(first,
                    last,
                    std::forward<Pred>(pred));
            }

            template <typename ExPolicy, typename Pred>
            static typename util::detail::algorithm_result<ExPolicy, bool>::type
            parallel(ExPolicy policy, Iter first, Iter last,
                Pred && pred)
            {
                typedef typename std::iterator_traits<Iter>::reference
                    reference;
                typedef typename std::iterator_traits<Iter>::difference_type
                    difference_type;
                typedef typename util::detail::algorithm_result<ExPolicy, bool>
                    result;

                difference_type count = std::distance(first, last);
                if (count <= 1)
                    return result::get(true);

                util::cancellation_token<> tok;
                return util::partitioner<ExPolicy, bool>::call(
                    policy, first, count,
                    [pred, tok](Iter part_begin,
                        std::size_t part_count) mutable -> bool
                    {
                        bool fst_bool = pred(*part_begin);
                        if (part_count == 1)
                            return fst_bool;
                        util::loop_n(++part_begin, --part_count, tok,
                            [&fst_bool, &pred, &tok](Iter const& a) {
                                if (fst_bool != pred(*a))
                                {
                                    if(fst_bool)
                                        fst_bool = false;
                                    else
                                        tok.cancel();
                                }
                            });
                        return fst_bool;
                    },
                    [tok](std::vector<hpx::future<bool> > && results) -> bool
                    {
                        if (tok.was_cancelled()) return false;
                        return sequential_is_partitioned(std::move(results));
                    });
            }
        };
        /// \endcond
    }

    /// Determines if the range [first, last) is partitioned.
    ///
    /// \note   Complexity: at most (N) predicate evaluations where
    ///         \a N = distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam InIter     The type of the source iterators used for the
    ///                     This iterator type must meet the requirements of a
    ///                     input iterator.
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of that the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     that the algorithm will be applied to.
    /// \param pred         Refers to the binary predicate which returns a bool.
    ///                     The signature of the function should be equivalent
    ///                     to
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a InIter1 and \a InIter2 can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type1 and \a Type2 respectively
    ///
    /// The predicate operations in the parallel \a is_partitioned algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// executes in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a is_partitioned algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a is_partitioned algorithm returns a \a hpx::future<bool>
    ///           if the execution policy is of type \a task_execution_policy
    ///           and returns \a bool otherwise.
    ///           The \a is_partitioned algorithm returns true if each element
    ///           in the sequence for which pred returns true precedes those for
    ///           which pred returns false. Otherwise is_partitioned returns
    ///           false. If the range [first, last) containes less than two
    ///           elements, the function is always true.
    ///
    template <typename ExPolicy, typename InIter, typename Pred>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, bool>::type
    >::type
    is_partitioned(ExPolicy && policy, InIter first, InIter last, Pred && pred)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;
        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
             std::input_iterator_tag, iterator_category
                 >::value),
            "Requires at least input iterator.");

        typedef typename boost::mpl::or_<
            parallel::is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::is_partitioned<InIter>().call(
            std::forward<ExPolicy>(policy), is_seq(), first, last,
            std::forward<Pred>(pred));
    }

}}}

#endif
