//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/search.hpp

#if !defined(HPX_PARALLEL_ALGORITHMS_SEARCH_NOV_9_2014_0317PM)
#define HPX_PARALLEL_ALGORITHMS_SEARCH_NOV_9_2014_0317PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/algorithm_result.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/algorithms/detail/is_negative.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx {namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // search
    namespace detail
    {
        /// \cond NOINTERNAL
        template<typename FwdIter>
        struct search: public detail::algorithm<search<FwdIter>, FwdIter>
        {
            search()
                : search::algorithm("search")
            {}

            template <typename ExPolicy, typename FwdIter2, typename Pred>
            static FwdIter
            sequential(ExPolicy const&, FwdIter first, FwdIter last, FwdIter2 s_first,
                FwdIter2 s_last, Pred && op)
            {
                return std::search(first, last, s_first, s_last, op);
            }

            template <typename ExPolicy, typename FwdIter2, typename Pred>
            static typename detail::algorithm_result<ExPolicy, FwdIter>::type
            parallel(ExPolicy const& policy, FwdIter first, FwdIter last,
                FwdIter2 s_first, FwdIter2 s_last, Pred && op)
            {
                typedef typename std::iterator_traits<FwdIter>::reference reference;
                typedef typename std::iterator_traits<FwdIter>::difference_type
                    difference_type;
                typedef typename std::iterator_traits<FwdIter2>::difference_type
                    s_difference_type;
                typedef detail::algorithm_result<ExPolicy, FwdIter> result;

                s_difference_type diff = std::distance(s_first, s_last);
                if (diff <= 0)
                    return result::get(std::move(last));

                difference_type count = std::distance(first, last);
                if (diff > count)
                    return result::get(std::move(last));

                typedef util::partitioner<ExPolicy, FwdIter, void> partitioner;

                util::cancellation_token<difference_type> tok(count);
                return partitioner::call_with_index(
                    policy, first, count-(diff-1),
                    [=](std::size_t base_idx, FwdIter it, std::size_t part_size) mutable
                    {
                        FwdIter curr = it;

                        util::loop_idx_n(
                            base_idx, it, part_size, tok,
                            [=, &tok, &curr](reference v, std::size_t i)
                            {
                                ++curr;
                                if (op(v, *s_first))
                                {
                                    difference_type local_count = 1;
                                    FwdIter2 needle = s_first;
                                    FwdIter mid = curr;

                                    for(difference_type len = 0;
                                        local_count != diff && len != count;
                                        ++local_count, ++len, ++mid)
                                    {
                                        if(*mid != *++needle)
                                            break;
                                    }

                                    if(local_count == diff)
                                        tok.cancel(i);
                                }
                            });
                    },
                    [=](std::vector<hpx::future<void> > &&) mutable -> FwdIter
                    {
                        difference_type search_res = tok.get_data();
                        if (search_res != count)
                            std::advance(first, search_res);
                        else
                            first = last;

                        return std::move(first);
                    });
            }
        };
        /// \endcond
    }

    /// Searches the range [first, last) for any elements in the range [s_first, s_last).
    /// Uses the operator== to compare elements.
    ///
    /// \note   Complexity: at most (S*N) comparisons where
    ///         \a S = distance(s_first, s_last) and
    ///         \a N = distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param s_first      Refers to the beginning of the sequence of elements
    ///                     the algorithm will be searching for.
    /// \param s_last       Refers to the end of the sequence of elements of
    ///                     the algorithm will be searching for.
    ///
    /// The comparison operations in the parallel \a search algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a search algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a search algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type \a task_execution_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a search algorithm returns an iterator to the beginning of
    ///           the first subsequence [s_first, s_last) in range [first, last).
    ///           If the length of the subsequence [s_first, s_last) is greater
    ///           than the length of the range [first, last), \a last is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, \a last is also returned.
    ///
    template <typename ExPolicy, typename FwdIter, typename FwdIter2>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, FwdIter>::type
    >::type
    search(ExPolicy && policy, FwdIter first, FwdIter last,
        FwdIter2 s_first, FwdIter2 s_last)
    {
        typedef typename std::iterator_traits<FwdIter>::iterator_category
            iterator_category;
        typedef typename std::iterator_traits<FwdIter2>::iterator_category
            s_iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::input_iterator_tag, iterator_category
            >::value),
            "Requires at least input iterator.");

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::input_iterator_tag, s_iterator_category
            >::value),
            "Subsequence requires at least forward iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::search<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, s_first, s_last, detail::equal_to());
    }

    /// Searches the range [first, last) for any elements in the range [s_first, s_last).
    /// Uses a provided predicate to compare elements.
    ///
    /// \note   Complexity: at most (S*N) comparisons where
    ///         \a S = distance(s_first, s_last) and
    ///         \a N = distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter      The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam FwdIter2     The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param s_first      Refers to the beginning of the sequence of elements
    ///                     the algorithm will be searching for.
    /// \param s_last       Refers to the end of the sequence of elements of
    ///                     the algorithm will be searching for.
    /// \param op           Refers to the binary predicate which returns true if the
    ///                     elements should be treated as equal. the signature of
    ///                     the function should be equivalent to
    ///                     \code bool pred(const Type1 &a, const Type2 &b); \endcode
    ///
    /// The comparison operations in the parallel \a search algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a search algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a search algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type \a task_execution_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a search algorithm returns an iterator to the beginning of
    ///           the first subsequence [s_first, s_last) in range [first, last).
    ///           If the length of the subsequence [s_first, s_last) is greater
    ///           than the length of the range [first, last), \a last is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, \a last is also returned.
    ///
    template <typename ExPolicy, typename FwdIter, typename FwdIter2,
        typename Pred>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, FwdIter>::type
    >::type
    search(ExPolicy && policy, FwdIter first, FwdIter last,
        FwdIter2 s_first, FwdIter2 s_last, Pred && op)
    {
        typedef typename std::iterator_traits<FwdIter>::iterator_category
            iterator_category;
        typedef typename std::iterator_traits<FwdIter2>::iterator_category
            s_iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::input_iterator_tag, iterator_category
            >::value),
            "Requires at least input iterator.");

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::input_iterator_tag, s_iterator_category
            >::value),
            "Subsequence requires at least forward iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::search<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, s_first, s_last, std::forward<Pred>(op));
    }

    ///////////////////////////////////////////////////////////////////////////
    // search_n
    namespace detail
    {
        /// \cond NOINTERNAL
        template<typename FwdIter>
        struct search_n: public detail::algorithm<search_n<FwdIter>, FwdIter>
        {
            search_n()
                : search_n::algorithm("search_n")
            {}

            template <typename ExPolicy, typename FwdIter2, typename Pred>
            static FwdIter
            sequential(ExPolicy const&, FwdIter first, std::size_t count,
                FwdIter2 s_first, FwdIter2 s_last, Pred && op)
            {
                return std::search(first, std::next(first, count),
                    s_first, s_last, op);
            }

            template <typename ExPolicy, typename FwdIter2, typename Pred>
            static typename detail::algorithm_result<ExPolicy, FwdIter>::type
            parallel(ExPolicy const& policy, FwdIter first, std::size_t count,
                FwdIter2 s_first, FwdIter2 s_last, Pred && op)
            {
                typedef typename std::iterator_traits<FwdIter>::reference reference;
                typedef typename std::iterator_traits<FwdIter>::difference_type
                    difference_type;
                typedef typename std::iterator_traits<FwdIter2>::difference_type
                    s_difference_type;
                typedef detail::algorithm_result<ExPolicy, FwdIter> result;

                s_difference_type diff = std::distance(s_first, s_last);
                if (diff <= 0)
                    return result::get(std::move(first));

                if (diff > s_difference_type(count))
                    return result::get(std::move(first));

                typedef util::partitioner<ExPolicy, FwdIter, void> partitioner;

                util::cancellation_token<difference_type> tok(count);
                return partitioner::call_with_index(
                    policy, first, count-(diff-1),
                    [=](std::size_t base_idx, FwdIter it, std::size_t part_size) mutable
                    {
                        FwdIter curr = it;

                        util::loop_idx_n(
                            base_idx, it, part_size, tok,
                            [=, &tok, &curr](reference v, std::size_t i)
                            {
                                ++curr;
                                if (op(v, *s_first))
                                {
                                    difference_type local_count = 1;
                                    FwdIter2 needle = s_first;
                                    FwdIter mid = curr;

                                    for(difference_type len = 0;
                                        local_count != diff && len != difference_type(count);
                                        ++local_count, ++len, ++mid)
                                    {
                                        if(*mid != *++needle)
                                            break;
                                    }

                                    if(local_count == diff)
                                        tok.cancel(i);
                                }
                            });
                    },
                    [=](std::vector<hpx::future<void> > &&) mutable -> FwdIter
                    {
                        difference_type search_res = tok.get_data();
                        if(search_res != s_difference_type(count))
                            std::advance(first, search_res);

                        return std::move(first);
                    });
            }
        };
        /// \endcond
    }

    /// Searches the range [first, first+count) for any elements in the range [s_first, s_last).
    /// Uses the operator== to compare elements.
    ///
    /// \note   Complexity: at most (S*N) comparisons where
    ///         \a S = distance(s_first, s_last) and
    ///         \a N = count.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter      The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam FwdIter2     The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param count        Refers to the range of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param s_first      Refers to the beginning of the sequence of elements
    ///                     the algorithm will be searching for.
    /// \param s_last       Refers to the end of the sequence of elements of
    ///                     the algorithm will be searching for.
    ///
    /// The comparison operations in the parallel \a search_n algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a search_n algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a search_n algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type \a task_execution_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a search_n algorithm returns an iterator to the beginning of
    ///           the first subsequence [s_first, s_last) in range [first, first+count).
    ///           If the length of the subsequence [s_first, s_last) is greater
    ///           than the length of the range [first, first+count), \a first is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, \a first is also returned.
    ///
    template <typename ExPolicy, typename FwdIter, typename FwdIter2>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, FwdIter>::type
    >::type
    search_n(ExPolicy && policy, FwdIter first, std::size_t count,
        FwdIter2 s_first, FwdIter2 s_last)
    {
        typedef typename std::iterator_traits<FwdIter>::iterator_category
            iterator_category;
        typedef typename std::iterator_traits<FwdIter2>::iterator_category
            s_iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::input_iterator_tag, iterator_category
            >::value),
            "Requires at least input iterator.");

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::input_iterator_tag, s_iterator_category
            >::value),
            "Subsequence requires at least forward iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::search_n<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, count, s_first, s_last, detail::equal_to());
    }

    /// Searches the range [first, last) for any elements in the range [s_first, s_last).
    /// Uses a provided predicate to compare elements.
    ///
    /// \note   Complexity: at most (S*N) comparisons where
    ///         \a S = distance(s_first, s_last) and
    ///         \a N = count.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter      The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam FwdIter2     The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param count        Refers to the range of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param s_first      Refers to the beginning of the sequence of elements
    ///                     the algorithm will be searching for.
    /// \param s_last       Refers to the end of the sequence of elements of
    ///                     the algorithm will be searching for.
    /// \param op           Refers to the binary predicate which returns true if the
    ///                     elements should be treated as equal. the signature of
    ///                     the function should be equivalent to
    ///                     \code bool pred(const Type1 &a, const Type2 &b); \endcode
    ///
    /// The comparison operations in the parallel \a search_n algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a search_n algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a search_n algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type \a task_execution_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a search_n algorithm returns an iterator to the beginning of
    ///           the last subsequence [s_first, s_last) in range [first, first+count).
    ///           If the length of the subsequence [s_first, s_last) is greater
    ///           than the length of the range [first, first+count), \a first is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, \a first is also returned.
    ///
    template <typename ExPolicy, typename FwdIter, typename FwdIter2,
        typename Pred>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, FwdIter>::type
    >::type
    search_n(ExPolicy && policy, FwdIter first, std::size_t count,
        FwdIter2 s_first, FwdIter2 s_last, Pred && op)
    {
        typedef typename std::iterator_traits<FwdIter>::iterator_category
            iterator_category;
        typedef typename std::iterator_traits<FwdIter2>::iterator_category
            s_iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::input_iterator_tag, iterator_category
            >::value),
            "Requires at least input iterator.");

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::input_iterator_tag, s_iterator_category
            >::value),
            "Subsequence requires at least forward iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::search_n<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, count, s_first, s_last, std::forward<Pred>(op));
    }
}}}

#endif
