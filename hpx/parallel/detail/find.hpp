//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/detail/find.hpp

#if !defined(HPX_PARALLEL_DETAIL_FIND_JULY_16_2014_0213PM)
#define HPX_PARALLEL_DETAIL_FIND_JULY_16_2014_0213PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/detail/algorithm_result.hpp>
#include <hpx/parallel/detail/predicates.hpp>
#include <hpx/parallel/detail/dispatch.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // find
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename InIter>
        struct find : public detail::algorithm<find<InIter>, InIter>
        {
            find()
                : find::algorithm("find")
            {}

            template <typename ExPolicy, typename T>
            static InIter
            sequential(ExPolicy const&, InIter first, InIter last, const T& val)
            {
                return std::find(first, last, val);
            }

            template <typename ExPolicy, typename T>
            static typename detail::algorithm_result<ExPolicy, InIter>::type
            parallel(ExPolicy const& policy, InIter first, InIter last,
                T const& val)
            {
                typedef typename std::iterator_traits<InIter>::iterator_category
                    category;
                typedef typename std::iterator_traits<InIter>::value_type type;

                std::size_t count = std::distance(first, last);
                util::cancellation_token<std::size_t> tok(count);

                return util::partitioner<ExPolicy, InIter, void>::call_with_index(
                    policy, first, count,
                    [val, tok](std::size_t base_idx, InIter it,
                        std::size_t part_size) mutable
                    {
                        util::loop_idx_n(
                            base_idx, it, part_size, tok,
                            [&val, &tok](type& v, std::size_t i)
                            {
                                if (v == val)
                                    tok.cancel(i);
                            });
                    },
                    [=](std::vector<hpx::future<void> > &&) mutable
                    {
                        std::size_t find_res = tok.get_data();
                        if(find_res != count)
                            std::advance(first, find_res);
                        else
                            first = last;

                        return std::move(first);
                    });
            }
        };
        /// \endcond
    }

    /// Returns the first element in the range [first, last) that is equal
    /// to value
    ///
    /// \note   Complexity: At most last - first
    ///         applications of the operator==().
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam InIter      The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam T           The type of the value to find (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param val          the value to compare the elements to
    ///
    /// The comparison operations in the parallel \a find algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find algorithm returns a \a hpx::future<InIter> if the
    ///           execution policy is of type \a task_execution_policy and
    ///           returns \a InIter otherwise.
    ///           The \a find algorithm returns the first element in the range
    ///           [first,last) that is equal to \a val.
    ///           If no such element in the range of [first,last) is equal to
    ///           \a val, then the algorithm returns \a last.
    ///
    template <typename ExPolicy, typename InIter, typename T>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, InIter>::type
    >::type
    find(ExPolicy && policy, InIter first, InIter last, T const& val)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::input_iterator_tag, iterator_category
            >::value),
            "Requires at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::find<InIter>().call(
            std::forward<ExPolicy>(policy),
            first, last, val, is_seq());
    }

    ///////////////////////////////////////////////////////////////////////////
    // find_if
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename InIter>
        struct find_if : public detail::algorithm<find_if<InIter>, InIter>
        {
            find_if()
                : find_if::algorithm("find_if")
            {}

            template <typename ExPolicy, typename F>
            static InIter
            sequential(ExPolicy const&, InIter first, InIter last, F && f)
            {
                return std::find_if(first, last, f);
            }

            template <typename ExPolicy, typename FwdIter, typename F>
            static typename detail::algorithm_result<ExPolicy, FwdIter>::type
            parallel(ExPolicy const& policy, FwdIter first, FwdIter last, F && f)
            {
                typedef typename std::iterator_traits<FwdIter>::iterator_category
                    category;
                typedef typename std::iterator_traits<FwdIter>::value_type type;

                std::size_t count = std::distance(first, last);
                util::cancellation_token<std::size_t> tok(count);

                return util::partitioner<ExPolicy, FwdIter, void>::call_with_index(
                    policy, first, count,
                    [f, tok](std::size_t base_idx, FwdIter it,
                        std::size_t part_size) mutable
                {
                    util::loop_idx_n(
                        base_idx, it, part_size, tok,
                        [&f, &tok](type& v, std::size_t i)
                    {
                        if ( f(v) )
                            tok.cancel(i);
                    });
                },
                [=](std::vector<hpx::future<void> > &&) mutable
                {
                    std::size_t find_res = tok.get_data();
                    if(find_res != count)
                        std::advance(first, find_res);
                    else
                        first = last;

                    return std::move(first);
                });
            }
        };
        /// \endcond
    }

    /// Returns the first element in the range [first, last) for which
    /// predicate \a f returns true
    ///
    /// \note   Complexity: At most last - first
    ///         applications of the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam InIter      The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param f            The unary predicate which returns true for the
    ///                     required element. The signature of the predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such
    ///                     that objects of type \a InIter can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type.
    ///
    /// The comparison operations in the parallel \a find_if algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find_if algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find_if algorithm returns a \a hpx::future<InIter> if the
    ///           execution policy is of type \a task_execution_policy and
    ///           returns \a InIter otherwise.
    ///           The \a find_if algorithm returns the first element in the range
    ///           [first,last) that satisfies the predicate \a f.
    ///           If no such element exists that satisfies the predicate f, the
    ///           algorithm returns \a last.
    ///
    template <typename ExPolicy, typename InIter, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, InIter>::type
    >::type
    find_if(ExPolicy && policy, InIter first, InIter last, F && f)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::input_iterator_tag, iterator_category
            >::value),
            "Requires at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::find_if<InIter>().call(
            std::forward<ExPolicy>(policy),
            first, last, std::forward<F>(f), is_seq());
    }

    ///////////////////////////////////////////////////////////////////////////
    // find_if_not
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename InIter>
        struct find_if_not : public detail::algorithm<find_if_not<InIter>, InIter>
        {
            find_if_not()
                : find_if_not::algorithm("find_if_not")
            {}

            template <typename ExPolicy, typename F>
            static InIter
            sequential(ExPolicy const&, InIter first, InIter last, F && f)
            {
                for (; first != last; ++first) {
                    if (!f(*first)) {
                        return first;
                    }
                }
                return last;
            }

            template <typename ExPolicy, typename FwdIter, typename F>
            static typename detail::algorithm_result<ExPolicy, FwdIter>::type
            parallel(ExPolicy const& policy, FwdIter first, FwdIter last, F && f)
            {
                typedef typename std::iterator_traits<FwdIter>::iterator_category
                    category;
                typedef typename std::iterator_traits<FwdIter>::value_type type;

                std::size_t count = std::distance(first, last);
                util::cancellation_token<std::size_t> tok(count);

                return util::partitioner<ExPolicy, FwdIter, void>::call_with_index(
                    policy, first, count,
                    [f, tok](std::size_t base_idx, FwdIter it, std::size_t part_size) mutable
                {
                    util::loop_idx_n(
                        base_idx, it, part_size, tok,
                        [&f, &tok](type& v, std::size_t i)
                    {
                        if (!f(v))
                            tok.cancel(i);
                    });

                },
                [=](std::vector<hpx::future<void> > &&) mutable
                {
                    std::size_t find_res = tok.get_data();
                    if(find_res != count)
                        std::advance(first, find_res);
                    else
                        first = last;

                    return std::move(first);
                });
            }
        };
        /// \endcond
    }

    /// Returns the first element in the range [first, last) for which
    /// predicate \a f returns false
    ///
    /// \note   Complexity: At most last - first
    ///         applications of the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam InIter      The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param f            The unary predicate which returns false for the
    ///                     required element. The signature of the predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such
    ///                     that objects of type \a InIter can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type.
    ///
    /// The comparison operations in the parallel \a find_if_not algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find_if_not algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find_if_not algorithm returns a \a hpx::future<InIter> if the
    ///           execution policy is of type \a task_execution_policy and
    ///           returns \a InIter otherwise.
    ///           The \a find_if_not algorithm returns the first element in the range
    ///           [first, last) that does \b not satisfy the predicate \a f.
    ///           If no such element exists that does not satisfy the predicate f, the
    ///           algorithm returns \a last.
    ///
    template <typename ExPolicy, typename InIter, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, InIter>::type
    >::type
    find_if_not(ExPolicy && policy, InIter first, InIter last, F && f)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::input_iterator_tag, iterator_category
            >::value),
            "Requires at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::find_if_not<InIter>().call(
            std::forward<ExPolicy>(policy),
            first, last, std::forward<F>(f), is_seq());
    }

    ///////////////////////////////////////////////////////////////////////////
    // find_end
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename FwdIter>
        struct find_end : public detail::algorithm<find_end<FwdIter>, FwdIter>
        {
            find_end()
              : find_end::algorithm("find_end")
            {}

            template <typename ExPolicy, typename FwdIter2, typename Pred>
            static FwdIter
            sequential(ExPolicy const&, FwdIter first1, FwdIter last1,
                FwdIter2 first2, FwdIter2 last2, Pred && op)
            {
                if(first2 == last2)
                    return last1;
                FwdIter result = last1;
                while(1) {
                    FwdIter new_result = std::search(first1, last1, first2, last2, op);
                    if (new_result == last1) {
                        return result;
                    } else {
                        result = new_result;
                        first1 = result;
                        ++first1;
                    }
                }
                return result;
            }

            template <typename ExPolicy, typename FwdIter2, typename Pred>
            static typename detail::algorithm_result<ExPolicy, FwdIter>::type
            parallel(ExPolicy const& policy, FwdIter first1, FwdIter last1,
                FwdIter2 first2, FwdIter2 last2, Pred && op)
            {
                typedef typename std::iterator_traits<FwdIter>::reference reference;
                typedef typename std::iterator_traits<FwdIter>::difference_type
                    difference_type;

                difference_type diff = std::distance(first2, last2);
                if (diff <= 0)
                {
                    return detail::algorithm_result<ExPolicy, FwdIter>::get(
                        std::move(last1));
                }

                difference_type count = std::distance(first1, last1);
                if (diff > count)
                {
                    return detail::algorithm_result<ExPolicy, FwdIter>::get(
                        std::move(last1));
                }

                util::cancellation_token<
                    difference_type, std::greater<difference_type>
                > tok(-1);

                return util::partitioner<ExPolicy, FwdIter, void>::call_with_index(
                    policy, first1, count-(diff-1),
                    [=](std::size_t base_idx, FwdIter it, std::size_t part_size) mutable
                    {
                        FwdIter curr = it;

                        util::loop_idx_n(
                            base_idx, it, part_size, tok,
                            [=, &tok, &curr](reference t, std::size_t i)
                            {
                                ++curr;
                                if (op(t, *first2))
                                {
                                    difference_type local_count = 1;
                                    FwdIter2 needle = first2;
                                    FwdIter mid = curr;

                                    for (difference_type len = 0;
                                         local_count != diff && len != count;
                                         ++local_count, ++len, ++mid)
                                    {
                                        if (*mid != *++needle)
                                            break;
                                    }

                                    if (local_count == diff)
                                        tok.cancel(i);
                                }
                            });
                },
                [=](std::vector<hpx::future<void> > &&) mutable
                {
                    difference_type find_end_res = tok.get_data();
                    if (find_end_res != count)
                        std::advance(first1, find_end_res);
                    else
                        first1 = last1;

                    return std::move(first1);
                });
            }
        };
        /// \endcond
    }

    /// Returns the last subsequence of elements [first2,last2) found in the range
    /// [first,last) using the operator== to compare elements.
    ///
    /// \note   Complexity: at most S*(N-S+1) comparisons where
    ///         \a S = distance(first2, last2) and
    ///         \a N = distance(first1, last1).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last1        Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the sequence of elements
    ///                     the algorithm will be searching for.
    /// \param last2        Refers to the end of the sequence of elements of
    ///                     the algorithm will be searching for.
    ///
    /// The comparison operations in the parallel \a find_end algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find_end algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find_end algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type \a task_execution_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a find_end algorithm returns an iterator to the beginning of
    ///           the last subsequence [first2, last2) in range [first, last).
    ///           If the length of the subsequence [first2, last2) is greater
    ///           than the length of the range [first1, last1), last1 is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, last1 is also returned.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, FwdIter1>::type
    >::type
    find_end(ExPolicy && policy, FwdIter1 first1, FwdIter1 last1,
        FwdIter2 first2, FwdIter2 last2)
    {
        typedef typename std::iterator_traits<FwdIter1>::iterator_category
            iterator_category1;

        typedef typename std::iterator_traits<FwdIter2>::iterator_category
            iterator_category2;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::forward_iterator_tag, iterator_category1
            >::value),
            "Requires at least forward iterator.");

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::forward_iterator_tag, iterator_category2
            >::value),
            "Requires at least forward iterator.");

        typedef is_sequential_execution_policy<ExPolicy> is_seq;

        return detail::find_end<FwdIter1>().call(
            std::forward<ExPolicy>(policy),
            first1, last1, first2, last2, detail::equal_to(),
            is_seq());
    }

    /// Returns the last subsequence of elements [first2, last2) found in the range
    /// [first,last) using the given predicate \a f to compare elements.
    ///
    /// \note   Complexity: at most S*(N-S+1) comparisons where
    ///         \a S = distance(first2, last2) and
    ///         \a N = distance(first1, last1).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last1        Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the sequence of elements
    ///                     the algorithm will be searching for.
    /// \param last2        Refers to the end of the sequence of elements of
    ///                     the algorithm will be searching for.
    /// \param f            The binary predicate which returns \a true
    ///                     if the elements should be treated as equal. The signature
    ///                     should be equivalent to the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a FwdIter1 and \a FwdIter2
    ///                     can be dereferenced and then implicitly converted
    ///                     to \a Type1 and \a Type2 respectively.
    ///
    /// The comparison operations in the parallel \a find_end algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find_end algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find_end algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type \a task_execution_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a find_end algorithm returns an iterator to the beginning of
    ///           the last subsequence [first2, last2) in range [first, last).
    ///           If the length of the subsequence [first2, last2) is greater
    ///           than the length of the range [first1, last1), last1 is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, last1 is also returned.
    ///
    /// This overload of \a find_end is available if the user decides to provide the
    /// algorithm their own predicate \a f.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, FwdIter1>::type
    >::type
    find_end(ExPolicy && policy, FwdIter1 first1, FwdIter1 last1,
        FwdIter2 first2, FwdIter2 last2, F && f)
    {
        typedef typename std::iterator_traits<FwdIter1>::iterator_category
            iterator_category1;

        typedef typename std::iterator_traits<FwdIter2>::iterator_category
            iterator_category2;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::forward_iterator_tag, iterator_category1
            >::value),
            "Requires at least forward iterator.");

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::forward_iterator_tag, iterator_category2
            >::value),
            "Requires at least forward iterator.");

        typedef is_sequential_execution_policy<ExPolicy> is_seq;

        return detail::find_end<FwdIter1>().call(
            std::forward<ExPolicy>(policy),
            first1, last1, first2, last2, std::forward<F>(f), is_seq());
    }
}}}

#endif
