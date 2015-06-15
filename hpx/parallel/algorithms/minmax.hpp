//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/minmax.hpp

#if !defined(HPX_PARALLEL_DETAIL_MINMAX_AUG_20_2014_1005AM)
#define HPX_PARALLEL_DETAIL_MINMAX_AUG_20_2014_1005AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/assert.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // min_element
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename FwdIter, typename F>
        FwdIter sequential_min_element(FwdIter it, std::size_t count,
            F const& f)
        {
            if (count == 0 || count == 1)
                return it;

            FwdIter smallest = it;
            util::loop_n(++it, count-1, [&f, &smallest](FwdIter const& curr)
            {
                if (f(*curr, *smallest))
                    smallest = curr;
            });
            return smallest;
        }

        template <typename FwdIter, typename F>
        typename std::iterator_traits<FwdIter>::value_type
        sequential_min_element_ind(FwdIter it, std::size_t count, F const& f)
        {
            HPX_ASSERT(count != 0);

            if (count == 1)
                return *it;

            typename std::iterator_traits<FwdIter>::value_type smallest = *it;
            util::loop_n(++it, count-1, [&f, &smallest](FwdIter const& curr)
            {
                if (f(**curr, *smallest))
                    smallest = *curr;
            });
            return smallest;
        }

        template <typename FwdIter>
        struct min_element
          : public detail::algorithm<min_element<FwdIter>, FwdIter>
        {
            min_element()
              : min_element::algorithm("min_element")
            {}

            template <typename ExPolicy, typename F>
            static FwdIter
            sequential(ExPolicy, FwdIter first, FwdIter last, F && f)
            {
                return std::min_element(first, last, std::forward<F>(f));
            }

            template <typename ExPolicy, typename F>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter
            >::type
            parallel(ExPolicy policy, FwdIter first, FwdIter last,
                F && f)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<ExPolicy, FwdIter>::
                        get(std::move(first));
                }

                return util::partitioner<ExPolicy, FwdIter, FwdIter>::
                    call(
                        policy, first, std::distance(first, last),
                        [f](FwdIter it, std::size_t part_count)
                        {
                            return sequential_min_element(it, part_count, f);
                        },
                        hpx::util::unwrapped([f](std::vector<FwdIter> && positions)
                        {
                            return sequential_min_element_ind(
                                positions.begin(), positions.size(), f);
                        }));
            }
        };
        /// \endcond
    }

    /// Finds the smallest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: Exactly \a max(N-1, 0) comparisons, where
    ///                     N = std::distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a min_element requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     the left argument is less than the right element.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
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
    /// The comparisons in the parallel \a min_element algorithm invoked with
    /// an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a min_element algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a min_element algorithm returns a \a hpx::future<FwdIter>
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy
    ///           and returns \a FwdIter otherwise.
    ///           The \a min_element algorithm returns the iterator to the
    ///           smallest element in the range [first, last). If several
    ///           elements in the range are equivalent to the smallest element,
    ///           returns the iterator to the first such element. Returns last
    ///           if the range is empty.
    ///
    template <typename ExPolicy, typename FwdIter, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    >::type
    min_element(ExPolicy && policy, FwdIter first, FwdIter last, F && f)
    {
        typedef typename std::iterator_traits<FwdIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::forward_iterator_tag, iterator_category>::value),
            "Required at least forward iterator.");

        typedef typename is_sequential_execution_policy<ExPolicy>::type is_seq;

        return detail::min_element<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, std::forward<F>(f));
    }

    /// Finds the smallest element in the range [first, last) using the given
    /// \a std::less predicate.
    ///
    /// \note   Complexity: Exactly \a max(N-1, 0) comparisons, where
    ///                     N = std::distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    ///
    /// The comparisons in the parallel \a min_element algorithm invoked with
    /// an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a min_element algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a min_element algorithm returns a \a hpx::future<FwdIter>
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy
    ///           and returns \a FwdIter otherwise.
    ///           The \a min_element algorithm returns the iterator to the
    ///           smallest element in the range [first, last). If several
    ///           elements in the range are equivalent to the smallest element,
    ///           returns the iterator to the first such element. Returns last
    ///           if the range is empty.
    ///
    template <typename ExPolicy, typename FwdIter>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    >::type
    min_element(ExPolicy && policy, FwdIter first, FwdIter last)
    {
        typedef typename std::iterator_traits<FwdIter>::iterator_category
            iterator_category;
        typedef typename std::iterator_traits<FwdIter>::value_type
            value_type;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::forward_iterator_tag, iterator_category>::value),
            "Required at least forward iterator.");

        typedef typename is_sequential_execution_policy<ExPolicy>::type is_seq;

        return detail::min_element<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, std::less<value_type>());
    }

    ///////////////////////////////////////////////////////////////////////////
    // max_element
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename FwdIter, typename F>
        FwdIter sequential_max_element(FwdIter it, std::size_t count,
            F const& f)
        {
            if (count == 0 || count == 1)
                return it;

            FwdIter greatest = it;
            util::loop_n(++it, count-1, [&f, &greatest](FwdIter const& curr)
            {
                if (f(*greatest, *curr))
                    greatest = curr;
            });
            return greatest;
        }

        template <typename FwdIter, typename F>
        typename std::iterator_traits<FwdIter>::value_type
        sequential_max_element_ind(FwdIter it, std::size_t count, F const& f)
        {
            HPX_ASSERT(count != 0);

            if (count == 1)
                return *it;

            typename std::iterator_traits<FwdIter>::value_type greatest = *it;
            util::loop_n(++it, count-1, [&f, &greatest](FwdIter const& curr)
            {
                if (f(*greatest, **curr))
                    greatest = *curr;
            });
            return greatest;
        }

        template <typename FwdIter>
        struct max_element
          : public detail::algorithm<max_element<FwdIter>, FwdIter>
        {
            max_element()
              : max_element::algorithm("max_element")
            {}

            template <typename ExPolicy, typename F>
            static FwdIter
            sequential(ExPolicy, FwdIter first, FwdIter last, F && f)
            {
                return std::max_element(first, last, std::forward<F>(f));
            }

            template <typename ExPolicy, typename F>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter
            >::type
            parallel(ExPolicy policy, FwdIter first, FwdIter last,
                F && f)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<ExPolicy, FwdIter>::
                        get(std::move(first));
                }

                return util::partitioner<ExPolicy, FwdIter, FwdIter>::
                    call(
                        policy, first, std::distance(first, last),
                        [f](FwdIter it, std::size_t part_count)
                        {
                            return sequential_max_element(it, part_count, f);
                        },
                        hpx::util::unwrapped([f](std::vector<FwdIter> && positions)
                        {
                            return sequential_max_element_ind(
                                positions.begin(), positions.size(), f);
                        }));
            }
        };
        /// \endcond
    }

    /// Finds the greatest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: Exactly \a max(N-1, 0) comparisons, where
    ///                     N = std::distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a max_element requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     the left argument is less than the right element.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
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
    /// The comparisons in the parallel \a max_element algorithm invoked with
    /// an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a max_element algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a max_element algorithm returns a \a hpx::future<FwdIter>
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy
    ///           and returns \a FwdIter otherwise.
    ///           The \a max_element algorithm returns the iterator to the
    ///           smallest element in the range [first, last). If several
    ///           elements in the range are equivalent to the smallest element,
    ///           returns the iterator to the first such element. Returns last
    ///           if the range is empty.
    ///
    template <typename ExPolicy, typename FwdIter, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    >::type
    max_element(ExPolicy && policy, FwdIter first, FwdIter last, F && f)
    {
        typedef typename std::iterator_traits<FwdIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::forward_iterator_tag, iterator_category>::value),
            "Required at least forward iterator.");

        typedef typename is_sequential_execution_policy<ExPolicy>::type is_seq;

        return detail::max_element<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, std::forward<F>(f));
    }

    /// Finds the greatest element in the range [first, last) using the given
    /// \a std::less predicate.
    ///
    /// \note   Complexity: Exactly \a max(N-1, 0) comparisons, where
    ///                     N = std::distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    ///
    /// The comparisons in the parallel \a max_element algorithm invoked with
    /// an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a max_element algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a max_element algorithm returns a \a hpx::future<FwdIter>
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy
    ///           and returns \a FwdIter otherwise.
    ///           The \a max_element algorithm returns the iterator to the
    ///           smallest element in the range [first, last). If several
    ///           elements in the range are equivalent to the smallest element,
    ///           returns the iterator to the first such element. Returns last
    ///           if the range is empty.
    ///
    template <typename ExPolicy, typename FwdIter>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    >::type
    max_element(ExPolicy && policy, FwdIter first, FwdIter last)
    {
        typedef typename std::iterator_traits<FwdIter>::iterator_category
            iterator_category;
        typedef typename std::iterator_traits<FwdIter>::value_type
            value_type;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::forward_iterator_tag, iterator_category>::value),
            "Required at least forward iterator.");

        typedef typename is_sequential_execution_policy<ExPolicy>::type is_seq;

        return detail::max_element<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, std::less<value_type>());
    }

    ///////////////////////////////////////////////////////////////////////////
    // minmax_element
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename FwdIter, typename F>
        std::pair<FwdIter, FwdIter>
        sequential_minmax_element(FwdIter it, std::size_t count, F const& f)
        {
            std::pair<FwdIter, FwdIter> result(it, it);

            if (count == 0 || count == 1)
                return result;

            util::loop_n(++it, count-1, [&f, &result](FwdIter const& curr)
            {
                if (f(*curr, *result.first))
                    result.first = curr;
                else if (!f(*curr, *result.second))
                    result.second = curr;
            });
            return result;
        }

        template <typename PairIter, typename F>
        typename std::iterator_traits<PairIter>::value_type
        sequential_minmax_element_ind(PairIter it, std::size_t count, F const& f)
        {
            HPX_ASSERT(count != 0);

            if (count == 1)
                return *it;

            typename std::iterator_traits<PairIter>::value_type result = *it;
            util::loop_n(++it, count-1, [&f, &result](PairIter const& curr)
            {
                if (f(*curr->first, *result.first))
                    result.first = curr->first;

                if (!f(*curr->second, *result.second))
                    result.second = curr->second;
            });
            return result;
        }

        template <typename FwdIter>
        struct minmax_element
          : public detail::algorithm<
                minmax_element<FwdIter>, std::pair<FwdIter, FwdIter>
            >
        {
            minmax_element()
              : minmax_element::algorithm("minmax_element")
            {}

            template <typename ExPolicy, typename F>
            static std::pair<FwdIter, FwdIter>
            sequential(ExPolicy, FwdIter first, FwdIter last, F && f)
            {
                return std::minmax_element(first, last, std::forward<F>(f));
            }

            template <typename ExPolicy, typename F>
            static typename util::detail::algorithm_result<
                ExPolicy, std::pair<FwdIter, FwdIter>
            >::type
            parallel(ExPolicy policy, FwdIter first, FwdIter last,
                F && f)
            {
                typedef std::pair<FwdIter, FwdIter> result_type;

                result_type result(first, first);
                if (first == last || ++first == last)
                {
                    return util::detail::algorithm_result<
                            ExPolicy, result_type
                        >::get(std::move(result));
                }

                return util::partitioner<ExPolicy, result_type, result_type>::
                    call(
                        policy, result.first, std::distance(result.first, last),
                        [f](FwdIter it, std::size_t part_count)
                        {
                            return sequential_minmax_element(it, part_count, f);
                        },
                        hpx::util::unwrapped([f](std::vector<result_type> && positions)
                        {
                            return sequential_minmax_element_ind(
                                positions.begin(), positions.size(), f);
                        }));
            }
        };
        /// \endcond
    }

    /// Finds the greatest element in the range [first, last) using the given
    /// comparison function \a f.
    ///
    /// \note   Complexity: At most \a max(floor(3/2*(N-1)), 0) applications of
    ///                     the predicate, where N = std::distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a minmax_element requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            The binary predicate which returns true if the
    ///                     the left argument is less than the right element.
    ///                     The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
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
    /// The comparisons in the parallel \a minmax_element algorithm invoked with
    /// an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a minmax_element algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a minmax_element algorithm returns a \a hpx::future<FwdIter>
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy
    ///           and returns \a FwdIter otherwise.
    ///           The \a minmax_element algorithm returns a pair consisting of
    ///           an iterator to the smallest element as the first element and
    ///           an iterator to the greatest element as the second. Returns
    ///           std::make_pair(first, first) if the range is empty. If
    ///           several elements are equivalent to the smallest element, the
    ///           iterator to the first such element is returned. If several
    ///           elements are equivalent to the largest element, the iterator
    ///           to the last such element is returned.
    ///
    template <typename ExPolicy, typename FwdIter, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy,
            std::pair<FwdIter, FwdIter>
        >::type
    >::type
    minmax_element(ExPolicy && policy, FwdIter first, FwdIter last, F && f)
    {
        typedef typename std::iterator_traits<FwdIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::forward_iterator_tag, iterator_category>::value),
            "Required at least forward iterator.");

        typedef typename is_sequential_execution_policy<ExPolicy>::type is_seq;

        return detail::minmax_element<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, std::forward<F>(f));
    }

    /// Finds the greatest element in the range [first, last) using the given
    /// \a std::less predicate.
    ///
    /// \note   Complexity: At most \a max(floor(3/2*(N-1)), 0) applications of
    ///                     the predicate, where N = std::distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    ///
    /// The comparisons in the parallel \a minmax_element algorithm invoked with
    /// an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a minmax_element algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a minmax_element algorithm returns a \a hpx::future<FwdIter>
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy
    ///           and returns \a FwdIter otherwise.
    ///           The \a minmax_element algorithm returns a pair consisting of
    ///           an iterator to the smallest element as the first element and
    ///           an iterator to the greatest element as the second. Returns
    ///           std::make_pair(first, first) if the range is empty. If
    ///           several elements are equivalent to the smallest element, the
    ///           iterator to the first such element is returned. If several
    ///           elements are equivalent to the largest element, the iterator
    ///           to the last such element is returned.
    ///
    template <typename ExPolicy, typename FwdIter>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<
            ExPolicy, std::pair<FwdIter, FwdIter>
        >::type
    >::type
    minmax_element(ExPolicy && policy, FwdIter first, FwdIter last)
    {
        typedef typename std::iterator_traits<FwdIter>::iterator_category
            iterator_category;
        typedef typename std::iterator_traits<FwdIter>::value_type
            value_type;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::forward_iterator_tag, iterator_category>::value),
            "Required at least forward iterator.");

        typedef typename is_sequential_execution_policy<ExPolicy>::type is_seq;

        return detail::minmax_element<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, std::less<value_type>());
    }
}}}

#endif
