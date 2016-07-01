//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/transform.hpp

#if !defined(HPX_PARALLEL_CONTAINER_ALGORITHM_TRANSFORM_JUL_18_2015_0804PM)
#define HPX_PARALLEL_CONTAINER_ALGORITHM_TRANSFORM_JUL_18_2015_0804PM

#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/tagged_pair.hpp>
#include <hpx/util/tagged_tuple.hpp>

#include <hpx/parallel/algorithms/transform.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/traits/is_range.hpp>
#include <hpx/parallel/traits/projected_range.hpp>
#include <hpx/parallel/traits/range_traits.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <boost/range/functions.hpp>

#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    /// Applies the given function \a f to the given range \a rng and stores
    /// the result in another range, beginning at dest.
    ///
    /// \note   Complexity: Exactly size(rng) applications of \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the invocations of \a f.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a transform requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&.
    ///                     The type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to \a Type. The type \a Ret
    ///                     must be such that an object of type \a OutIter can
    ///                     be dereferenced and assigned a value of type
    ///                     \a Ret.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a f is invoked.
    ///
    /// The invocations of \a f in the parallel \a transform algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The invocations of \a f in the parallel \a transform algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a transform algorithm returns a
    ///           \a hpx::future<tagged_pair<tag::in(InIter), tag::out(OutIter)> >
    ///           if the execution policy is of type \a parallel_task_execution_policy
    ///           and returns \a tagged_pair<tag::in(InIter), tag::out(OutIter)>
    ///           otherwise.
    ///           The \a transform algorithm returns a tuple holding an iterator
    ///           referring to the first element after the input sequence and
    ///           the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename Rng, typename OutIter, typename F,
        typename Proj,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        traits::is_range<Rng>::value &&
        hpx::traits::is_iterator<OutIter>::value &&
        traits::is_projected_range<Proj, Rng>::value &&
        traits::is_indirect_callable<
            F, traits::projected_range<Proj, Rng>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy,
        hpx::util::tagged_pair<
            tag::in(typename traits::range_iterator<Rng>::type),
            tag::out(OutIter)
        >
    >::type
    transform(ExPolicy && policy, Rng && rng, OutIter dest, F && f, Proj && proj)
    {
        return transform(std::forward<ExPolicy>(policy),
            boost::begin(rng), boost::end(rng), std::move(dest),
            std::forward<F>(f), std::forward<Proj>(proj));
    }

    /// \cond NOINTERNAL
    template <typename ExPolicy, typename Rng, typename OutIter, typename F,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        traits::is_range<Rng>::value &&
        hpx::traits::is_iterator<OutIter>::value &&
        traits::is_projected_range<util::projection_identity, Rng>::value &&
        traits::is_indirect_callable<
            F, traits::projected_range<util::projection_identity, Rng>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy,
        hpx::util::tagged_pair<
            tag::in(typename traits::range_iterator<Rng>::type),
            tag::out(OutIter)
        >
    >::type
    transform(ExPolicy && policy, Rng && rng, OutIter dest, F && f)
    {
        return transform(std::forward<ExPolicy>(policy),
            boost::begin(rng), boost::end(rng), std::move(dest),
            std::forward<F>(f), util::projection_identity());
    }
    /// \endcond

    /// Applies the given function \a f to pairs of elements from two ranges:
    /// one defined by \a rng and the other beginning at first2, and
    /// stores the result in another range, beginning at dest.
    ///
    /// \note   Complexity: Exactly size(rng) applications of \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the invocations of \a f.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam InIter2     The type of the source iterators for the second
    ///                     range used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a transform requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj1       The type of an optional projection function to be
    ///                     used for elements of the first sequence. This
    ///                     defaults to \a util::projection_identity
    /// \tparam Proj2       The type of an optional projection function to be
    ///                     used for elements of the second sequence. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param first2       Refers to the beginning of the second sequence of
    ///                     elements the algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&.
    ///                     The types \a Type1 and \a Type2 must be such that
    ///                     objects of types InIter1 and InIter2 can be
    ///                     dereferenced and then implicitly converted to
    ///                     \a Type1 and \a Type2 respectively. The type \a Ret
    ///                     must be such that an object of type \a OutIter can
    ///                     be dereferenced and assigned a value of type
    ///                     \a Ret.
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     first sequence as a projection operation before the
    ///                     actual predicate \a f is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     second sequence as a projection operation before
    ///                     the actual predicate \a f is invoked.
    ///
    /// The invocations of \a f in the parallel \a transform algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The invocations of \a f in the parallel \a transform algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a transform algorithm returns a
    /// \a hpx::future<tagged_tuple<tag::in1(InIter1), tag::in2(InIter2), tag::out(OutIter)> >
    ///           if the execution policy is of type \a parallel_task_execution_policy
    ///           and returns
    /// \a tagged_tuple<tag::in1(InIter1), tag::in2(InIter2), tag::out(OutIter)>
    ///           otherwise.
    ///           The \a transform algorithm returns a tuple holding an iterator
    ///           referring to the first element after the first input sequence,
    ///           an iterator referring to the first element after the second
    ///           input sequence, and the output iterator referring to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <
        typename ExPolicy, typename Rng, typename InIter2,
        typename OutIter, typename F, typename Proj1, typename Proj2,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        traits::is_range<Rng>::value &&
        hpx::traits::is_iterator<InIter2>::value &&
        hpx::traits::is_iterator<OutIter>::value &&
        traits::is_projected_range<Proj1, Rng>::value &&
        traits::is_projected<Proj2, InIter2>::value &&
        traits::is_indirect_callable<
            F, traits::projected_range<Proj1, Rng>,
                traits::projected<Proj2, InIter2>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy,
        hpx::util::tagged_tuple<
            tag::in1(typename traits::range_iterator<Rng>::type),
            tag::in2(InIter2), tag::out(OutIter)
        >
    >::type
    transform(ExPolicy && policy, Rng && rng, InIter2 first2, OutIter dest,
        F && f, Proj1 && proj1, Proj2 && proj2)
    {
        return transform(std::forward<ExPolicy>(policy),
            boost::begin(rng), boost::end(rng), std::move(first2),
            std::move(dest), std::forward<F>(f),
            std::forward<Proj1>(proj1), std::forward<Proj2>(proj2));
    }

    /// \cond NOINTERNAL
    template <
        typename ExPolicy, typename Rng, typename InIter2,
        typename OutIter, typename F, typename Proj1,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        traits::is_range<Rng>::value &&
        hpx::traits::is_iterator<InIter2>::value &&
        hpx::traits::is_iterator<OutIter>::value &&
        traits::is_projected_range<Proj1, Rng>::value &&
        traits::is_projected<util::projection_identity, InIter2>::value &&
        traits::is_indirect_callable<
            F, traits::projected_range<Proj1, Rng>,
                traits::projected<util::projection_identity, InIter2>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy,
        hpx::util::tagged_tuple<
            tag::in1(typename traits::range_iterator<Rng>::type),
            tag::in2(InIter2), tag::out(OutIter)
        >
    >::type
    transform(ExPolicy && policy, Rng && rng, InIter2 first2, OutIter dest,
        F && f, Proj1 && proj1)
    {
        return transform(std::forward<ExPolicy>(policy),
            boost::begin(rng), boost::end(rng), std::move(first2),
            std::move(dest), std::forward<F>(f),
            std::forward<Proj1>(proj1), util::projection_identity());
    }

    template <
        typename ExPolicy, typename Rng, typename InIter2,
        typename OutIter, typename F,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        traits::is_range<Rng>::value &&
        hpx::traits::is_iterator<InIter2>::value &&
        hpx::traits::is_iterator<OutIter>::value &&
        traits::is_projected_range<util::projection_identity, Rng>::value &&
        traits::is_projected<util::projection_identity, InIter2>::value &&
        traits::is_indirect_callable<
            F, traits::projected_range<util::projection_identity, Rng>,
                traits::projected<util::projection_identity, InIter2>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy,
        hpx::util::tagged_tuple<
            tag::in1(typename traits::range_iterator<Rng>::type),
            tag::in2(InIter2), tag::out(OutIter)
        >
    >::type
    transform(ExPolicy && policy, Rng && rng, InIter2 first2, OutIter dest,
        F && f)
    {
        return transform(std::forward<ExPolicy>(policy),
            boost::begin(rng), boost::end(rng), std::move(first2),
            std::move(dest), std::forward<F>(f),
            util::projection_identity(), util::projection_identity());
    }
    /// \endcond

    /// Applies the given function \a f to pairs of elements from two ranges:
    /// one defined by [first1, last1) and the other beginning at first2, and
    /// stores the result in another range, beginning at dest.
    ///
    /// \note   Complexity: Exactly min(last2-first2, last1-first1)
    ///         applications of \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the invocations of \a f.
    /// \tparam Rng1        The type of the first source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Rng2        The type of the second source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a transform requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj1       The type of an optional projection function to be
    ///                     used for elements of the first sequence. This
    ///                     defaults to \a util::projection_identity
    /// \tparam Proj2       The type of an optional projection function to be
    ///                     used for elements of the second sequence. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng1         Refers to the first sequence of elements the
    ///                     algorithm will be applied to.
    /// \param rng2         Refers to the second sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&.
    ///                     The types \a Type1 and \a Type2 must be such that
    ///                     objects of types InIter1 and InIter2 can be
    ///                     dereferenced and then implicitly converted to
    ///                     \a Type1 and \a Type2 respectively. The type \a Ret
    ///                     must be such that an object of type \a OutIter can
    ///                     be dereferenced and assigned a value of type
    ///                     \a Ret.
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     first sequence as a projection operation before the
    ///                     actual predicate \a f is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     second sequence as a projection operation before
    ///                     the actual predicate \a f is invoked.
    ///
    /// The invocations of \a f in the parallel \a transform algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The invocations of \a f in the parallel \a transform algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note The algorithm will invoke the binary predicate until it reaches
    ///       the end of the shorter of the two given input sequences
    ///
    /// \returns  The \a transform algorithm returns a
    /// \a hpx::future<tagged_tuple<tag::in1(InIter1), tag::in2(InIter2), tag::out(OutIter)> >
    ///           if the execution policy is of type \a parallel_task_execution_policy
    ///           and returns
    /// \a tagged_tuple<tag::in1(InIter1), tag::in2(InIter2), tag::out(OutIter)>
    ///           otherwise.
    ///           The \a transform algorithm returns a tuple holding an iterator
    ///           referring to the first element after the first input sequence,
    ///           an iterator referring to the first element after the second
    ///           input sequence, and the output iterator referring to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <
        typename ExPolicy, typename Rng1, typename Rng2,
        typename OutIter, typename F, typename Proj1, typename Proj2,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        traits::is_range<Rng1>::value &&
        traits::is_range<Rng2>::value &&
        hpx::traits::is_iterator<OutIter>::value &&
        traits::is_projected_range<Proj1, Rng1>::value &&
        traits::is_projected_range<Proj2, Rng2>::value &&
        traits::is_indirect_callable<
            F, traits::projected_range<Proj1, Rng1>,
                traits::projected_range<Proj2, Rng2>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy,
        hpx::util::tagged_tuple<
            tag::in1(typename traits::range_iterator<Rng1>::type),
            tag::in2(typename traits::range_iterator<Rng2>::type),
            tag::out(OutIter)
        >
    >::type
    transform(ExPolicy && policy, Rng1 && rng1, Rng2 && rng2, OutIter dest,
        F && f, Proj1 && proj1, Proj2 && proj2)
    {
        return transform(std::forward<ExPolicy>(policy),
            boost::begin(rng1), boost::end(rng1),
            boost::begin(rng2), boost::end(rng2),
            std::move(dest), std::forward<F>(f),
            std::forward<Proj1>(proj1), std::forward<Proj2>(proj2));
    }

    /// \cond NOINTERNAL
    template <
        typename ExPolicy, typename Rng1, typename Rng2,
        typename OutIter, typename F, typename Proj1,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        traits::is_range<Rng1>::value &&
        traits::is_range<Rng2>::value &&
        hpx::traits::is_iterator<OutIter>::value &&
        traits::is_projected_range<Proj1, Rng1>::value &&
        traits::is_projected_range<util::projection_identity, Rng2>::value &&
        traits::is_indirect_callable<
            F, traits::projected_range<Proj1, Rng1>,
                traits::projected_range<util::projection_identity, Rng2>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy,
        hpx::util::tagged_tuple<
            tag::in1(typename traits::range_iterator<Rng1>::type),
            tag::in2(typename traits::range_iterator<Rng2>::type),
            tag::out(OutIter)
        >
    >::type
    transform(ExPolicy && policy, Rng1 && rng1, Rng2 && rng2, OutIter dest,
        F && f, Proj1 && proj1)
    {
        return transform(std::forward<ExPolicy>(policy),
            boost::begin(rng1), boost::end(rng1),
            boost::begin(rng2), boost::end(rng2),
            std::move(dest), std::forward<F>(f),
            std::forward<Proj1>(proj1), util::projection_identity());
    }

    template <
        typename ExPolicy, typename Rng1, typename Rng2,
        typename OutIter, typename F,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        traits::is_range<Rng1>::value &&
        traits::is_range<Rng2>::value &&
        hpx::traits::is_iterator<OutIter>::value &&
        traits::is_projected_range<util::projection_identity, Rng1>::value &&
        traits::is_projected_range<util::projection_identity, Rng2>::value &&
        traits::is_indirect_callable<
            F, traits::projected_range<util::projection_identity, Rng1>,
                traits::projected_range<util::projection_identity, Rng2>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy,
        hpx::util::tagged_tuple<
            tag::in1(typename traits::range_iterator<Rng1>::type),
            tag::in2(typename traits::range_iterator<Rng2>::type),
            tag::out(OutIter)
        >
    >::type
    transform(ExPolicy && policy, Rng1 && rng1, Rng2 && rng2, OutIter dest,
        F && f)
    {
        return transform(std::forward<ExPolicy>(policy),
            boost::begin(rng1), boost::end(rng1),
            boost::begin(rng2), boost::end(rng2),
            std::move(dest), std::forward<F>(f),
            util::projection_identity(), util::projection_identity());
    }
    /// \endcond
}}}

#endif


