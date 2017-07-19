//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/remove_copy.hpp

#if !defined(HPX_PARALLEL_DETAIL_REMOVE_COPY_FEB_25_2015_0137PM)
#define HPX_PARALLEL_DETAIL_REMOVE_COPY_FEB_25_2015_0137PM

#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/tagged_pair.hpp>

#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/util/unused.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1
{
    /////////////////////////////////////////////////////////////////////////////
    // remove_copy
    namespace detail
    {
        /// \cond NOINTERNAL

        // sequential remove_copy
        template <typename InIter, typename OutIter, typename T, typename Proj>
        inline std::pair<InIter, OutIter>
        sequential_remove_copy(InIter first, InIter last, OutIter dest,
            T const& value, Proj && proj)
        {
            for (/* */; first != last; ++first)
            {
                if (!(hpx::util::invoke(proj, *first) == value))
                {
                    *dest++ = *first;
                }
            }
            return std::make_pair(first, dest);
        }

        template <typename IterPair>
        struct remove_copy
          : public detail::algorithm<remove_copy<IterPair>, IterPair>
        {
            remove_copy()
              : remove_copy::algorithm("remove_copy")
            {}

            template <typename ExPolicy, typename InIter, typename OutIter,
                typename T, typename Proj>
            static std::pair<InIter, OutIter>
            sequential(ExPolicy, InIter first, InIter last, OutIter dest,
                T const& val, Proj && proj)
            {
                return sequential_remove_copy(first, last, dest, val,
                    std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
                typename T, typename Proj>
            static typename util::detail::algorithm_result<
                ExPolicy, std::pair<FwdIter1, FwdIter2>
            >::type
            parallel(ExPolicy && policy, FwdIter1 first, FwdIter1 last,
                FwdIter2 dest, T const& val, Proj && proj)
            {
                return copy_if<IterPair>().call(
                    std::forward<ExPolicy>(policy), std::false_type(),
                    first, last, dest,
                    [val, proj](T const& a) -> bool
                    {
                        HPX_UNUSED(proj);
                        return !(a == val);
                    },
                    std::forward<Proj>(proj));
            }
        };
        /// \endcond
    }

    /// Copies the elements in the range, defined by [first, last), to another
    /// range beginning at \a dest. Copies only the elements for which the
    /// comparison operator returns false when compare to val.
    /// The order of the elements that are not removed is preserved.
    ///
    /// Effects: Copies all the elements referred to by the iterator it in the
    ///          range [first,last) for which the following corresponding
    ///          conditions do not hold: INVOKE(proj, *it) == value
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first applications of the
    ///         predicate \a f.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam T           The type that the result of dereferencing FwdIter1 is
    ///                     compared to.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param val          Value to be removed.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a remove_copy algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a remove_copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a remove_copy algorithm returns a
    ///           \a hpx::future<tagged_pair<tag::in(FwdIter1), tag::out(FwdIter2)> >
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a tagged_pair<tag::in(FwdIter1), tag::out(FwdIter2)>
    ///           otherwise.
    ///           The \a copy algorithm returns the pair of the input iterator
    ///           forwarded to the first element after the last in the input
    ///           sequence and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2, typename T,
        typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<FwdIter1>::value &&
        hpx::traits::is_iterator<FwdIter2>::value &&
        traits::is_projected<Proj, FwdIter1>::value &&
        traits::is_indirect_callable<
            ExPolicy, std::equal_to<T>,
                traits::projected<Proj, FwdIter1>,
                traits::projected<Proj, T const*>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy, hpx::util::tagged_pair<tag::in(FwdIter1), tag::out(FwdIter2)>
    >::type
    remove_copy(ExPolicy && policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest,
        T const& val, Proj && proj = Proj())
    {
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
        static_assert(
            (hpx::traits::is_input_iterator<FwdIter1>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<FwdIter2>::value ||
                hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least output iterator.");

        typedef std::integral_constant<bool,
                execution::is_sequenced_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<FwdIter1>::value ||
               !hpx::traits::is_forward_iterator<FwdIter2>::value
            > is_seq;
#else
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;
#endif

        return hpx::util::make_tagged_pair<tag::in, tag::out>(
            detail::remove_copy<std::pair<FwdIter1, FwdIter2> >().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, dest, val, std::forward<Proj>(proj)));
    }

    /////////////////////////////////////////////////////////////////////////////
    // remove_copy_if
    namespace detail
    {
        /// \cond NOINTERNAL

        // sequential remove_copy_if
        template <typename InIter, typename OutIter, typename F, typename Proj>
        inline std::pair<InIter, OutIter>
        sequential_remove_copy_if(InIter first, InIter last, OutIter dest, F p,
            Proj && proj)
        {
            for (/* */; first != last; ++first)
            {
                using hpx::util::invoke;
                if (!invoke(p, invoke(proj, *first)))
                {
                    *dest++ = *first;
                }
            }
            return std::make_pair(first, dest);
        }

        template <typename IterPair>
        struct remove_copy_if
          : public detail::algorithm<remove_copy_if<IterPair>, IterPair>
        {
            remove_copy_if()
              : remove_copy_if::algorithm("remove_copy_if")
            {}

            template <typename ExPolicy, typename InIter, typename OutIter,
                typename F, typename Proj>
            static std::pair<InIter, OutIter>
            sequential(ExPolicy, InIter first, InIter last, OutIter dest,
                F && f, Proj && proj)
            {
                return sequential_remove_copy_if(first, last, dest,
                    std::forward<F>(f), std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
                typename F, typename Proj>
            static typename util::detail::algorithm_result<
                ExPolicy, std::pair<FwdIter1, FwdIter2>
            >::type
            parallel(ExPolicy && policy, FwdIter1 first, FwdIter1 last,
                FwdIter2 dest, F && f, Proj && proj)
            {
                typedef typename std::iterator_traits<FwdIter1>::value_type
                    value_type;

                return copy_if<IterPair>().call(
                    std::forward<ExPolicy>(policy), std::false_type(),
                    first, last, dest,
                    [f](value_type const& a) -> bool
                    {
                        return !hpx::util::invoke(f, a);
                    },
                    std::forward<Proj>(proj));
            }
        };
        /// \endcond
    }

    /// Copies the elements in the range, defined by [first, last), to another
    /// range beginning at \a dest. Copies only the elements for which the
    /// predicate \a f returns false. The order of the elements that are not
    /// removed is preserved.
    ///
    /// Effects: Copies all the elements referred to by the iterator it in the
    ///          range [first,last) for which the following corresponding
    ///          conditions do not hold:
    ///          INVOKE(pred, INVOKE(proj, *it)) != false.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first applications of the
    ///         predicate \a f.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a copy_if requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     elements to be removed. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a remove_copy_if algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a remove_copy_if algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a remove_copy_if algorithm returns a
    ///           \a hpx::future<tagged_pair<tag::in(FwdIter1), tag::out(FwdIter2)> >
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a tagged_pair<tag::in(FwdIter1), tag::out(FwdIter2)>
    ///           otherwise.
    ///           The \a copy algorithm returns the pair of the input iterator
    ///           forwarded to the first element after the last in the input
    ///           sequence and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2, typename F,
        typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<FwdIter1>::value &&
        traits::is_projected<Proj, FwdIter1>::value &&
        traits::is_indirect_callable<
            ExPolicy, F, traits::projected<Proj, FwdIter1>
        >::value &&
        hpx::traits::is_iterator<FwdIter2>::value)>
    typename util::detail::algorithm_result<
        ExPolicy, hpx::util::tagged_pair<tag::in(FwdIter1), tag::out(FwdIter2)>
    >::type
    remove_copy_if(ExPolicy && policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest,
        F && f, Proj && proj = Proj())
    {
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
        static_assert(
            (hpx::traits::is_input_iterator<FwdIter1>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<FwdIter2>::value ||
                hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least output iterator.");

        typedef std::integral_constant<bool,
                execution::is_sequenced_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<FwdIter1>::value ||
               !hpx::traits::is_forward_iterator<FwdIter2>::value
            > is_seq;
#else
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;
#endif

        return hpx::util::make_tagged_pair<tag::in, tag::out>(
            detail::remove_copy_if<std::pair<FwdIter1, FwdIter2> >().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, dest, std::forward<F>(f),
                std::forward<Proj>(proj)));
    }
}}}

#endif
