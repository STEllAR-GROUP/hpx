//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/replace.hpp

#if !defined(HPX_PARALLEL_DETAIL_REPLACE_AUG_18_2014_0136PM)
#define HPX_PARALLEL_DETAIL_REPLACE_AUG_18_2014_0136PM

#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/tagged_pair.hpp>
#include <hpx/util/unused.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/traits/projected.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1
{
    ///////////////////////////////////////////////////////////////////////////
    // replace
    namespace detail
    {
        /// \cond NOINTERNAL

        // sequential replace
        template <typename FwdIter, typename T1, typename T2, typename Proj>
        inline FwdIter
        sequential_replace(FwdIter first, FwdIter last, T1 const& old_value,
            T2 const& new_value, Proj && proj)
        {
            for (/* */; first != last; ++first)
            {
                if (hpx::util::invoke(proj, *first) == old_value)
                {
                    *first = new_value;
                }
            }
            return first;
        }

        template <typename Iter>
        struct replace : public detail::algorithm<replace<Iter>, Iter>
        {
            replace()
              : replace::algorithm("replace")
            {}

            template <typename ExPolicy, typename FwdIter, typename T1,
                typename T2, typename Proj>
            static FwdIter
            sequential(ExPolicy, FwdIter first, FwdIter last,
                T1 const& old_value, T2 const& new_value, Proj && proj)
            {
                return sequential_replace(first, last, old_value, new_value,
                    std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename FwdIter, typename T1,
                typename T2, typename Proj>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last,
                T1 const& old_value, T2 const& new_value, Proj && proj)
            {
                typedef typename std::iterator_traits<FwdIter>::value_type type;

                return for_each_n<FwdIter>().call(
                    std::forward<ExPolicy>(policy), std::false_type(),
                    first, std::distance(first, last),
                    [old_value, new_value, proj](type& t)
                    {
                        if (hpx::util::invoke(proj, t) == old_value)
                        {
                            t = new_value;
                        }
                    },
                    util::projection_identity());
            }
        };
        /// \endcond
    }

    /// Replaces all elements satisfying specific criteria with \a new_value
    /// in the range [first, last).
    ///
    /// Effects: Substitutes elements referred by the iterator it in the range
    ///          [first, last) with new_value, when the following corresponding
    ///          conditions hold: INVOKE(proj, *it) == old_value
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam T1          The type of the old value to replace (deduced).
    /// \tparam T2          The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param old_value    Refers to the old value of the elements to replace.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace algorithm invoked with an
    /// execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a replace algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a replace algorithm returns a \a hpx::future<FwdIter> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a void otherwise.
    ///           It returns \a last.
    ///
    template <typename ExPolicy, typename FwdIter, typename T1, typename T2,
        typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<FwdIter>::value &&
        traits::is_projected<Proj, FwdIter>::value &&
        traits::is_indirect_callable<
            ExPolicy, std::equal_to<T1>,
                traits::projected<Proj, FwdIter>,
                traits::projected<Proj, T1 const*>
        >::value)>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    replace(ExPolicy && policy, FwdIter first, FwdIter last,
        T1 const& old_value, T2 const& new_value, Proj && proj = Proj())
    {
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter>::value),
            "Required at least forward iterator.");

        typedef execution::is_sequential_execution_policy<ExPolicy> is_seq;

        return detail::replace<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, old_value, new_value, std::forward<Proj>(proj));
    }

    ///////////////////////////////////////////////////////////////////////////
    // replace_if
    namespace detail
    {
        /// \cond NOINTERNAL

        // sequential replace_if
        template <typename FwdIter, typename F, typename T, typename Proj>
        inline FwdIter
        sequential_replace_if(FwdIter first, FwdIter last, F && f,
            T const& new_value, Proj && proj)
        {
            for (/* */; first != last; ++first)
            {
                using hpx::util::invoke;
                if (invoke(f, invoke(proj, *first)))
                {
                    *first = new_value;
                }
            }
            return first;
        }

        template <typename Iter>
        struct replace_if : public detail::algorithm<replace_if<Iter>, Iter>
        {
            replace_if()
              : replace_if::algorithm("replace_if")
            {}

            template <typename ExPolicy, typename FwdIter, typename F,
                typename T, typename Proj>
            static FwdIter
            sequential(ExPolicy, FwdIter first, FwdIter last, F && f,
                T const& new_value, Proj && proj)
            {
                return sequential_replace_if(first, last, std::forward<F>(f),
                    new_value, std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename FwdIter, typename F,
                typename T, typename Proj>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last,
                F && f, T const& new_value, Proj && proj)
            {
                typedef typename std::iterator_traits<FwdIter>::value_type type;

                return for_each_n<FwdIter>().call(
                    std::forward<ExPolicy>(policy), std::false_type(),
                    first, std::distance(first, last),
                    [f, new_value, proj](type& t)
                    {
                        using hpx::util::invoke;
                        if (invoke(f, invoke(proj, t)))
                            t = new_value;
                    },
                    util::projection_identity());
            }
        };
        /// \endcond
    }

    /// Replaces all elements satisfying specific criteria (for which predicate
    /// \a f returns true) with \a new_value in the range [first, last).
    ///
    /// Effects: Substitutes elements referred by the iterator it in the range
    ///          [first, last) with new_value, when the following corresponding
    ///          conditions hold: INVOKE(f, INVOKE(proj, *it)) != false
    ///
    /// \note   Complexity: Performs exactly \a last - \a first applications of
    ///         the predicate.
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
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///                     (deduced).
    /// \tparam T           The type of the new values to replace (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     elements which need to replaced. The
    ///                     signature of this predicate should be equivalent
    ///                     to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace_if algorithm invoked with an
    /// execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a replace_if algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a replace_if algorithm returns a \a hpx::future<FwdIter>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy
    ///           and returns \a void otherwise.
    ///           It returns \a last.
    ///
    template <typename ExPolicy, typename FwdIter, typename F, typename T,
        typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<FwdIter>::value &&
        traits::is_projected<Proj, FwdIter>::value &&
        traits::is_indirect_callable<
            ExPolicy, F, traits::projected<Proj, FwdIter>
        >::value)>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    replace_if(ExPolicy && policy, FwdIter first, FwdIter last,
        F && f, T const& new_value, Proj && proj = Proj())
    {
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter>::value),
            "Required at least forward iterator.");

        typedef execution::is_sequential_execution_policy<ExPolicy> is_seq;

        return detail::replace_if<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, std::forward<F>(f), new_value,
            std::forward<Proj>(proj));
    }

    ///////////////////////////////////////////////////////////////////////////
    // replace_copy
    namespace detail
    {
        /// \cond NOINTERNAL

        // sequential replace_copy
        template <typename InIter, typename OutIter, typename T, typename Proj>
        inline std::pair<InIter, OutIter>
        sequential_replace_copy(InIter first, InIter last, OutIter dest,
            T const& old_value, T const& new_value, Proj && proj)
        {
            for (/* */; first != last; ++first)
            {
                if (hpx::util::invoke(proj, *first) == old_value)
                    *dest++ = new_value;
                else
                    *dest++ = *first;
            }
            return std::make_pair(first, dest);
        }

        template <typename IterPair>
        struct replace_copy
          : public detail::algorithm<replace_copy<IterPair>, IterPair>
        {
            replace_copy()
              : replace_copy::algorithm("replace_copy")
            {}

            template <typename ExPolicy, typename InIter, typename OutIter,
                typename T, typename Proj>
            static std::pair<InIter, OutIter>
            sequential(ExPolicy, InIter first, InIter last, OutIter dest,
                T const& old_value, T const& new_value, Proj && proj)
            {
                return sequential_replace_copy(first, last, dest,
                    old_value, new_value, std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename FwdIter, typename OutIter,
                typename T, typename Proj>
            static typename util::detail::algorithm_result<
                ExPolicy, std::pair<FwdIter, OutIter>
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last,
                OutIter dest, T const& old_value, T const& new_value,
                Proj && proj)
            {
                typedef hpx::util::zip_iterator<FwdIter, OutIter> zip_iterator;
                typedef typename zip_iterator::reference reference;

                return get_iter_pair(
                    for_each_n<zip_iterator>().call(
                        std::forward<ExPolicy>(policy), std::false_type(),
                        hpx::util::make_zip_iterator(first, dest),
                        std::distance(first, last),
                        [old_value, new_value, proj](reference t)
                        {
                            using hpx::util::get;
                            if (hpx::util::invoke(proj, get<0>(t)) == old_value)
                                get<1>(t) = new_value;
                            else
                                get<1>(t) = get<0>(t); //-V573
                        },
                        util::projection_identity()));
            }
        };
        /// \endcond
    }

    /// Copies the all elements from the range [first, last) to another range
    /// beginning at \a dest replacing all elements satisfying a specific
    /// criteria with \a new_value.
    ///
    /// Effects: Assigns to every iterator it in the range
    ///          [result, result + (last - first)) either new_value or
    ///          *(first + (it - result)) depending on whether the following
    ///          corresponding condition holds:
    ///          INVOKE(proj, *(first + (i - result))) == old_value
    ///
    /// \note   Complexity: Performs exactly \a last - \a first applications of
    ///         the predicate.
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
    /// \tparam T1          The type of the old value to replace (deduced).
    /// \tparam T2          The type of the new values to replace (deduced).
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
    /// \param old_value    Refers to the old value of the elements to replace.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace_copy algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a replace_copy algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a replace_copy algorithm returns a
    ///           \a hpx::future<tagged_pair<tag::in(InIter), tag::out(OutIter)> >
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a tagged_pair<tag::in(InIter), tag::out(OutIter)>
    ///           otherwise.
    ///           The \a copy algorithm returns the pair of the input iterator
    ///           \a last and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename InIter, typename OutIter,
        typename T1, typename T2, typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<InIter>::value &&
        traits::is_projected<Proj, InIter>::value &&
        traits::is_indirect_callable<
            ExPolicy, std::equal_to<T1>,
                traits::projected<Proj, InIter>,
                traits::projected<Proj, T1 const*>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy, hpx::util::tagged_pair<tag::in(InIter), tag::out(OutIter)>
    >::type
    replace_copy(ExPolicy && policy, InIter first, InIter last, OutIter dest,
        T1 const& old_value, T2 const& new_value, Proj && proj = Proj())
    {
        static_assert(
            (hpx::traits::is_input_iterator<InIter>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<OutIter>::value||
                hpx::traits::is_forward_iterator<OutIter>::value),
            "Requires at least output iterator.");

        typedef std::integral_constant<bool,
                execution::is_sequential_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<InIter>::value ||
               !hpx::traits::is_forward_iterator<OutIter>::value
            > is_seq;

        return hpx::util::make_tagged_pair<tag::in, tag::out>(
            detail::replace_copy<std::pair<InIter, OutIter> >().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, dest, old_value, new_value,
                std::forward<Proj>(proj)));
    }

    ///////////////////////////////////////////////////////////////////////////
    // replace_copy_if
    namespace detail
    {
        /// \cond NOINTERNAL

        // sequential replace_copy_if
        template <typename InIter, typename OutIter, typename F, typename T,
            typename Proj>
        inline std::pair<InIter, OutIter>
        sequential_replace_copy_if(InIter first, InIter last, OutIter dest,
            F && f, T const& new_value, Proj && proj)
        {
            for (/* */; first != last; ++first)
            {
                using hpx::util::invoke;
                if (invoke(f, invoke(proj, *first)))
                    *dest++ = new_value;
                else
                    *dest++ = *first;
            }
            return std::make_pair(first, dest);
        }

        template <typename IterPair>
        struct replace_copy_if
          : public detail::algorithm<replace_copy_if<IterPair>, IterPair>
        {
            replace_copy_if()
              : replace_copy_if::algorithm("replace_copy_if")
            {}

            template <typename ExPolicy, typename InIter, typename OutIter,
                typename F, typename T, typename Proj>
            static std::pair<InIter, OutIter>
            sequential(ExPolicy, InIter first, InIter last,
                OutIter dest, F && f, T const& new_value, Proj && proj)
            {
                return sequential_replace_copy_if(first, last, dest,
                    std::forward<F>(f), new_value, std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename FwdIter, typename OutIter,
                typename F, typename T, typename Proj>
            static typename util::detail::algorithm_result<
                ExPolicy, std::pair<FwdIter, OutIter>
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last,
                OutIter dest, F && f, T const& new_value, Proj && proj)
            {
                typedef hpx::util::zip_iterator<FwdIter, OutIter> zip_iterator;
                typedef typename zip_iterator::reference reference;

                return get_iter_pair(
                    for_each_n<zip_iterator>().call(
                        std::forward<ExPolicy>(policy), std::false_type(),
                        hpx::util::make_zip_iterator(first, dest),
                        std::distance(first, last),
                        [f, new_value, proj](reference t)
                        {
                            using hpx::util::get;
                            using hpx::util::invoke;
                            if (invoke(f, invoke(proj, get<0>(t))))
                                get<1>(t) = new_value;
                            else
                                get<1>(t) = get<0>(t); //-V573
                        },
                        util::projection_identity()));
            }
        };
        /// \endcond
    }

    /// Copies the all elements from the range [first, last) to another range
    /// beginning at \a dest replacing all elements satisfying a specific
    /// criteria with \a new_value.
    ///
    /// Effects: Assigns to every iterator it in the range
    ///          [result, result + (last - first)) either new_value or
    ///          *(first + (it - result)) depending on whether the following
    ///          corresponding condition holds:
    ///          INVOKE(f, INVOKE(proj, *(first + (i - result)))) != false
    ///
    /// \note   Complexity: Performs exactly \a last - \a first applications of
    ///         the predicate.
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
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///                     (deduced).
    /// \tparam T           The type of the new values to replace (deduced).
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
    ///                     elements which need to replaced. The
    ///                     signature of this predicate should be equivalent
    ///                     to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param new_value    Refers to the new value to use as the replacement.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a replace_copy_if algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a replace_copy_if algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a replace_copy_if algorithm returns a
    ///           \a hpx::future<tagged_pair<tag::in(InIter), tag::out(OutIter)> >
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy
    ///           and returns \a tagged_pair<tag::in(InIter), tag::out(OutIter)>
    ///           otherwise.
    ///           The \a replace_copy_if algorithm returns the input iterator
    ///           \a last and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename InIter, typename OutIter, typename F,
        typename T, typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<InIter>::value &&
        traits::is_projected<Proj, InIter>::value &&
        traits::is_indirect_callable<
            ExPolicy, F, traits::projected<Proj, InIter>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy, hpx::util::tagged_pair<tag::in(InIter), tag::out(OutIter)>
    >::type
    replace_copy_if(ExPolicy && policy, InIter first, InIter last, OutIter dest,
        F && f, T const& new_value, Proj && proj = Proj())
    {
        static_assert(
            (hpx::traits::is_input_iterator<InIter>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<OutIter>::value||
                hpx::traits::is_forward_iterator<OutIter>::value),
            "Requires at least output iterator.");

        typedef std::integral_constant<bool,
                execution::is_sequential_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<InIter>::value ||
               !hpx::traits::is_forward_iterator<OutIter>::value
            > is_seq;

        return hpx::util::make_tagged_pair<tag::in, tag::out>(
            detail::replace_copy_if<std::pair<InIter, OutIter> >().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, dest, std::forward<F>(f), new_value,
                std::forward<Proj>(proj)));
    }
}}}

#endif
