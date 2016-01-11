//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/transform.hpp

#if !defined(HPX_PARALLEL_DETAIL_TRANSFORM_MAY_29_2014_0932PM)
#define HPX_PARALLEL_DETAIL_TRANSFORM_MAY_29_2014_0932PM

#include <hpx/config.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/tagged_pair.hpp>
#include <hpx/util/tagged_tuple.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/traits/projected.hpp>

#include <algorithm>
#include <iterator>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // transform
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename InIter1, typename OutIter, typename F,
            typename Proj>
        HPX_FORCEINLINE std::pair<InIter1, OutIter>
        sequential_transform(InIter1 first1, InIter1 last1,
            OutIter dest, F && f, Proj && proj)
        {
            while (first1 != last1)
            {
                using hpx::util::invoke;
                *dest++ = invoke(f, invoke(proj, *first1++));
            }
            return std::make_pair(first1, dest);
        }

        template <typename IterPair>
        struct transform
          : public detail::algorithm<transform<IterPair>, IterPair>
        {
            transform()
              : transform::algorithm("transform")
            {}

            template <typename ExPolicy, typename InIter, typename OutIter,
                typename F, typename Proj>
            static std::pair<InIter, OutIter>
            sequential(ExPolicy, InIter first, InIter last,
                OutIter dest, F && f, Proj && proj)
            {
                return sequential_transform(first, last, dest,
                    std::forward<F>(f), std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename FwdIter, typename OutIter,
                typename F, typename Proj>
            static typename util::detail::algorithm_result<
                ExPolicy, std::pair<FwdIter, OutIter>
            >::type
            parallel(ExPolicy policy, FwdIter first, FwdIter last,
                OutIter dest, F && f, Proj && proj)
            {
                typedef hpx::util::zip_iterator<FwdIter, OutIter>
                    zip_iterator;
                typedef typename zip_iterator::reference reference;

                return get_iter_pair(
                    for_each_n<zip_iterator>().call(
                        policy, boost::mpl::false_(),
                        hpx::util::make_zip_iterator(first, dest),
                        std::distance(first, last),
                        [f, proj](reference t)
                        {
                            using hpx::util::get;
                            using hpx::util::invoke;
                            get<1>(t) = invoke(f, invoke(proj, get<0>(t)));
                        }));
            }
        };
        /// \endcond
    }

    /// Applies the given function \a f to the range [first, last) and stores
    /// the result in another range, beginning at dest.
    ///
    /// \note   Complexity: Exactly \a last - \a first applications of \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the invocations of \a f.
    /// \tparam InIter      The type of the source iterators used (deduced).
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
    /// \a hpx::future<tagged_pair<tag::in(FwdIter), tag::out(OutIter)> >
    ///           if the execution policy is of type \a parallel_task_execution_policy
    ///           and returns
    /// \a tagged_pair<tag::in(FwdIter), tag::out(OutIter)> otherwise.
    ///           The \a transform algorithm returns a tuple holding an iterator
    ///           referring to the first element after the input sequence and
    ///           the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename InIter, typename OutIter, typename F,
        typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        traits::detail::is_iterator<InIter>::value &&
        traits::detail::is_iterator<OutIter>::value &&
        traits::is_projected<Proj, InIter>::value &&
        traits::is_indirect_callable<
            F, traits::projected<Proj, InIter>
        >::value)>
    typename util::detail::algorithm_result<ExPolicy,
        hpx::util::tagged_pair<tag::in(InIter), tag::out(OutIter)>
    >::type
    transform(ExPolicy && policy, InIter first, InIter last, OutIter dest,
        F && f, Proj && proj = Proj())
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        static_assert(
            (boost::is_base_of<std::input_iterator_tag, iterator_category>::value),
            "Required at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return hpx::util::make_tagged_pair<tag::in, tag::out>(
            detail::transform<std::pair<InIter, OutIter> >().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, dest, std::forward<F>(f),
                std::forward<Proj>(proj)));
    }

    ///////////////////////////////////////////////////////////////////////////
    // transform binary predicate
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename InIter1, typename InIter2, typename OutIter,
            typename F, typename Proj1, typename Proj2>
        HPX_FORCEINLINE hpx::util::tuple<InIter1, InIter2, OutIter>
        sequential_transform(InIter1 first1, InIter1 last1,
            InIter2 first2,  OutIter dest, F && f,
            Proj1 && proj1, Proj2 && proj2)
        {
            while (first1 != last1)
            {
                using hpx::util::invoke;
                *dest++ =
                    invoke(f,
                        invoke(proj1, *first1++),
                        invoke(proj2, *first2++));
            }
            return hpx::util::make_tuple(first1, first2, dest);
        }

        template <typename IterTuple>
        struct transform_binary
          : public detail::algorithm<transform_binary<IterTuple>, IterTuple>
        {
            transform_binary()
              : transform_binary::algorithm("transform_binary")
            {}

            template <typename ExPolicy, typename InIter1, typename InIter2,
                typename OutIter, typename F, typename Proj1, typename Proj2>
            static hpx::util::tuple<InIter1, InIter2, OutIter>
            sequential(ExPolicy, InIter1 first1, InIter1 last1, InIter2 first2,
                OutIter dest, F && f, Proj1 && proj1, Proj2 && proj2)
            {
                return sequential_transform(first1, last1, first2, dest,
                    std::forward<F>(f), std::forward<Proj1>(proj1),
                    std::forward<Proj2>(proj2));
            }

            template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
                typename OutIter, typename F, typename Proj1, typename Proj2>
            static typename util::detail::algorithm_result<
                ExPolicy, hpx::util::tuple<FwdIter1, FwdIter2, OutIter>
            >::type
            parallel(ExPolicy policy, FwdIter1 first1, FwdIter1 last1,
                FwdIter2 first2, OutIter dest, F && f,
                Proj1 && proj1, Proj2 && proj2)
            {
                typedef hpx::util::zip_iterator<
                        FwdIter1, FwdIter2, OutIter
                    > zip_iterator;
                typedef typename zip_iterator::reference reference;

                return get_iter_tuple(
                    for_each_n<zip_iterator>().call(
                        policy, boost::mpl::false_(),
                        hpx::util::make_zip_iterator(first1, first2, dest),
                        std::distance(first1, last1),
                        [f, proj1, proj2](reference t)
                        {
                            using hpx::util::get;
                            using hpx::util::invoke;
                            get<2>(t) =
                                invoke(f,
                                    invoke(proj1, get<0>(t)),
                                    invoke(proj2, get<1>(t)));
                        }));
            }
        };
        /// \endcond
    }

    /// Applies the given function \a f to pairs of elements from two ranges:
    /// one defined by [first1, last1) and the other beginning at first2, and
    /// stores the result in another range, beginning at dest.
    ///
    /// \note   Complexity: Exactly \a last - \a first applications of \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the invocations of \a f.
    /// \tparam InIter1     The type of the source iterators for the first
    ///                     range used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
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
    /// \param first1       Refers to the beginning of the first sequence of
    ///                     elements the algorithm will be applied to.
    /// \param last1        Refers to the end of the first sequence of elements
    ///                     the algorithm will be applied to.
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
        typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F,
        typename Proj1 = util::projection_identity,
        typename Proj2 = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        traits::detail::is_iterator<InIter1>::value &&
        traits::detail::is_iterator<InIter2>::value &&
        traits::detail::is_iterator<OutIter>::value &&
        traits::is_projected<Proj1, InIter1>::value &&
        traits::is_projected<Proj2, InIter2>::value &&
        traits::is_indirect_callable<
            F, traits::projected<Proj1, InIter1>,
                traits::projected<Proj2, InIter2>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy,
        hpx::util::tagged_tuple<
            tag::in1(InIter1), tag::in2(InIter2), tag::out(OutIter)
        >
    >::type
    transform(ExPolicy && policy,
        InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest, F && f,
        Proj1 && proj1 = Proj1{}, Proj2 && proj2 = Proj2{})
    {
        typedef typename std::iterator_traits<InIter1>::iterator_category category1;
        typedef typename std::iterator_traits<InIter2>::iterator_category category2;

        static_assert(
            (boost::is_base_of<std::input_iterator_tag, category1>::value),
            "Required at least input iterator.");
        static_assert(
            (boost::is_base_of<std::input_iterator_tag, category2>::value),
            "Required at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, category1>,
            boost::is_same<std::input_iterator_tag, category2>
        >::type is_seq;

        typedef hpx::util::tuple<InIter1, InIter2, OutIter> result_type;

        return hpx::util::make_tagged_tuple<tag::in1, tag::in2, tag::out>(
            detail::transform_binary<result_type>().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first1, last1, first2, dest, std::forward<F>(f),
                std::forward<Proj1>(proj1), std::forward<Proj2>(proj2)));
    }

    ///////////////////////////////////////////////////////////////////////////
    // transform binary predicate
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename InIter1, typename InIter2, typename OutIter,
            typename F, typename Proj1, typename Proj2>
        HPX_FORCEINLINE hpx::util::tuple<InIter1, InIter2, OutIter>
        sequential_transform(InIter1 first1, InIter1 last1,
            InIter2 first2, InIter2 last2, OutIter dest, F && f,
            Proj1 && proj1, Proj2 && proj2)
        {
            while (first1 != last1 && first2 != last2)
            {
                using hpx::util::invoke;
                *dest++ =
                    invoke(f,
                        invoke(proj1, *first1++),
                        invoke(proj2, *first2++));
            }
            return hpx::util::make_tuple(first1, first2, dest);
        }

        template <typename IterTuple>
        struct transform_binary2
          : public detail::algorithm<transform_binary2<IterTuple>, IterTuple>
        {
            transform_binary2()
              : transform_binary2::algorithm("transform_binary")
            {}

            template <typename ExPolicy, typename InIter1, typename InIter2,
                typename OutIter, typename F, typename Proj1, typename Proj2>
            static hpx::util::tuple<InIter1, InIter2, OutIter>
            sequential(ExPolicy, InIter1 first1, InIter1 last1,
                InIter2 first2, InIter2 last2,
                OutIter dest, F && f, Proj1 && proj1, Proj2 && proj2)
            {
                return sequential_transform(first1, last1, first2, last2, dest,
                    std::forward<F>(f), std::forward<Proj1>(proj1),
                    std::forward<Proj2>(proj2));
            }

            template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
                typename OutIter, typename F, typename Proj1, typename Proj2>
            static typename util::detail::algorithm_result<
                ExPolicy, hpx::util::tuple<FwdIter1, FwdIter2, OutIter>
            >::type
            parallel(ExPolicy policy, FwdIter1 first1, FwdIter1 last1,
                FwdIter2 first2, FwdIter2 last2, OutIter dest, F && f,
                Proj1 && proj1, Proj2 && proj2)
            {
                typedef hpx::util::zip_iterator<
                        FwdIter1, FwdIter2, OutIter
                    > zip_iterator;
                typedef typename zip_iterator::reference reference;

                return get_iter_tuple(
                    for_each_n<zip_iterator>().call(
                        policy, boost::mpl::false_(),
                        hpx::util::make_zip_iterator(first1, first2, dest),
                        (std::min)(
                            std::distance(first1, last1),
                            std::distance(first2, last2)),
                        [f, proj1, proj2](reference t)
                        {
                            using hpx::util::get;
                            using hpx::util::invoke;
                            get<2>(t) =
                                invoke(f,
                                    invoke(proj1, get<0>(t)),
                                    invoke(proj2, get<1>(t)));
                        }));
            }
        };
        /// \endcond
    }

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
    /// \tparam InIter1     The type of the source iterators for the first
    ///                     range used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
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
    /// \param first1       Refers to the beginning of the first sequence of
    ///                     elements the algorithm will be applied to.
    /// \param last1        Refers to the end of the first sequence of elements
    ///                     the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the second sequence of
    ///                     elements the algorithm will be applied to.
    /// \param last2        Refers to the end of the second sequence of elements
    ///                     the algorithm will be applied to.
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
        typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F,
        typename Proj1 = util::projection_identity,
        typename Proj2 = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        traits::detail::is_iterator<InIter1>::value &&
        traits::detail::is_iterator<InIter2>::value &&
        traits::detail::is_iterator<OutIter>::value &&
        traits::is_projected<Proj1, InIter1>::value &&
        traits::is_projected<Proj2, InIter2>::value &&
        traits::is_indirect_callable<
            F, traits::projected<Proj1, InIter1>,
                traits::projected<Proj2, InIter2>
        >::value)>
    typename util::detail::algorithm_result<
            ExPolicy,
            hpx::util::tagged_tuple<
                tag::in1(InIter1), tag::in2(InIter2), tag::out(OutIter)
            >
        >::type
    transform(ExPolicy && policy,
        InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
        OutIter dest, F && f,
        Proj1 && proj1 = Proj1{}, Proj2 && proj2 = Proj2{})
    {
        typedef typename std::iterator_traits<InIter1>::iterator_category category1;
        typedef typename std::iterator_traits<InIter2>::iterator_category category2;

        static_assert(
            (boost::is_base_of<std::input_iterator_tag, category1>::value),
            "Required at least input iterator.");
        static_assert(
            (boost::is_base_of<std::input_iterator_tag, category2>::value),
            "Required at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, category1>,
            boost::is_same<std::input_iterator_tag, category2>
        >::type is_seq;

        typedef hpx::util::tuple<InIter1, InIter2, OutIter> result_type;

        return hpx::util::make_tagged_tuple<tag::in1, tag::in2, tag::out>(
            detail::transform_binary2<result_type>().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first1, last1, first2, last2, dest, std::forward<F>(f),
                std::forward<Proj1>(proj1), std::forward<Proj2>(proj2)));
    }
}}}

#endif
