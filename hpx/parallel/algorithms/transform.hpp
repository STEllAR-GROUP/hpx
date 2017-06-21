//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/transform.hpp

#if !defined(HPX_PARALLEL_DETAIL_TRANSFORM_JUN_21_2017_0932PM)
#define HPX_PARALLEL_DETAIL_TRANSFORM_JUN_21_2017_0932PM

#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
#include <hpx/traits/get_function_address.hpp>
#include <hpx/traits/get_function_annotation.hpp>
#endif
#include <hpx/traits/is_callable.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/annotated_function.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/tagged_pair.hpp>
#include <hpx/util/tagged_tuple.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/traits/projected.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/foreach_partitioner.hpp>
#include <hpx/parallel/util/transform_loop.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // transform
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename F, typename Proj>
        struct transform_projected
        {
            typename hpx::util::decay<F>::type& f_;
            typename hpx::util::decay<Proj>::type& proj_;

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            auto operator()(Iter curr)
            ->  decltype(
                    hpx::util::invoke(f_, hpx::util::invoke(proj_, *curr))
                )
            {
                return hpx::util::invoke(f_, hpx::util::invoke(proj_, *curr));
            }
        };

        template <typename ExPolicy, typename F, typename Proj>
        struct transform_iteration
        {
            typedef typename hpx::util::decay<ExPolicy>::type execution_policy_type;
            typedef typename hpx::util::decay<F>::type fun_type;
            typedef typename hpx::util::decay<Proj>::type proj_type;

            fun_type f_;
            proj_type proj_;

            template <typename F_, typename Proj_>
            HPX_HOST_DEVICE transform_iteration(F_ && f, Proj_ && proj)
              : f_(std::forward<F_>(f))
              , proj_(std::forward<Proj_>(proj))
            {}

#if defined(HPX_HAVE_CXX11_DEFAULTED_FUNCTIONS) && !defined(__NVCC__) && \
    !defined(__CUDACC__)
            transform_iteration(transform_iteration const&) = default;
            transform_iteration(transform_iteration&&) = default;
#else
            HPX_HOST_DEVICE transform_iteration(transform_iteration const& rhs)
              : f_(rhs.f_)
              , proj_(rhs.proj_)
            {}

            HPX_HOST_DEVICE transform_iteration(transform_iteration && rhs)
              : f_(std::move(rhs.f_))
              , proj_(std::move(rhs.proj_))
            {}
#endif

            HPX_DELETE_COPY_ASSIGN(transform_iteration);
            HPX_DELETE_MOVE_ASSIGN(transform_iteration);

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            std::pair<
                typename hpx::util::tuple_element<
                    0, typename Iter::iterator_tuple_type
                >::type,
                typename hpx::util::tuple_element<
                    1, typename Iter::iterator_tuple_type
                >::type
            >
            execute(Iter part_begin, std::size_t part_size)
            {
                auto iters = part_begin.get_iterator_tuple();
                return util::transform_loop_n<execution_policy_type>(
                    hpx::util::get<0>(iters), part_size,
                    hpx::util::get<1>(iters),
                    transform_projected<F, Proj>{f_, proj_});
            }

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            std::pair<
                typename hpx::util::tuple_element<
                    0, typename Iter::iterator_tuple_type
                >::type,
                typename hpx::util::tuple_element<
                    1, typename Iter::iterator_tuple_type
                >::type
            >
            operator()(Iter part_begin, std::size_t part_size,
                std::size_t /*part_index*/)
            {
                hpx::util::annotate_function annotate(f_);
                (void)annotate;     // suppress warning about unused variable
                return execute(part_begin, part_size);
            }
        };

        template <typename IterPair>
        struct transform
          : public detail::algorithm<transform<IterPair>, IterPair>
        {
            transform()
              : transform::algorithm("transform")
            {}

            template <typename ExPolicy, typename InIter, typename OutIter,
                typename F, typename Proj>
            HPX_HOST_DEVICE
            static std::pair<InIter, OutIter>
            sequential(ExPolicy && policy, InIter first, InIter last,
                OutIter dest, F && f, Proj && proj)
            {
                return util::transform_loop(std::forward<ExPolicy>(policy),
                    first, last, dest, transform_projected<F, Proj>{f, proj});
            }

            template <typename ExPolicy, typename FwdIter, typename OutIter,
                typename F, typename Proj>
            static typename util::detail::algorithm_result<
                ExPolicy, std::pair<FwdIter, OutIter>
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last,
                OutIter dest, F && f, Proj && proj)
            {
                if (first != last)
                {
                    auto f1 = transform_iteration<ExPolicy, F, Proj>(
                        std::forward<F>(f), std::forward<Proj>(proj));

                    return get_iter_pair(
                        util::foreach_partitioner<ExPolicy>::call(
                            std::forward<ExPolicy>(policy),
                            hpx::util::make_zip_iterator(first, dest),
                            std::distance(first, last),
                            std::move(f1), util::projection_identity()));
                }

                return util::detail::algorithm_result<
                        ExPolicy, std::pair<FwdIter, OutIter>
                    >::get(std::make_pair(std::move(first), std::move(dest)));
            }
        };

        /// non_segmented version
        template <typename ExPolicy, typename InIter, typename OutIter, typename F,
            typename Proj>
        typename util::detail::algorithm_result<ExPolicy,
            hpx::util::tagged_pair<tag::in(InIter), tag::out(OutIter)>
        >::type
        transform_(ExPolicy && policy, InIter first, InIter last, OutIter dest,
            F && f, Proj && proj, std::false_type)
        {
            typedef std::integral_constant<bool,
                    execution::is_sequential_execution_policy<ExPolicy>::value ||
                   !hpx::traits::is_forward_iterator<InIter>::value ||
                   !hpx::traits::is_forward_iterator<OutIter>::value
                > is_seq;

            return hpx::util::make_tagged_pair<tag::in, tag::out>(
                detail::transform<std::pair<InIter, OutIter> >().call(
                    std::forward<ExPolicy>(policy), is_seq(),
                    first, last, dest, std::forward<F>(f),
                    std::forward<Proj>(proj)));
        }

        /// forward declare the segmented version
        template <typename ExPolicy, typename InIter, typename OutIter, typename F,
            typename Proj>
        typename util::detail::algorithm_result<ExPolicy,
            hpx::util::tagged_pair<tag::in(InIter), tag::out(OutIter)>
        >::type
        transform_(ExPolicy && policy, InIter first, InIter last, OutIter dest,
            F && f, Proj && proj, std::true_type);
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
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The invocations of \a f in the parallel \a transform algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a transform algorithm returns a
    /// \a hpx::future<tagged_pair<tag::in(FwdIter), tag::out(OutIter)> >
    ///           if the execution policy is of type \a parallel_task_policy
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
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<InIter>::value &&
        hpx::traits::is_iterator<OutIter>::value &&
        traits::is_projected<Proj, InIter>::value)
#if defined(HPX_MSVC) && HPX_MSVC <= 1800     // MSVC12 can't pattern match this
  , HPX_CONCEPT_REQUIRES_(
        traits::is_indirect_callable<
            ExPolicy, F, traits::projected<Proj, InIter>
        >::value)
#endif
    >
    typename util::detail::algorithm_result<ExPolicy,
        hpx::util::tagged_pair<tag::in(InIter), tag::out(OutIter)>
    >::type
    transform(ExPolicy && policy, InIter first, InIter last, OutIter dest,
        F && f, Proj && proj = Proj())
    {
        static_assert(
            (hpx::traits::is_input_iterator<InIter>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<OutIter>::value ||
                hpx::traits::is_input_iterator<OutIter>::value),
            "Requires at least output iterator.");
        typedef hpx::traits::is_segmented_iterator<InIter> is_segmented;
        return detail::transform_(
            std::forward<ExPolicy>(policy), first, last, dest,
            std::forward<F>(f), proj, is_segmented());
    }

    ///////////////////////////////////////////////////////////////////////////
    // transform binary predicate
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename F, typename Proj1, typename Proj2>
        struct transform_binary_projected
        {
            typename hpx::util::decay<F>::type& f_;
            typename hpx::util::decay<Proj1>::type& proj1_;
            typename hpx::util::decay<Proj2>::type& proj2_;

            template <typename Iter1, typename Iter2>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            auto operator()(Iter1 curr1, Iter2 curr2)
            ->  decltype(
                    hpx::util::invoke(f_,
                        hpx::util::invoke(proj1_, *curr1),
                        hpx::util::invoke(proj2_, *curr2))
                )
            {
                return hpx::util::invoke(f_,
                    hpx::util::invoke(proj1_, *curr1),
                    hpx::util::invoke(proj2_, *curr2));
            }
        };

        template <typename ExPolicy, typename F, typename Proj1, typename Proj2>
        struct transform_binary_iteration
        {
            typedef typename hpx::util::decay<ExPolicy>::type execution_policy_type;
            typedef typename hpx::util::decay<F>::type fun_type;
            typedef typename hpx::util::decay<Proj1>::type proj1_type;
            typedef typename hpx::util::decay<Proj2>::type proj2_type;

            fun_type f_;
            proj1_type proj1_;
            proj2_type proj2_;

            template <typename F_, typename Proj1_, typename Proj2_>
            HPX_HOST_DEVICE
            transform_binary_iteration(F_ && f, Proj1_ && proj1, Proj2_ && proj2)
              : f_(std::forward<F_>(f))
              , proj1_(std::forward<Proj1_>(proj1))
              , proj2_(std::forward<Proj2_>(proj2))
            {}

#if defined(HPX_HAVE_CXX11_DEFAULTED_FUNCTIONS) && !defined(__NVCC__) && \
    !defined(__CUDACC__)
            transform_binary_iteration(transform_binary_iteration const&) = default;
            transform_binary_iteration(transform_binary_iteration&&) = default;
#else
            HPX_HOST_DEVICE
            transform_binary_iteration(transform_binary_iteration const& rhs)
              : f_(rhs.f_)
              , proj1_(rhs.proj1_)
              , proj2_(rhs.proj2_)
            {}

            HPX_HOST_DEVICE
            transform_binary_iteration(transform_binary_iteration && rhs)
              : f_(std::move(rhs.f_))
              , proj1_(std::move(rhs.proj1_))
              , proj2_(std::move(rhs.proj2_))
            {}
#endif

            HPX_DELETE_COPY_ASSIGN(transform_binary_iteration);
            HPX_DELETE_MOVE_ASSIGN(transform_binary_iteration);

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            hpx::util::tuple<
                typename hpx::util::tuple_element<
                    0, typename Iter::iterator_tuple_type
                >::type,
                typename hpx::util::tuple_element<
                    1, typename Iter::iterator_tuple_type
                >::type,
                typename hpx::util::tuple_element<
                    2, typename Iter::iterator_tuple_type
                >::type
            >
            operator()(Iter part_begin, std::size_t part_size,
                std::size_t /*part_index*/)
            {
                auto iters = part_begin.get_iterator_tuple();
                return util::transform_binary_loop_n<execution_policy_type>(
                    hpx::util::get<0>(iters), part_size,
                    hpx::util::get<1>(iters), hpx::util::get<2>(iters),
                    transform_binary_projected<F, Proj1, Proj2>{
                        f_, proj1_, proj2_
                    });
            }
        };

        ///////////////////////////////////////////////////////////////////////
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
            sequential(ExPolicy && policy, InIter1 first1, InIter1 last1,
                InIter2 first2, OutIter dest, F && f, Proj1 && proj1,
                Proj2 && proj2)
            {
                return util::transform_binary_loop<ExPolicy>(
                    first1, last1, first2, dest,
                    transform_binary_projected<F, Proj1, Proj2>{
                        f, proj1, proj2
                    });
            }

            template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
                typename OutIter, typename F, typename Proj1, typename Proj2>
            static typename util::detail::algorithm_result<
                ExPolicy, hpx::util::tuple<FwdIter1, FwdIter2, OutIter>
            >::type
            parallel(ExPolicy && policy, FwdIter1 first1, FwdIter1 last1,
                FwdIter2 first2, OutIter dest, F && f, Proj1 && proj1,
                Proj2 && proj2)
            {
                if (first1 != last1)
                {
                    auto f1 = transform_binary_iteration<ExPolicy, F, Proj1,
                        Proj2>(std::forward<F>(f), std::forward<Proj1>(proj1),
                        std::forward<Proj2>(proj2));

                    return get_iter_tuple(
                        util::foreach_partitioner<ExPolicy>::call(
                            std::forward<ExPolicy>(policy),
                            hpx::util::make_zip_iterator(first1, first2, dest),
                            std::distance(first1, last1),
                            std::move(f1), util::projection_identity()));
                }

                return util::detail::algorithm_result<
                        ExPolicy, hpx::util::tuple<FwdIter1, FwdIter2, OutIter>
                    >::get(hpx::util::make_tuple(std::move(first1),
                        std::move(first2), std::move(dest)));
            }
        };

        template <
            typename ExPolicy, typename InIter1, typename InIter2,
            typename OutIter, typename F,
            typename Proj1, typename Proj2>
        typename util::detail::algorithm_result<
            ExPolicy,
            hpx::util::tagged_tuple<
                tag::in1(InIter1), tag::in2(InIter2), tag::out(OutIter)
            >
        >::type
        transform_(ExPolicy && policy,
            InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest, F && f,
            Proj1 && proj1, Proj2 && proj2, std::false_type)
        {
            typedef std::integral_constant<bool,
                    execution::is_sequential_execution_policy<ExPolicy>::value ||
                   !hpx::traits::is_forward_iterator<InIter1>::value ||
                   !hpx::traits::is_forward_iterator<InIter2>::value ||
                   !hpx::traits::is_forward_iterator<OutIter>::value
                > is_seq;

            typedef hpx::util::tuple<InIter1, InIter2, OutIter> result_type;

            return hpx::util::make_tagged_tuple<tag::in1, tag::in2, tag::out>(
                detail::transform_binary<result_type>().call(
                    std::forward<ExPolicy>(policy), is_seq(),
                    first1, last1, first2, dest, std::forward<F>(f),
                    std::forward<Proj1>(proj1), std::forward<Proj2>(proj2)));
        }

        template <
            typename ExPolicy, typename InIter1, typename InIter2,
            typename OutIter, typename F,
            typename Proj1, typename Proj2>
        typename util::detail::algorithm_result<
            ExPolicy,
            hpx::util::tagged_tuple<
                tag::in1(InIter1), tag::in2(InIter2), tag::out(OutIter)
            >
        >::type
        transform_(ExPolicy && policy,
            InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest, F && f,
            Proj1 && proj1, Proj2 && proj2, std::true_type);
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
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The invocations of \a f in the parallel \a transform algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a transform algorithm returns a
    /// \a hpx::future<tagged_tuple<tag::in1(InIter1), tag::in2(InIter2), tag::out(OutIter)> >
    ///           if the execution policy is of type \a parallel_task_policy
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
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<InIter1>::value &&
        hpx::traits::is_iterator<InIter2>::value &&
        hpx::traits::is_iterator<OutIter>::value &&
        traits::is_projected<Proj1, InIter1>::value &&
        traits::is_projected<Proj2, InIter2>::value)
#if defined(HPX_MSVC) && HPX_MSVC <= 1800     // MSVC12 can't pattern match this
  , HPX_CONCEPT_REQUIRES_(
        traits::is_indirect_callable<
            ExPolicy, F, traits::projected<Proj1, InIter1>,
                traits::projected<Proj2, InIter2>
        >::value)
#endif
    >
    typename util::detail::algorithm_result<
        ExPolicy,
        hpx::util::tagged_tuple<
            tag::in1(InIter1), tag::in2(InIter2), tag::out(OutIter)
        >
    >::type
    transform(ExPolicy && policy,
        InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest, F && f,
        Proj1 && proj1 = Proj1(), Proj2 && proj2 = Proj2())
    {
        static_assert(
            (hpx::traits::is_input_iterator<InIter1>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_input_iterator<InIter2>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<OutIter>::value ||
                hpx::traits::is_input_iterator<OutIter>::value),
            "Requires at least output iterator.");

        typedef hpx::traits::is_segmented_iterator<InIter1> is_segmented;

        return detail::transform_(
            std::forward<ExPolicy>(policy), first1, last1, first2, dest,
            std::forward<F>(f), proj1, proj2, is_segmented());
    }

    ///////////////////////////////////////////////////////////////////////////
    // transform binary predicate
    namespace detail
    {
        /// \cond NOINTERNAL
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
            sequential(ExPolicy && policy, InIter1 first1, InIter1 last1,
                InIter2 first2, InIter2 last2,
                OutIter dest, F && f, Proj1 && proj1, Proj2 && proj2)
            {
                return util::transform_binary_loop<ExPolicy>(
                    first1, last1, first2, last2, dest,
                    transform_binary_projected<F, Proj1, Proj2>{
                        f, proj1, proj2
                    });
            }

            template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
                typename OutIter, typename F, typename Proj1, typename Proj2>
            static typename util::detail::algorithm_result<
                ExPolicy, hpx::util::tuple<FwdIter1, FwdIter2, OutIter>
            >::type
            parallel(ExPolicy && policy, FwdIter1 first1, FwdIter1 last1,
                FwdIter2 first2, FwdIter2 last2, OutIter dest, F && f,
                Proj1 && proj1, Proj2 && proj2)
            {
                if (first1 != last1 && first2 != last2)
                {
                    auto f1 = transform_binary_iteration<ExPolicy, F, Proj1, Proj2>(
                        std::forward<F>(f), std::forward<Proj1>(proj1),
                        std::forward<Proj2>(proj2));

                    return get_iter_tuple(
                        util::foreach_partitioner<ExPolicy>::call(
                            std::forward<ExPolicy>(policy),
                            hpx::util::make_zip_iterator(first1, first2, dest),
                            (std::min)(
                                std::distance(first1, last1),
                                std::distance(first2, last2)),
                            std::move(f1), util::projection_identity()));
                }

                return util::detail::algorithm_result<
                        ExPolicy, hpx::util::tuple<FwdIter1, FwdIter2, OutIter>
                    >::get(hpx::util::make_tuple(std::move(first1),
                        std::move(first2), std::move(dest)));
            }
        };

        template <
            typename ExPolicy, typename InIter1, typename InIter2,
            typename OutIter, typename F, typename Proj1, typename Proj2>
        typename util::detail::algorithm_result<
                ExPolicy,
                hpx::util::tagged_tuple<
                    tag::in1(InIter1), tag::in2(InIter2), tag::out(OutIter)
                >
            >::type
        transform_(ExPolicy && policy,
            InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
            OutIter dest, F && f, Proj1 && proj1, Proj2 && proj2, std::false_type)
            {
                typedef std::integral_constant<bool,
                        execution::is_sequential_execution_policy<ExPolicy>::value ||
                       !hpx::traits::is_forward_iterator<InIter1>::value ||
                       !hpx::traits::is_forward_iterator<InIter2>::value ||
                       !hpx::traits::is_forward_iterator<OutIter>::value
                    > is_seq;

                typedef hpx::util::tuple<InIter1, InIter2, OutIter> result_type;

                return hpx::util::make_tagged_tuple<tag::in1, tag::in2, tag::out>(
                    detail::transform_binary2<result_type>().call(
                        std::forward<ExPolicy>(policy), is_seq(),
                        first1, last1, first2, last2, dest, std::forward<F>(f),
                        std::forward<Proj1>(proj1), std::forward<Proj2>(proj2)));
            }
        template <
            typename ExPolicy, typename InIter1, typename InIter2,
            typename OutIter, typename F, typename Proj1, typename Proj2>
        typename util::detail::algorithm_result<
                ExPolicy,
                hpx::util::tagged_tuple<
                    tag::in1(InIter1), tag::in2(InIter2), tag::out(OutIter)
                >
            >::type
        transform_(ExPolicy && policy,
            InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
            OutIter dest, F && f, Proj1 && proj1, Proj2 && proj2, std::true_type);
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
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The invocations of \a f in the parallel \a transform algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note The algorithm will invoke the binary predicate until it reaches
    ///       the end of the shorter of the two given input sequences
    ///
    /// \returns  The \a transform algorithm returns a
    /// \a hpx::future<tagged_tuple<tag::in1(InIter1), tag::in2(InIter2), tag::out(OutIter)> >
    ///           if the execution policy is of type \a parallel_task_policy
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
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<InIter1>::value &&
        hpx::traits::is_iterator<InIter2>::value &&
        hpx::traits::is_iterator<OutIter>::value &&
        traits::is_projected<Proj1, InIter1>::value &&
        traits::is_projected<Proj2, InIter2>::value)
#if defined(HPX_MSVC) && HPX_MSVC <= 1800       // MSVC12 can't pattern match this
  , HPX_CONCEPT_REQUIRES_(
        traits::is_indirect_callable<
            ExPolicy, F,
                traits::projected<Proj1, InIter1>,
                traits::projected<Proj2, InIter2>
        >::value)
#endif
    >
    typename util::detail::algorithm_result<
            ExPolicy,
            hpx::util::tagged_tuple<
                tag::in1(InIter1), tag::in2(InIter2), tag::out(OutIter)
            >
        >::type
    transform(ExPolicy && policy,
        InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
        OutIter dest, F && f, Proj1 && proj1 = Proj1(), Proj2 && proj2 = Proj2())
    {
        static_assert(
            (hpx::traits::is_input_iterator<InIter1>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_input_iterator<InIter2>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<OutIter>::value ||
                hpx::traits::is_input_iterator<OutIter>::value),
            "Requires at least output iterator.");

        typedef hpx::traits::is_segmented_iterator<InIter1> is_segmented;

        return detail::transform_(
            std::forward<ExPolicy>(policy), first1, last1, first2, last2, dest,
            std::forward<F>(f), proj1, proj2, is_segmented());

    }
}}}

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
namespace hpx { namespace traits
{
    template <typename ExPolicy, typename F, typename Proj>
    struct get_function_address<
        parallel::v1::detail::transform_iteration<ExPolicy, F, Proj> >
    {
        static std::size_t call(
            parallel::v1::detail::transform_iteration<ExPolicy, F, Proj> const& f)
                HPX_NOEXCEPT
        {
            return get_function_address<
                    typename hpx::util::decay<F>::type
                >::call(f.f_);
        }
    };

    template <typename ExPolicy, typename F, typename Proj>
    struct get_function_annotation<
        parallel::v1::detail::transform_iteration<ExPolicy, F, Proj> >
    {
        static char const* call(
            parallel::v1::detail::transform_iteration<ExPolicy, F, Proj> const& f)
                HPX_NOEXCEPT
        {
            return get_function_annotation<
                    typename hpx::util::decay<F>::type
                >::call(f.f_);
        }
    };
}}
#endif

#endif
