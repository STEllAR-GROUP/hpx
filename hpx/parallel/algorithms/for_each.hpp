//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/for_each.hpp

#if !defined(HPX_PARALLEL_DETAIL_FOR_EACH_MAY_29_2014_0932PM)
#define HPX_PARALLEL_DETAIL_FOR_EACH_MAY_29_2014_0932PM

#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
#include <hpx/traits/get_function_address.hpp>
#include <hpx/traits/get_function_annotation.hpp>
#endif
#include <hpx/traits/is_callable.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/traits/is_value_proxy.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/util/annotated_function.hpp>
#include <hpx/util/identity.hpp>
#include <hpx/util/invoke.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/is_negative.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/traits/projected.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/foreach_partitioner.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    /// forward declaration of for_each
    template <typename ExPolicy, typename InIter, typename F,
        typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<InIter>::value &&
        parallel::traits::is_projected<Proj, InIter>::value)
#if (!defined(__NVCC__) && !defined(__CUDACC__)) || defined(__CUDA_ARCH__)
    , HPX_CONCEPT_REQUIRES_(
        parallel::traits::is_indirect_callable<
            ExPolicy, F, traits::projected<Proj, InIter>
        >::value)
#endif
    >
    typename util::detail::algorithm_result<ExPolicy, InIter>::type
    for_each(ExPolicy && policy, InIter first, InIter last, F && f,
        Proj && proj = Proj());
    ///////////////////////////////////////////////////////////////////////////
    // for_each_n
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename F, typename Proj>
        struct invoke_projected
        {
            typename hpx::util::decay<F>::type& f_;
            typename hpx::util::decay<Proj>::type& proj_;

            template <typename T>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename std::enable_if<
               !hpx::traits::is_value_proxy<T>::value
            >::type
            call(T && t)
            {
                T && tmp = std::forward<T>(t);
                hpx::util::invoke(f_, hpx::util::invoke(proj_, tmp));
            }

            template <typename T>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename std::enable_if<
                hpx::traits::is_value_proxy<T>::value
            >::type
            call(T && t)
            {
                auto tmp = hpx::util::invoke(proj_, std::forward<T>(t));
                hpx::util::invoke_r<void>(f_,tmp);
            }

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            void operator()(Iter curr)
            {
                call(*curr);
            }
        };

        template <typename ExPolicy, typename F, typename Proj>
        struct for_each_iteration
        {
            typedef typename hpx::util::decay<ExPolicy>::type execution_policy_type;
            typedef typename hpx::util::decay<F>::type fun_type;
            typedef typename hpx::util::decay<Proj>::type proj_type;

            fun_type f_;
            proj_type proj_;

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            void execute(Iter part_begin, std::size_t part_size)
            {
                util::loop_n<execution_policy_type>(part_begin, part_size,
                    invoke_projected<fun_type, proj_type>{f_, proj_});
            }

            template <typename F_, typename Proj_>
            HPX_HOST_DEVICE for_each_iteration(F_ && f, Proj_ && proj)
              : f_(std::forward<F_>(f))
              , proj_(std::forward<Proj_>(proj))
            {}

#if defined(HPX_HAVE_CXX11_DEFAULTED_FUNCTIONS) && !defined(__NVCC__) && \
    !defined(__CUDACC__)
            for_each_iteration(for_each_iteration const&) = default;
            for_each_iteration(for_each_iteration&&) = default;
#else
            HPX_HOST_DEVICE for_each_iteration(for_each_iteration const& rhs)
              : f_(rhs.f_)
              , proj_(rhs.proj_)
            {}

            HPX_HOST_DEVICE for_each_iteration(for_each_iteration && rhs)
              : f_(std::move(rhs.f_))
              , proj_(std::move(rhs.proj_))
            {}
#endif

            HPX_DELETE_COPY_ASSIGN(for_each_iteration);
            HPX_DELETE_MOVE_ASSIGN(for_each_iteration);

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            void operator()(Iter part_begin, std::size_t part_size,
                std::size_t /*part_index*/)
            {
                hpx::util::annotate_function annotate(f_);
                (void)annotate;     // suppress warning about unused variable
                execute(part_begin, part_size);
            }
        };

        template <typename Iter>
        struct for_each_n : public detail::algorithm<for_each_n<Iter>, Iter>
        {
            for_each_n()
              : for_each_n::algorithm("for_each_n")
            {}

            template <typename ExPolicy, typename InIter, typename F,
                typename Proj = util::projection_identity>
            HPX_HOST_DEVICE
            static Iter
            sequential(ExPolicy && policy, InIter first, std::size_t count,
                F && f, Proj && proj/* = Proj()*/)
            {
                return util::loop_n<ExPolicy>(first, count,
                    invoke_projected<F, Proj>{f, proj});
            }

            template <typename ExPolicy, typename InIter, typename F,
                typename Proj = util::projection_identity>
            static typename util::detail::algorithm_result<
                ExPolicy, InIter
            >::type
            parallel(ExPolicy && policy, InIter first, std::size_t count,
                F && f, Proj && proj/* = Proj()*/)
            {
                if (count != 0)
                {
                    auto f1 = for_each_iteration<ExPolicy, F, Proj>(
                        std::forward<F>(f), std::forward<Proj>(proj));

                    return util::foreach_partitioner<ExPolicy>::call(
                        std::forward<ExPolicy>(policy), first, count,
                        std::move(f1), util::projection_identity());
                }

                return util::detail::algorithm_result<ExPolicy, InIter>::get(
                    std::move(first));
            }
        };
        /// Non Segmented implementation
        template <typename ExPolicy, typename InIter, typename Size, typename F,
            typename Proj = util::projection_identity>
        typename util::detail::algorithm_result<ExPolicy, InIter>::type
        for_each_n_(ExPolicy && policy, InIter first, Size count, F && f,
            std::false_type, Proj && proj = Proj())
        {
            typedef std::integral_constant<bool,
                    execution::is_sequential_execution_policy<ExPolicy>::value ||
                   !hpx::traits::is_forward_iterator<InIter>::value
                > is_seq;
            return detail::for_each_n<InIter>().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, std::size_t(count), std::forward<F>(f),
                std::forward<Proj>(proj));
        }
        /// Segmented implementaion using for_each.
        template <typename ExPolicy, typename InIter, typename Size, typename F,
            typename Proj = util::projection_identity>
        typename util::detail::algorithm_result<ExPolicy, InIter>::type
        for_each_n_(ExPolicy && policy, InIter first, Size count, F && f,
            std::true_type, Proj && proj = Proj())
        {
            auto last = first;
            detail::advance(last,std::size_t(count));
            return for_each(
                std::forward<ExPolicy>(policy),
                first, last, std::forward<F>(f), std::forward<Proj>(proj));
        }
        /// \endcond
    }

    /// Applies \a f to the result of dereferencing every iterator in the range
    /// [first, first + count), starting from first and proceeding to
    /// first + count - 1.
    ///
    /// \note   Complexity: Applies \a f exactly \a count times.
    ///
    /// If \a f returns a result, the result is ignored.
    ///
    /// If the type of \a first satisfies the requirements of a mutable
    /// iterator, \a f may apply non-constant functions through the
    /// dereferenced iterator.
    ///
    /// Unlike its sequential form, the parallel overload of
    /// \a for_each does not return a copy of its \a Function parameter,
    /// since parallelization may not permit efficient state
    /// accumulation.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a for_each requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     <ignored> pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. The
    ///                     type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a f is invoked.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a for_each_n algorithm returns a
    ///           \a hpx::future<InIter> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a InIter
    ///           otherwise.
    ///           It returns \a first + \a count for non-negative values of
    ///           \a count and \a first for negative values.
    ///

    template <typename ExPolicy, typename InIter, typename Size, typename F,
        typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<InIter>::value &&
        parallel::traits::is_projected<Proj, InIter>::value)
#if (!defined(__NVCC__) && !defined(__CUDACC__)) || defined(__CUDA_ARCH__)
    , HPX_CONCEPT_REQUIRES_(
        parallel::traits::is_indirect_callable<
            ExPolicy, F, traits::projected<Proj, InIter>
        >::value)
#endif
    >
    typename util::detail::algorithm_result<ExPolicy, InIter>::type
    for_each_n(ExPolicy && policy, InIter first, Size count, F && f,
        Proj && proj = Proj())
    {
        static_assert(
            (hpx::traits::is_input_iterator<InIter>::value),
            "Requires at least input iterator.");

        // if count is representing a negative value, we do nothing
        if (detail::is_negative(count))
        {
            return util::detail::algorithm_result<ExPolicy, InIter>::get(
                std::move(first));
        }

        typedef hpx::traits::is_segmented_iterator<InIter> is_segmented;

        return detail::for_each_n_(policy, first, count, f, is_segmented(),proj);
    }

    ///////////////////////////////////////////////////////////////////////////
    // for_each
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename Iter>
        struct for_each : public detail::algorithm<for_each<Iter>, Iter>
        {
            for_each()
              : for_each::algorithm("for_each")
            {}

            template <typename ExPolicy, typename InIter, typename F,
                typename Proj>
            static InIter
            sequential(ExPolicy && policy, InIter first, InIter last, F && f,
                Proj && proj)
            {
                return util::loop(std::forward<ExPolicy>(policy), first, last,
                    invoke_projected<F, Proj>{f, proj});
            }

            template <typename ExPolicy, typename InIter, typename F,
                typename Proj>
            static typename util::detail::algorithm_result<
                ExPolicy, InIter
            >::type
            parallel(ExPolicy && policy, InIter first, InIter last, F && f,
                Proj && proj)
            {
                if (first != last)
                {
                    auto f1 = for_each_iteration<ExPolicy, F, Proj>(
                        std::forward<F>(f), std::forward<Proj>(proj));

                    return util::foreach_partitioner<ExPolicy>::call(
                        std::forward<ExPolicy>(policy),
                        first, std::distance(first, last),
                        std::move(f1), util::projection_identity());
                }

                return util::detail::algorithm_result<ExPolicy, InIter>::get(
                    std::move(first));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // non-segmented implementation
        template <typename ExPolicy, typename InIter, typename F,
            typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy, InIter>::type
        for_each_(ExPolicy && policy, InIter first, InIter last, F && f,
            Proj && proj, std::false_type)
        {
            typedef std::integral_constant<bool,
                    parallel::execution::is_sequential_execution_policy<
                        ExPolicy
                    >::value ||
                   !hpx::traits::is_forward_iterator<InIter>::value
                > is_seq;

            if (first == last)
            {
                typedef util::detail::algorithm_result<ExPolicy, InIter> result;
                return result::get(std::move(last));
            }

            return for_each<InIter>().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, std::forward<F>(f), std::forward<Proj>(proj));
        }

        // forward declare the segmented version of this algorithm
        template <typename ExPolicy, typename SegIter, typename F,
            typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy, SegIter>::type
        for_each_(ExPolicy && policy, SegIter first, SegIter last, F && f,
            Proj && proj, std::true_type);

        /// \endcond
    }

    /// Applies \a f to the result of dereferencing every iterator in the
    /// range [first, last).
    ///
    /// \note   Complexity: Applies \a f exactly \a last - \a first times.
    ///
    /// If \a f returns a result, the result is ignored.
    ///
    /// If the type of \a first satisfies the requirements of a mutable
    /// iterator, \a f may apply non-constant functions through the
    /// dereferenced iterator.
    ///
    /// Unlike its sequential form, the parallel overload of
    /// \a for_each does not return a copy of its \a Function parameter,
    /// since parallelization may not permit efficient state
    /// accumulation.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a for_each requires \a F to meet the
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
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     <ignored> pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. The
    ///                     type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a f is invoked.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a for_each algorithm returns a
    ///           \a hpx::future<InIter> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a InIter
    ///           otherwise.
    ///           It returns \a last.
    ///


    // FIXME : is_indirect_callable does not work properly when compiling
    //         Cuda host code

    template <typename ExPolicy, typename InIter, typename F,
        typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<InIter>::value &&
        parallel::traits::is_projected<Proj, InIter>::value)
#if (!defined(__NVCC__) && !defined(__CUDACC__)) || defined(__CUDA_ARCH__)
  , HPX_CONCEPT_REQUIRES_(
        parallel::traits::is_indirect_callable<
            ExPolicy, F, traits::projected<Proj, InIter>
        >::value)
#endif
    >
    typename util::detail::algorithm_result<ExPolicy, InIter>::type
    for_each(ExPolicy && policy, InIter first, InIter last, F && f,
        Proj && proj)
    {
        static_assert(
            (hpx::traits::is_input_iterator<InIter>::value),
            "Requires at least input iterator.");

        typedef hpx::traits::is_segmented_iterator<InIter> is_segmented;

        return detail::for_each_(
            std::forward<ExPolicy>(policy), first, last,
            std::forward<F>(f), std::forward<Proj>(proj), is_segmented());
    }
}}}

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
namespace hpx { namespace traits
{
    template <typename ExPolicy, typename F, typename Proj>
    struct get_function_address<
        parallel::v1::detail::for_each_iteration<ExPolicy, F, Proj> >
    {
        static std::size_t call(
            parallel::v1::detail::for_each_iteration<ExPolicy, F, Proj> const& f)
                HPX_NOEXCEPT
        {
            return get_function_address<
                    typename hpx::util::decay<F>::type
                >::call(f.f_);
        }
    };

    template <typename ExPolicy, typename F, typename Proj>
    struct get_function_annotation<
        parallel::v1::detail::for_each_iteration<ExPolicy, F, Proj> >
    {
        static char const* call(
            parallel::v1::detail::for_each_iteration<ExPolicy, F, Proj> const& f)
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
