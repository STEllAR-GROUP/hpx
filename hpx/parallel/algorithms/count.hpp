//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/count.hpp

#if !defined(HPX_PARALLEL_DETAIL_COUNT_JUNE_17_2014_1154AM)
#define HPX_PARALLEL_DETAIL_COUNT_JUNE_17_2014_1154AM

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/util/bind_back.hpp>
#include <hpx/util/range.hpp>
#include <hpx/util/unwrap.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/traits/projected.hpp>
#include <hpx/parallel/traits/vector_pack_count_bits.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/invoke_projected.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1
{
    ///////////////////////////////////////////////////////////////////////////
    // count
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename ExPolicy, typename Op, typename Proj>
        struct count_iteration
        {
            typedef typename hpx::util::decay<ExPolicy>::type execution_policy_type;
            typedef typename hpx::util::decay<Proj>::type proj_type;
            typedef typename hpx::util::decay<Op>::type op_type;

            op_type op_;
            proj_type proj_;

            template <typename Op_, typename Proj_,
                typename U = typename std::enable_if<
                   !std::is_same<
                        typename hpx::util::decay<Op_>::type, count_iteration
                    >::value
                >::type>
            HPX_HOST_DEVICE count_iteration(Op_ && op, Proj_ && proj)
              : op_(std::forward<Op_>(op)), proj_(std::forward<Proj_>(proj))
            {}

#if !defined(__NVCC__) && !defined(__CUDACC__)
            count_iteration(count_iteration const&) = default;
            count_iteration(count_iteration&&) = default;
#else
            HPX_HOST_DEVICE count_iteration(count_iteration const& rhs)
              : op_(rhs.op_), proj_(rhs.proj_)
            {}

            HPX_HOST_DEVICE count_iteration(count_iteration && rhs)
              : op_(std::move(rhs.op_)), proj_(std::move(rhs.proj_))
            {}
#endif

            count_iteration& operator=(count_iteration const&) = default;
            count_iteration& operator=(count_iteration &&) = default;

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename std::iterator_traits<Iter>::difference_type
            operator()(Iter part_begin, std::size_t part_size)
            {
                typename std::iterator_traits<Iter>::difference_type ret = 0;
                util::loop_n<execution_policy_type>(
                    part_begin, part_size,
                    hpx::util::bind_back(*this, std::ref(ret))
                );
                return ret;
            }

            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            void operator()(Iter curr,
                typename std::iterator_traits<Iter>::difference_type& ret)
            {
                ret += traits::count_bits(hpx::util::invoke(op_,
                                hpx::util::invoke(proj_, *curr)));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Value>
        struct count
          : public detail::algorithm<count<Value>, Value>
        {
            typedef Value difference_type;

            count()
              : count::algorithm("count")
            {}

            template <typename ExPolicy, typename InIter, typename T,
                typename Proj>
            static difference_type
            sequential(ExPolicy && policy, InIter first, InIter last,
                       T const& value, Proj && proj)
            {
                auto f1 =
                    count_iteration<ExPolicy, detail::compare_to<T>, Proj>(
                        detail::compare_to<T>(value),
                        std::forward<Proj>(proj));

                typename std::iterator_traits<InIter>::difference_type ret = 0;

                util::loop(
                    policy, first, last,
                    hpx::util::bind_back(std::move(f1), std::ref(ret)));

                return ret;
            }

            template <typename ExPolicy, typename Iter, typename T,
                typename Proj>
            static typename util::detail::algorithm_result<
                ExPolicy, difference_type
            >::type
            parallel(ExPolicy && policy, Iter first, Iter last,
                T const& value, Proj && proj)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<
                            ExPolicy, difference_type
                        >::get(0);
                }

                auto f1 =
                    count_iteration<ExPolicy, detail::compare_to<T>, Proj>(
                        detail::compare_to<T>(value),
                        std::forward<Proj>(proj));

                return util::partitioner<ExPolicy, difference_type>::call(
                    std::forward<ExPolicy>(policy),
                    first, std::distance(first, last),
                    std::move(f1),
                    hpx::util::unwrapping(
                        [](std::vector<difference_type>&& results)
                        {
                            return util::accumulate_n(
                                hpx::util::begin(results), hpx::util::size(results),
                                difference_type(0), std::plus<difference_type>());
                        }));
            }
        };

        template <typename ExPolicy, typename FwdIter, typename T,
            typename Proj>
        inline typename util::detail::algorithm_result<
            ExPolicy, typename std::iterator_traits<FwdIter>::difference_type
        >::type
        count_(ExPolicy && policy, FwdIter first, FwdIter last, T const& value,
            Proj && proj, std::false_type)
        {
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
            typedef std::integral_constant<bool,
                    parallel::execution::is_sequenced_execution_policy<
                        ExPolicy
                    >::value ||
                   !hpx::traits::is_forward_iterator<FwdIter>::value
                > is_seq;
#else
            typedef parallel::execution::is_sequenced_execution_policy<
                        ExPolicy
                    > is_seq;
#endif

            typedef typename std::iterator_traits<FwdIter>::difference_type
                difference_type;

            return detail::count<difference_type>().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last, value,
                std::forward<Proj>(proj));
        }

        // forward declare the segmented version of this algorithm
        template <typename ExPolicy, typename FwdIter, typename T,
            typename Proj>
        typename util::detail::algorithm_result<
            ExPolicy, typename std::iterator_traits<FwdIter>::difference_type
        >::type
        count_(ExPolicy&& policy, FwdIter first, FwdIter last, T const& value,
            Proj && proj, std::true_type);

        /// \endcond
    }

    /// Returns the number of elements in the range [first, last) satisfying
    /// a specific criteria. This version counts the elements that are equal to
    /// the given \a value.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first comparisons.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the comparisons.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam T           The type of the value to search for (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param value        The value to search for.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparisons in the parallel \a count algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// \note The comparisons in the parallel \a count algorithm invoked with
    ///       an execution policy object of type \a parallel_policy or
    ///       \a parallel_task_policy are permitted to execute in an unordered
    ///       fashion in unspecified threads, and indeterminately sequenced
    ///       within each thread.
    ///
    /// \returns  The \a count algorithm returns a
    ///           \a hpx::future<difference_type> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a difference_type otherwise (where \a difference_type
    ///           is defined by \a std::iterator_traits<FwdIter>::difference_type.
    ///           The \a count algorithm returns the number of elements
    ///           satisfying the given criteria.
    ///
    template <typename ExPolicy, typename FwdIter, typename T,
        typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        traits::is_projected<Proj, FwdIter>::value &&
        hpx::traits::is_iterator<FwdIter>::value)>
    typename util::detail::algorithm_result<
        ExPolicy, typename std::iterator_traits<FwdIter>::difference_type
    >::type
    count(ExPolicy && policy, FwdIter first, FwdIter last, T const& value,
        Proj && proj = Proj())
    {
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
        static_assert(
            (hpx::traits::is_input_iterator<FwdIter>::value),
            "Required at least input iterator.");
#else
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter>::value),
            "Required at least forward iterator.");
#endif

        typedef hpx::traits::is_segmented_iterator<FwdIter> is_segmented;

        return detail::count_(
            std::forward<ExPolicy>(policy), first, last, value,
            std::forward<Proj>(proj), is_segmented());
    }

    ///////////////////////////////////////////////////////////////////////////
    // count_if
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename Value>
        struct count_if
          : public detail::algorithm<count_if<Value>, Value>
        {
            typedef Value difference_type;

            count_if()
              : count_if::algorithm("count_if")
            {}

            template <typename ExPolicy, typename InIter, typename Pred,
                typename Proj>
            static difference_type
            sequential(ExPolicy && policy, InIter first, InIter last, Pred && op,
                Proj && proj)
            {
                auto f1 = count_iteration<ExPolicy, Pred, Proj>(op,
                    std::forward<Proj>(proj));

                typename std::iterator_traits<InIter>::difference_type ret = 0;

                util::loop(
                    policy, first, last,
                    hpx::util::bind_back(std::move(f1), std::ref(ret)));

                return ret;
            }

            template <typename ExPolicy, typename Iter, typename Pred,
                typename Proj>
            static typename util::detail::algorithm_result<
                ExPolicy, difference_type
            >::type
            parallel(ExPolicy && policy, Iter first, Iter last, Pred && op,
                Proj && proj)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<
                            ExPolicy, difference_type
                        >::get(0);
                }

                auto f1 = count_iteration<ExPolicy, Pred, Proj>(op,
                    std::forward<Proj>(proj));

                return util::partitioner<ExPolicy, difference_type>::call(
                    std::forward<ExPolicy>(policy),
                    first, std::distance(first, last),
                    std::move(f1),
                    hpx::util::unwrapping(
                        [](std::vector<difference_type> && results)
                        {
                            return util::accumulate_n(
                                hpx::util::begin(results), hpx::util::size(results),
                                difference_type(0), std::plus<difference_type>());
                        }));
            }
        };

        template <typename ExPolicy, typename FwdIter, typename F,
            typename Proj>
        typename util::detail::algorithm_result<
            ExPolicy, typename std::iterator_traits<FwdIter>::difference_type
        >::type
        count_if_(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
            Proj && proj, std::false_type)
        {
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
            typedef std::integral_constant<bool,
                    parallel::execution::is_sequenced_execution_policy<
                        ExPolicy
                    >::value ||
                   !hpx::traits::is_forward_iterator<FwdIter>::value
                > is_seq;
#else
            typedef parallel::execution::is_sequenced_execution_policy<
                        ExPolicy
                    > is_seq;
#endif

            typedef typename std::iterator_traits<FwdIter>::difference_type
                difference_type;

            return detail::count_if<difference_type>().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, std::forward<F>(f),
                std::forward<Proj>(proj));
        }

        // forward declare the segmented version of this algorithm
        template <typename ExPolicy, typename FwdIter, typename F,
            typename Proj>
        typename util::detail::algorithm_result<
            ExPolicy, typename std::iterator_traits<FwdIter>::difference_type
        >::type
        count_if_(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
            Proj && proj, std::true_type);

        /// \endcond
    }

    /// Returns the number of elements in the range [first, last) satisfying
    /// a specific criteria. This version counts elements for which predicate
    /// \a f returns true.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first applications of
    ///         the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the comparisons.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a count_if requires \a F to meet the
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
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \note The assignments in the parallel \a count_if algorithm invoked with
    ///       an execution policy object of type \a sequenced_policy
    ///       execute in sequential order in the calling thread.
    /// \note The assignments in the parallel \a count_if algorithm invoked with
    ///       an execution policy object of type \a parallel_policy or
    ///       \a parallel_task_policy are permitted to execute in an unordered
    ///       fashion in unspecified threads, and indeterminately sequenced
    ///       within each thread.
    ///
    /// \returns  The \a count_if algorithm returns
    ///           \a hpx::future<difference_type> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a difference_type otherwise (where \a difference_type
    ///           is defined by \a std::iterator_traits<FwdIter>::difference_type.
    ///           The \a count algorithm returns the number of elements
    ///           satisfying the given criteria.
    ///
    template <typename ExPolicy, typename FwdIter, typename F,
        typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<FwdIter>::value &&
        traits::is_projected<Proj, FwdIter>::value &&
        traits::is_indirect_callable<
            ExPolicy, F, traits::projected<Proj, FwdIter>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy, typename std::iterator_traits<FwdIter>::difference_type
    >::type
    count_if(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
        Proj && proj = Proj())
    {
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
        static_assert(
            (hpx::traits::is_input_iterator<FwdIter>::value),
            "Required at least input iterator.");
#else
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter>::value),
            "Required at least forward iterator.");
#endif
        typedef hpx::traits::is_segmented_iterator<FwdIter> is_segmented;

        return detail::count_if_(
            std::forward<ExPolicy>(policy), first, last, std::forward<F>(f),
            std::forward<Proj>(proj), is_segmented());
    }
}}}

#endif
