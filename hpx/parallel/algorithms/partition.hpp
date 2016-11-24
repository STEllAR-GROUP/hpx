//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_ALGORITHM_PARTITION_SEP_24_2016_1055AM)
#define HPX_PARALLEL_ALGORITHM_PARTITION_SEP_24_2016_1055AM

#include <hpx/config.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/dataflow.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/invoke.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/traits/projected.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/util/invoke_projected.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <boost/exception_ptr.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <list>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // stable_partition
    namespace detail
    {
        /// \cond NOINTERNAL
        struct stable_partition_helper
        {
            template <typename ExPolicy, typename RandIter, typename F, typename Proj>
            hpx::future<RandIter>
            operator()(ExPolicy && policy, RandIter first, RandIter last,
                std::size_t size, F f, Proj proj, std::size_t chunks)
            {
                typedef typename hpx::util::decay<ExPolicy>::type::executor_type
                    executor_type;
                typedef typename hpx::parallel::executor_traits<executor_type>
                    executor_traits;
                if (chunks < 2)
                {
                    return executor_traits::async_execute(
                        policy.executor(),
                        [first, last, f, proj]() -> RandIter
                        {
                            return std::stable_partition(
                                first, last,
                                util::invoke_projected<F, Proj>(f, proj));
                        });
                }

                std::size_t mid_point = size / 2;
                chunks /= 2;

                RandIter mid = first;
                std::advance(mid, mid_point);

                hpx::future<RandIter> left = executor_traits::async_execute(
                    policy.executor(), *this, policy, first, mid, mid_point,
                    f, proj, chunks);
                hpx::future<RandIter> right = executor_traits::async_execute(
                    policy.executor(), *this, policy, mid, last, size - mid_point,
                    f, proj, chunks);

                return
                    dataflow(
                        policy.executor(),
                        [mid](
                            hpx::future<RandIter> && left,
                            hpx::future<RandIter> && right
                        ) -> RandIter
                        {
                            if (left.has_exception() || right.has_exception())
                            {
                                std::list<boost::exception_ptr> errors;
                                if(left.has_exception())
                                    hpx::parallel::util::detail::
                                    handle_local_exceptions<ExPolicy>::call(
                                        left.get_exception_ptr(), errors);
                                if(right.has_exception())
                                    hpx::parallel::util::detail::
                                    handle_local_exceptions<ExPolicy>::call(
                                        right.get_exception_ptr(), errors);

                                if (!errors.empty())
                                {
                                    boost::throw_exception(
                                        exception_list(std::move(errors)));
                                }
                            }
                            RandIter first = left.get();
                            RandIter last = right.get();

                            std::rotate(first, mid, last);

                            // for some library implementations std::rotate
                            // does not return the new middle point
                            std::advance(first, std::distance(mid, last));
                            return first;
                        },
                        std::move(left), std::move(right));
            }
        };

        template <typename Iter>
        struct stable_partition
          : public detail::algorithm<stable_partition<Iter>, Iter>
        {
            stable_partition()
              : stable_partition::algorithm("stable_partition")
            {}

            template <typename ExPolicy, typename BidirIter, typename F,
                typename Proj>
            static BidirIter
            sequential(ExPolicy && policy, BidirIter first, BidirIter last,
                F && f, Proj && proj)
            {
                return std::stable_partition(first, last,
                    util::invoke_projected<F, Proj>(
                        std::forward<F>(f), std::forward<Proj>(proj)
                    ));
            }

            template <typename ExPolicy, typename RandIter, typename F,
                typename Proj>
            static typename util::detail::algorithm_result<
                ExPolicy, RandIter
            >::type
            parallel(ExPolicy && policy, RandIter first, RandIter last,
                F && f, Proj && proj)
            {
                typedef util::detail::algorithm_result<ExPolicy, RandIter>
                    algorithm_result;
                typedef typename std::iterator_traits<RandIter>::difference_type
                    difference_type;

                future<RandIter> result;

                try {
                    difference_type size = std::distance(first, last);

                    if (size == 0)
                    {
                        result = hpx::make_ready_future(std::move(last));
                    }

                    typedef typename hpx::util::decay<ExPolicy>::type::executor_type
                        executor_type;
                    typedef typename
                        hpx::util::decay<ExPolicy>::type::executor_parameters_type
                        parameters_type;

                    typedef executor_parameter_traits<parameters_type> traits;
                    typedef executor_information_traits<executor_type> info_traits;

                    std::size_t const cores =
                        info_traits::processing_units_count(policy.executor(),
                                policy.parameters());
                    std::size_t max_chunks = traits::maximal_number_of_chunks(
                        policy.parameters(), policy.executor(), cores, size);

                    result = stable_partition_helper()(
                        std::forward<ExPolicy>(policy), first, last, size,
                        std::forward<F>(f), std::forward<Proj>(proj),
                        size == 1 ? 1 : (std::min)(std::size_t(size), max_chunks));
                }
                catch (...) {
                    result = hpx::make_exceptional_future<RandIter>(
                        boost::current_exception());
                }

                if (result.has_exception())
                {
                    return algorithm_result::get(
                        detail::handle_exception<ExPolicy, RandIter>::call(
                            std::move(result)));
                }

                return algorithm_result::get(std::move(result));
            }
        };
        /// \endcond
    }

    /// Permutes the elements in the range [first, last) such that there exists
    /// an iterator i such that for every iterator j in the range [first, i)
    /// INVOKE(f, INVOKE (proj, *j)) != false, and for every iterator k in the
    /// range [i, last), INVOKE(f, INVOKE (proj, *k)) == false
    ///
    /// \note   Complexity: At most (last - first) * log(last - first) swaps,
    ///         but only linear number of swaps if there is enough extra memory.
    ///         Exactly \a last - \a first applications of the predicate and
    ///         projection.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the invocations of \a f.
    /// \tparam BidirIter   The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
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
    /// \param f            Unary predicate which returns true if the element
    ///                     should be ordered before other elements.
    ///                     Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). The signature
    ///                     of this predicate should be equivalent to:
    ///                     \code
    ///                     bool fun(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&.
    ///                     The type \a Type must be such that an object of
    ///                     type \a BidirIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a f is invoked.
    ///
    /// The invocations of \a f in the parallel \a stable_partition algorithm
    /// invoked with an execution policy object of type
    /// \a sequential_execution_policy executes in sequential order in the
    /// calling thread.
    ///
    /// The invocations of \a f in the parallel \a stable_partition algorithm
    /// invoked with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a stable_partition algorithm returns an iterator i such that
    ///           for every iterator j in the range [first, i), f(*j) != false
    ///           INVOKE(f, INVOKE(proj, *j)) != false, and for every iterator
    ///           k in the range [i, last), f(*k) == false
    ///           INVOKE(f, INVOKE (proj, *k)) == false. The relative order of
    ///           the elements in both groups is preserved.
    ///           If the execution policy is of type \a parallel_task_execution_policy
    ///           the algorithm returns a future<> referring to this iterator.
    ///
    template <typename ExPolicy, typename BidirIter, typename F,
        typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<BidirIter>::value &&
        traits::is_projected<Proj, BidirIter>::value)
#if defined(HPX_MSVC) && HPX_MSVC <= 1800       // MSVC12 can't pattern match this
  , HPX_CONCEPT_REQUIRES_(
        traits::is_indirect_callable<
            ExPolicy, F, traits::projected<Proj, BidirIter>
        >::value)
#endif
    >
    typename util::detail::algorithm_result<ExPolicy, BidirIter>::type
    stable_partition(ExPolicy && policy, BidirIter first, BidirIter last,
        F && f, Proj && proj = Proj())
    {
        static_assert(
            (hpx::traits::is_bidirectional_iterator<BidirIter>::value),
            "Requires at least bidirectional iterator.");

        typedef std::integral_constant<bool,
                is_sequential_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_random_access_iterator<BidirIter>::value
            > is_seq;

        return detail::stable_partition<BidirIter>().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last,
                std::forward<F>(f), std::forward<Proj>(proj));
    }
}}}

#endif
