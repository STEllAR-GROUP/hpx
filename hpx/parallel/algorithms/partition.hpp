//  Copyright (c) 2014-2017 Hartmut Kaiser
//  Copyright (c)      2017 Taeguk Kwon
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
#include <hpx/util/tagged_tuple.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/executors/execution.hpp>
#include <hpx/parallel/executors/execution_information.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/traits/projected.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/util/invoke_projected.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/scan_partitioner.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>
#include <hpx/util/unused.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iterator>
#include <list>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/shared_array.hpp>

namespace hpx { namespace parallel { inline namespace v1
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
                if (chunks < 2)
                {
                    return execution::async_execute(
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

                hpx::future<RandIter> left = execution::async_execute(
                    policy.executor(), *this, policy, first, mid,
                    mid_point, f, proj, chunks);
                hpx::future<RandIter> right = execution::async_execute(
                    policy.executor(), *this, policy, mid, last,
                    size - mid_point, f, proj, chunks);

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
                                std::list<std::exception_ptr> errors;
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
                                    throw exception_list(std::move(errors));
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

                    typedef typename
                        hpx::util::decay<ExPolicy>::type::executor_parameters_type
                        parameters_type;

                    typedef executor_parameter_traits<parameters_type> traits;

                    std::size_t const cores =
                        execution::processing_units_count(policy.executor(),
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
                        std::current_exception());
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
    /// \a sequenced_policy executes in sequential order in the
    /// calling thread.
    ///
    /// The invocations of \a f in the parallel \a stable_partition algorithm
    /// invoked with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a stable_partition algorithm returns an iterator i such that
    ///           for every iterator j in the range [first, i), f(*j) != false
    ///           INVOKE(f, INVOKE(proj, *j)) != false, and for every iterator
    ///           k in the range [i, last), f(*k) == false
    ///           INVOKE(f, INVOKE (proj, *k)) == false. The relative order of
    ///           the elements in both groups is preserved.
    ///           If the execution policy is of type \a parallel_task_policy
    ///           the algorithm returns a future<> referring to this iterator.
    ///
    template <typename ExPolicy, typename BidirIter, typename F,
        typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<BidirIter>::value &&
        traits::is_projected<Proj, BidirIter>::value &&
        traits::is_indirect_callable<
            ExPolicy, F, traits::projected<Proj, BidirIter>
        >::value)>
    typename util::detail::algorithm_result<ExPolicy, BidirIter>::type
    stable_partition(ExPolicy && policy, BidirIter first, BidirIter last,
        F && f, Proj && proj = Proj())
    {
        static_assert(
            (hpx::traits::is_bidirectional_iterator<BidirIter>::value),
            "Requires at least bidirectional iterator.");

        typedef std::integral_constant<bool,
                execution::is_sequenced_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_random_access_iterator<BidirIter>::value
            > is_seq;

        return detail::stable_partition<BidirIter>().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last,
                std::forward<F>(f), std::forward<Proj>(proj));
    }

    /////////////////////////////////////////////////////////////////////////////
    // partition_copy
    namespace detail
    {
        /// \cond NOINTERNAL

        // sequential partition_copy with projection function
        template <typename InIter, typename OutIter1, typename OutIter2,
            typename Pred, typename Proj>
        hpx::util::tuple<InIter, OutIter1, OutIter2>
        sequential_partition_copy(InIter first, InIter last,
            OutIter1 dest_true, OutIter2 dest_false, Pred && pred, Proj && proj)
        {
            while (first != last)
            {
                if (hpx::util::invoke(pred, hpx::util::invoke(proj, *first)))
                    *dest_true++ = *first;
                else
                    *dest_false++ = *first;
                first++;
            }
            return hpx::util::make_tuple(std::move(last),
                std::move(dest_true), std::move(dest_false));
        }

        template <typename IterTuple>
        struct partition_copy
          : public detail::algorithm<partition_copy<IterTuple>, IterTuple>
        {
            partition_copy()
              : partition_copy::algorithm("partition_copy")
            {}

            template <typename ExPolicy, typename InIter,
                typename OutIter1, typename OutIter2,
                typename Pred, typename Proj = util::projection_identity>
            static hpx::util::tuple<InIter, OutIter1, OutIter2>
            sequential(ExPolicy, InIter first, InIter last,
                OutIter1 dest_true, OutIter2 dest_false,
                Pred && pred, Proj && proj)
            {
                return sequential_partition_copy(first, last, dest_true, dest_false,
                    std::forward<Pred>(pred), std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename FwdIter1,
                typename FwdIter2, typename FwdIter3,
                typename Pred, typename Proj = util::projection_identity>
            static typename util::detail::algorithm_result<
                ExPolicy, hpx::util::tuple<FwdIter1, FwdIter2, FwdIter3>
            >::type
            parallel(ExPolicy && policy, FwdIter1 first, FwdIter1 last,
                FwdIter2 dest_true, FwdIter3 dest_false, Pred && pred, Proj && proj)
            {
                typedef hpx::util::zip_iterator<FwdIter1, bool*> zip_iterator;
                typedef util::detail::algorithm_result<
                    ExPolicy, hpx::util::tuple<FwdIter1, FwdIter2, FwdIter3>
                > result;
                typedef typename std::iterator_traits<FwdIter1>::difference_type
                    difference_type;
                typedef std::pair<std::size_t, std::size_t>
                    output_iterator_offset;

                if (first == last)
                    return result::get(hpx::util::make_tuple(
                        last, dest_true, dest_false));

                difference_type count = std::distance(first, last);

                boost::shared_array<bool> flags(new bool[count]);
                output_iterator_offset init = { 0, 0 };

                using hpx::util::get;
                using hpx::util::make_zip_iterator;
                typedef util::scan_partitioner<
                        ExPolicy, hpx::util::tuple<FwdIter1, FwdIter2, FwdIter3>,
                        output_iterator_offset
                    > scan_partitioner_type;

                auto f1 =
                    [pred, proj, flags, policy]
                    (
                       zip_iterator part_begin, std::size_t part_size
                    )   -> output_iterator_offset
                    {
                        HPX_UNUSED(flags);
                        HPX_UNUSED(policy);

                        std::size_t true_count = 0;

                        // MSVC complains if pred or proj is captured by ref below
                        util::loop_n<ExPolicy>(
                            part_begin, part_size,
                            [pred, proj, &true_count](zip_iterator it) mutable
                            {
                                using hpx::util::invoke;
                                bool f = invoke(pred, invoke(proj, get<0>(*it)));

                                if ((get<1>(*it) = f))
                                    ++true_count;
                            });

                        return output_iterator_offset(
                            true_count, part_size - true_count);
                    };
                auto f3 =
                    [dest_true, dest_false, flags, policy](
                        zip_iterator part_begin, std::size_t part_size,
                        hpx::shared_future<output_iterator_offset> curr,
                        hpx::shared_future<output_iterator_offset> next
                    ) mutable -> void
                    {
                        HPX_UNUSED(flags);
                        HPX_UNUSED(policy);

                        next.get();     // rethrow exceptions

                        output_iterator_offset offset = curr.get();
                        std::size_t count_true = get<0>(offset);
                        std::size_t count_false = get<1>(offset);
                        std::advance(dest_true, count_true);
                        std::advance(dest_false, count_false);

                        util::loop_n<ExPolicy>(
                            part_begin, part_size,
                            [&dest_true, &dest_false](zip_iterator it) mutable
                            {
                                if(get<1>(*it))
                                    *dest_true++ = get<0>(*it);
                                else
                                    *dest_false++ = get<0>(*it);
                            });
                    };

                return scan_partitioner_type::call(
                    std::forward<ExPolicy>(policy),
                    make_zip_iterator(first, flags.get()), count, init,
                    // step 1 performs first part of scan algorithm
                    std::move(f1),
                    // step 2 propagates the partition results from left
                    // to right
                    hpx::util::unwrapped(
                        [](output_iterator_offset const& prev_sum,
                            output_iterator_offset const& curr)
                        -> output_iterator_offset
                        {
                            return output_iterator_offset(
                                get<0>(prev_sum) + get<0>(curr),
                                get<1>(prev_sum) + get<1>(curr));
                        }),
                    // step 3 runs final accumulation on each partition
                    std::move(f3),
                    // step 4 use this return value
                    [last, dest_true, dest_false, count, flags](
                        std::vector<
                            hpx::shared_future<output_iterator_offset>
                        > && items,
                        std::vector<hpx::future<void> > &&) mutable
                    ->  hpx::util::tuple<FwdIter1, FwdIter2, FwdIter3>
                    {
                        HPX_UNUSED(flags);
                        HPX_UNUSED(count);

                        output_iterator_offset count_pair = items.back().get();
                        std::size_t count_true = get<0>(count_pair);
                        std::size_t count_false = get<1>(count_pair);
                        std::advance(dest_true, count_true);
                        std::advance(dest_false, count_false);

                        return hpx::util::make_tuple(last, dest_true, dest_false);
                    });
            }
        };
        /// \endcond
    }

    /// Copies the elements in the range, defined by [first, last),
    /// to two different ranges depending on the value returned by
    /// the predicate \a pred. The elements, that satisfy the predicate \a pred,
    /// are copied to the range beginning at \a dest_true. The rest of
    /// the elements are copied to the range beginning at \a dest_false.
    /// The order of the elements is preserved.
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
    ///                     destination range for the elements that satisfy
    ///                     the predicate \a pred (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter3    The type of the iterator representing the
    ///                     destination range for the elements that don't satisfy
    ///                     the predicate \a pred (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a partition_copy requires \a Pred to meet
    ///                     the requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest_true    Refers to the beginning of the destination range for
    ///                     the elements that satisfy the predicate \a pred.
    /// \param dest_false   Refers to the beginning of the destination range for
    ///                     the elements that don't satisfy the predicate \a pred.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the sequence
    ///                     specified by [first, last). This is an unary predicate
    ///                     for partitioning the source iterators. The signature of
    ///                     this predicate should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a partition_copy algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a partition_copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a partition_copy algorithm returns a
    /// \a hpx::future<tagged_tuple<tag::in(InIter), tag::out1(OutIter1), tag::out2(OutIter2)> >
    ///           if the execution policy is of type \a parallel_task_policy
    ///           and returns
    /// \a tagged_tuple<tag::in(InIter), tag::out1(OutIter1), tag::out2(OutIter2)>
    ///           otherwise.
    ///           The \a partition_copy algorithm returns the tuple of
    ///           the source iterator \a last,
    ///           the destination iterator to the end of the \a dest_true range, and
    ///           the destination iterator to the end of the \a dest_false range.
    ///
    template <typename ExPolicy, typename FwdIter1,
        typename FwdIter2, typename FwdIter3,
        typename Pred, typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<FwdIter1>::value &&
        hpx::traits::is_iterator<FwdIter2>::value &&
        hpx::traits::is_iterator<FwdIter3>::value &&
        traits::is_projected<Proj, FwdIter1>::value &&
        traits::is_indirect_callable<
            ExPolicy, Pred, traits::projected<Proj, FwdIter1>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy, hpx::util::tagged_tuple<
        tag::in(FwdIter1), tag::out1(FwdIter2), tag::out2(FwdIter3)>
    >::type
    partition_copy(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
        FwdIter2 dest_true, FwdIter3 dest_false, Pred && pred,
        Proj && proj = Proj())
    {
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
        static_assert(
            (hpx::traits::is_input_iterator<FwdIter1>::value),
            "Required at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<FwdIter2>::value ||
                hpx::traits::is_forward_iterator<FwdIter2>::value) &&
            (hpx::traits::is_output_iterator<FwdIter3>::value ||
                hpx::traits::is_forward_iterator<FwdIter3>::value),
            "Requires at least output iterator.");

        typedef std::integral_constant<bool,
                execution::is_sequenced_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<FwdIter1>::value ||
               !hpx::traits::is_forward_iterator<FwdIter2>::value ||
               !hpx::traits::is_forward_iterator<FwdIter3>::value
            > is_seq;
#else
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Required at least forward iterator.");
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter3>::value),
            "Requires at least forward iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;
#endif

        typedef hpx::util::tuple<FwdIter1, FwdIter2, FwdIter3> result_type;

        return hpx::util::make_tagged_tuple<tag::in, tag::out1, tag::out2>(
            detail::partition_copy<result_type>().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, dest_true, dest_false, std::forward<Pred>(pred),
                std::forward<Proj>(proj)));
    }
}}}

#endif
