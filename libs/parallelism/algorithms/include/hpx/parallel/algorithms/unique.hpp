//  Copyright (c) 2017 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/util/tagged_pair.hpp>
#include <hpx/type_support/unused.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/execution/algorithms/detail/is_negative.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/transfer.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/util/compare_projected.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/foreach_partitioner.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/scan_partitioner.hpp>
#include <hpx/parallel/util/transfer.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#if !defined(HPX_HAVE_CXX17_SHARED_PTR_ARRAY)
#include <boost/shared_array.hpp>
#endif

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 {
    /////////////////////////////////////////////////////////////////////////////
    // unique
    namespace detail {
        /// \cond NOINTERNAL

        // sequential unique with projection function
        template <typename FwdIter, typename Pred, typename Proj>
        FwdIter sequential_unique(
            FwdIter first, FwdIter last, Pred&& pred, Proj&& proj)
        {
            return std::unique(first, last,
                util::compare_projected<Pred, Proj>(
                    std::forward<Pred>(pred), std::forward<Proj>(proj)));
        }

        template <typename Iter>
        struct unique : public detail::algorithm<unique<Iter>, Iter>
        {
            unique()
              : unique::algorithm("unique")
            {
            }

            template <typename ExPolicy, typename InIter, typename Pred,
                typename Proj>
            static InIter sequential(
                ExPolicy, InIter first, InIter last, Pred&& pred, Proj&& proj)
            {
                return sequential_unique(first, last, std::forward<Pred>(pred),
                    std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename FwdIter, typename Pred,
                typename Proj>
            static
                typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
                parallel(ExPolicy&& policy, FwdIter first, FwdIter last,
                    Pred&& pred, Proj&& proj)
            {
                typedef hpx::util::zip_iterator<FwdIter, bool*> zip_iterator;
                typedef util::detail::algorithm_result<ExPolicy, FwdIter>
                    algorithm_result;
                typedef typename std::iterator_traits<FwdIter>::difference_type
                    difference_type;

                difference_type count = std::distance(first, last);

                if (count < 2)
                    return algorithm_result::get(std::move(last));

#if defined(HPX_HAVE_CXX17_SHARED_PTR_ARRAY)
                std::shared_ptr<bool[]> flags(new bool[count]);
#else
                boost::shared_array<bool> flags(new bool[count]);
#endif
                std::size_t init = 0u;

                flags[0] = false;

                using hpx::get;
                using hpx::util::make_zip_iterator;
                typedef util::scan_partitioner<ExPolicy, FwdIter, std::size_t,
                    void, util::scan_partitioner_sequential_f3_tag>
                    scan_partitioner_type;

                auto f1 = [pred = std::forward<Pred>(pred),
                              proj = std::forward<Proj>(proj)](
                              zip_iterator part_begin,
                              std::size_t part_size) -> std::size_t {
                    FwdIter base = get<0>(part_begin.get_iterator_tuple());

                    // MSVC complains if pred or proj is captured by ref below
                    util::loop_n<ExPolicy>(++part_begin, part_size,
                        [base, pred, proj](zip_iterator it) mutable {
                            using hpx::util::invoke;

                            bool f = invoke(pred, invoke(proj, *base),
                                invoke(proj, get<0>(*it)));

                            if (!(get<1>(*it) = f))
                                base = get<0>(it.get_iterator_tuple());
                        });

                    // There is no need to return the partition result.
                    // But, the scan_partitioner doesn't support 'void' as
                    // Result1. So, unavoidably return non-meaning value.
                    return 0u;
                };

                std::shared_ptr<FwdIter> dest_ptr =
                    std::make_shared<FwdIter>(first);
                auto f3 =
                    [dest_ptr, flags](zip_iterator part_begin,
                        std::size_t part_size,
                        hpx::shared_future<std::size_t> curr,
                        hpx::shared_future<std::size_t> next) mutable -> void {
                    HPX_UNUSED(flags);

                    curr.get();    // rethrow exceptions
                    next.get();    // rethrow exceptions

                    FwdIter& dest = *dest_ptr;

                    if (dest == get<0>(part_begin.get_iterator_tuple()))
                    {
                        // Self-assignment must be detected.
                        util::loop_n<ExPolicy>(
                            part_begin, part_size, [&dest](zip_iterator it) {
                                if (!get<1>(*it))
                                {
                                    if (dest != get<0>(it.get_iterator_tuple()))
                                        *dest++ = std::move(get<0>(*it));
                                    else
                                        ++dest;
                                }
                            });
                    }
                    else
                    {
                        // Self-assignment can't be performed.
                        util::loop_n<ExPolicy>(
                            part_begin, part_size, [&dest](zip_iterator it) {
                                if (!get<1>(*it))
                                    *dest++ = std::move(get<0>(*it));
                            });
                    }
                };

                auto f4 =
                    [dest_ptr = std::move(dest_ptr), first, count, flags](
                        std::vector<hpx::shared_future<std::size_t>>&&,
                        std::vector<hpx::future<void>>&&) mutable -> FwdIter {
                    if (!flags[count - 1])
                    {
                        std::advance(first, count - 1);
                        if (first != (*dest_ptr))
                            *(*dest_ptr)++ = std::move(*first);
                        else
                            ++(*dest_ptr);
                    }
                    return *dest_ptr;
                };

                return scan_partitioner_type::call(
                    std::forward<ExPolicy>(policy),
                    make_zip_iterator(first, flags.get()), count - 1, init,
                    // step 1 performs first part of scan algorithm
                    std::move(f1),
                    // step 2 propagates the partition results from left
                    // to right
                    [](hpx::shared_future<std::size_t> fut1,
                        hpx::shared_future<std::size_t> fut2) -> std::size_t {
                        fut1.get();
                        fut2.get();    // propagate exceptions
                        // There is no need to propagate the partition
                        // results. But, the scan_partitioner doesn't
                        // support 'void' as Result1. So, unavoidably
                        // return non-meaning value.
                        return 0u;
                    },
                    // step 3 runs final accumulation on each partition
                    std::move(f3),
                    // step 4 use this return value
                    std::move(f4));
            }
        };
        /// \endcond
    }    // namespace detail

    /// Eliminates all but the first element from every consecutive group of
    /// equivalent elements from the range [first, last) and returns a
    /// past-the-end iterator for the new logical end of the range.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first - 1 applications of
    ///         the predicate \a pred and no more than twice as many
    ///         applications of the projection \a proj.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a unique requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     binary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be
    ///                     such that objects of types \a FwdIter can be
    ///                     dereferenced and then implicitly converted to
    ///                     both \a Type1 and \a Type2
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a unique algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a unique algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a unique algorithm returns a \a hpx::future<FwdIter>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a unique algorithm returns the iterator to the new end
    ///           of the range.
    ///
    template <typename ExPolicy, typename FwdIter,
        typename Pred = detail::equal_to,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(hpx::is_execution_policy<ExPolicy>::value&& hpx::
                traits::is_iterator<FwdIter>::value&& traits::is_projected<Proj,
                    FwdIter>::value&& traits::is_indirect_callable<ExPolicy,
                    Pred, traits::projected<Proj, FwdIter>,
                    traits::projected<Proj, FwdIter>>::value)>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type unique(
        ExPolicy&& policy, FwdIter first, FwdIter last, Pred&& pred = Pred(),
        Proj&& proj = Proj())
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
            "Required at least forward iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;

        return detail::unique<FwdIter>().call(std::forward<ExPolicy>(policy),
            is_seq(), first, last, std::forward<Pred>(pred),
            std::forward<Proj>(proj));
    }

    /////////////////////////////////////////////////////////////////////////////
    // unique_copy
    namespace detail {
        /// \cond NOINTERNAL

        // sequential unique_copy with projection function
        template <typename FwdIter, typename OutIter, typename Pred,
            typename Proj>
        std::pair<FwdIter, OutIter> sequential_unique_copy(FwdIter first,
            FwdIter last, OutIter dest, Pred&& pred, Proj&& proj,
            std::true_type)
        {
            if (first == last)
                return std::make_pair(std::move(last), std::move(dest));

            FwdIter base = first;

            *dest++ = *first;

            while (++first != last)
            {
                if (!hpx::util::invoke(pred, hpx::util::invoke(proj, *base),
                        hpx::util::invoke(proj, *first)))
                {
                    base = first;
                    *dest++ = *first;
                }
            }
            return std::make_pair(std::move(last), std::move(dest));
        }

        // sequential unique_copy with projection function
        template <typename InIter, typename OutIter, typename Pred,
            typename Proj>
        std::pair<InIter, OutIter> sequential_unique_copy(InIter first,
            InIter last, OutIter dest, Pred&& pred, Proj&& proj,
            std::false_type)
        {
            if (first == last)
                return std::make_pair(std::move(last), std::move(dest));

            auto base_val = *first;

            *dest++ = base_val;

            while (++first != last)
            {
                if (!hpx::util::invoke(pred, hpx::util::invoke(proj, base_val),
                        hpx::util::invoke(proj, *first)))
                {
                    base_val = *first;
                    *dest++ = base_val;
                }
            }
            return std::make_pair(std::move(last), std::move(dest));
        }

        template <typename IterPair>
        struct unique_copy
          : public detail::algorithm<unique_copy<IterPair>, IterPair>
        {
            unique_copy()
              : unique_copy::algorithm("unique_copy")
            {
            }

            template <typename ExPolicy, typename InIter, typename OutIter,
                typename Pred, typename Proj>
            static std::pair<InIter, OutIter> sequential(ExPolicy, InIter first,
                InIter last, OutIter dest, Pred&& pred, Proj&& proj)
            {
                return sequential_unique_copy(first, last, dest,
                    std::forward<Pred>(pred), std::forward<Proj>(proj),
                    hpx::traits::is_forward_iterator<InIter>());
            }

            template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
                typename Pred, typename Proj>
            static typename util::detail::algorithm_result<ExPolicy,
                std::pair<FwdIter1, FwdIter2>>::type
            parallel(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
                FwdIter2 dest, Pred&& pred, Proj&& proj)
            {
                typedef hpx::util::zip_iterator<FwdIter1, bool*> zip_iterator;
                typedef util::detail::algorithm_result<ExPolicy,
                    std::pair<FwdIter1, FwdIter2>>
                    algorithm_result;
                typedef typename std::iterator_traits<FwdIter1>::difference_type
                    difference_type;

                difference_type count = std::distance(first, last);

                if (count == 0)
                    return algorithm_result::get(
                        std::make_pair(std::move(last), std::move(dest)));

                *dest++ = *first;

                if (count == 1)
                    return algorithm_result::get(
                        std::make_pair(std::move(last), std::move(dest)));

#if defined(HPX_HAVE_CXX17_SHARED_PTR_ARRAY)
                std::shared_ptr<bool[]> flags(new bool[count - 1]);
#else
                boost::shared_array<bool> flags(new bool[count - 1]);
#endif
                std::size_t init = 0;

                using hpx::get;
                using hpx::util::make_zip_iterator;
                typedef util::scan_partitioner<ExPolicy,
                    std::pair<FwdIter1, FwdIter2>, std::size_t>
                    scan_partitioner_type;

                auto f1 = [pred = std::forward<Pred>(pred),
                              proj = std::forward<Proj>(proj)](
                              zip_iterator part_begin,
                              std::size_t part_size) -> std::size_t {
                    FwdIter1 base = get<0>(part_begin.get_iterator_tuple());
                    std::size_t curr = 0;

                    // MSVC complains if pred or proj is captured by ref below
                    util::loop_n<ExPolicy>(++part_begin, part_size,
                        [base, pred, proj, &curr](zip_iterator it) mutable {
                            using hpx::util::invoke;

                            bool f = invoke(pred, invoke(proj, *base),
                                invoke(proj, get<0>(*it)));

                            if (!(get<1>(*it) = f))
                            {
                                base = get<0>(it.get_iterator_tuple());
                                ++curr;
                            }
                        });

                    return curr;
                };
                auto f3 =
                    [dest, flags](zip_iterator part_begin,
                        std::size_t part_size,
                        hpx::shared_future<std::size_t> curr,
                        hpx::shared_future<std::size_t> next) mutable -> void {
                    HPX_UNUSED(flags);

                    next.get();    // rethrow exceptions

                    std::advance(dest, curr.get());
                    util::loop_n<ExPolicy>(++part_begin, part_size,
                        [&dest](zip_iterator it) mutable {
                            if (!get<1>(*it))
                                *dest++ = get<0>(*it);
                        });
                };

                auto f4 =
                    [last, dest, flags](
                        std::vector<hpx::shared_future<std::size_t>>&& items,
                        std::vector<hpx::future<void>>&&) mutable
                    -> std::pair<FwdIter1, FwdIter2> {
                    HPX_UNUSED(flags);

                    std::advance(dest, items.back().get());
                    return std::make_pair(std::move(last), std::move(dest));
                };

                return scan_partitioner_type::call(
                    std::forward<ExPolicy>(policy),
                    //make_zip_iterator(first, flags.get() - 1),
                    make_zip_iterator(first, flags.get() - 1), count - 1, init,
                    // step 1 performs first part of scan algorithm
                    std::move(f1),
                    // step 2 propagates the partition results from left
                    // to right
                    hpx::util::unwrapping(std::plus<std::size_t>()),
                    // step 3 runs final accumulation on each partition
                    std::move(f3),
                    // step 4 use this return value
                    std::move(f4));
            }
        };
        /// \endcond
    }    // namespace detail

    /// Copies the elements from the range [first, last),
    /// to another range beginning at \a dest in such a way that
    /// there are no consecutive equal elements. Only the first element of
    /// each group of equal elements is copied.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first - 1 applications of
    ///         the predicate \a pred and no more than twice as many
    ///         applications of the projection \a proj
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
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a unique_copy requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
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
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     binary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a unique_copy algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a unique_copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a unique_copy algorithm returns a
    ///           \a hpx::future<tagged_pair<tag::in(FwdIter1), tag::out(FwdIter2)> >
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a tagged_pair<tag::in(FwdIter1), tag::out(FwdIter2)>
    ///           otherwise.
    ///           The \a unique_copy algorithm returns the pair of
    ///           the source iterator to \a last, and
    ///           the destination iterator to the end of the \a dest range.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred = detail::equal_to,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(hpx::is_execution_policy<ExPolicy>::value&&
                hpx::traits::is_iterator<FwdIter1>::value&&
                    hpx::traits::is_iterator<FwdIter2>::value&&
                        traits::is_projected<Proj, FwdIter1>::value&&
                            traits::is_indirect_callable<ExPolicy, Pred,
                                traits::projected<Proj, FwdIter1>,
                                traits::projected<Proj, FwdIter1>>::value)>
    typename util::detail::algorithm_result<ExPolicy,
        hpx::util::tagged_pair<tag::in(FwdIter1), tag::out(FwdIter2)>>::type
    unique_copy(ExPolicy&& policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest,
        Pred&& pred = Pred(), Proj&& proj = Proj())
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Required at least forward iterator.");
        static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;
        typedef std::pair<FwdIter1, FwdIter2> result_type;

        return hpx::util::make_tagged_pair<tag::in, tag::out>(
            detail::unique_copy<result_type>().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last, dest,
                std::forward<Pred>(pred), std::forward<Proj>(proj)));
    }
}}}    // namespace hpx::parallel::v1
