//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2015 Daniel Bourgeois
//  Copyright (c) 2016-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/copy.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    // clang-format off

    /// Copies the elements in the range, defined by [first, last), to another
    /// range beginning at \a dest.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
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
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    ///
    /// The assignments in the parallel \a copy algorithm invoked with an
    /// execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy algorithm returns a
    ///           \a hpx::future<FwdIter2> >
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2> otherwise.
    ///           The \a copy algorithm returns the pair of the input iterator
    ///           \a last and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    copy(ExPolicy&& policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest);

    /// Copies the elements in the range [first, first + count), starting from
    /// first and proceeding to first + count - 1., to another range beginning
    /// at dest.
    ///
    /// \note   Complexity: Performs exactly \a count assignments, if
    ///         count > 0, no assignments otherwise.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    ///
    /// The assignments in the parallel \a copy_n algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a copy_n algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy_n algorithm returns a
    ///           \a hpx::future<FwdIter2>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2
    ///           otherwise.
    ///           The \a copy algorithm returns the pair of the input iterator
    ///           forwarded to the first element after the last in the input
    ///           sequence and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Size, typename FwdIter2>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    copy_n(ExPolicy&& policy, FwdIter1 first, Size count, FwdIter2 dest);

    /// Copies the elements in the range, defined by [first, last), to another
    /// range beginning at \a dest. Copies only the elements for which the
    /// predicate \a f returns true. The order of the elements that are not
    /// removed is preserved.
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
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a copy_if requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
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
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// The assignments in the parallel \a copy_if algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a copy_if algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy_if algorithm returns a
    ///           \a hpx::future<FwdIter2> >
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2 otherwise.
    ///           The \a copy algorithm returns the pair of the input iterator
    ///           forwarded to the first element after the last in the input
    ///           sequence and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename F>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy, FwdIter2> >::type
    copy_if(ExPolicy&& policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest,
        Pred&& pred);

    // clang-format on
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/execution/algorithms/detail/is_negative.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/algorithms/detail/transfer.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/foreach_partitioner.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/result_types.hpp>
#include <hpx/parallel/util/scan_partitioner.hpp>
#include <hpx/parallel/util/transfer.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>
#include <hpx/type_support/unused.hpp>

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

    ///////////////////////////////////////////////////////////////////////////
    // copy
    namespace detail {

        struct copy_iteration
        {
            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE void operator()(
                Iter part_begin, std::size_t part_size, std::size_t)
            {
                using hpx::util::get;
                auto iters = part_begin.get_iterator_tuple();
                util::copy_n(get<0>(iters), part_size, get<1>(iters));
            }
        };

        template <typename IterPair>
        struct copy : public detail::algorithm<copy<IterPair>, IterPair>
        {
            copy()
              : copy::algorithm("copy")
            {
            }

            template <typename ExPolicy, typename InIter, typename Sent,
                typename OutIter>
            static util::in_out_result<InIter, OutIter> sequential(
                ExPolicy, InIter first, Sent last, OutIter dest)
            {
                util::in_out_result<InIter, OutIter> result =
                    util::copy(first, last, dest);
                util::copy_synchronize(first, dest);
                return result;
            }

            template <typename ExPolicy, typename FwdIter1, typename Sent1,
                typename FwdIter2>
            static typename util::detail::algorithm_result<ExPolicy,
                util::in_out_result<FwdIter1, FwdIter2>>::type
            parallel(
                ExPolicy&& policy, FwdIter1 first, Sent1 last, FwdIter2 dest)
            {
#if defined(HPX_COMPUTE_DEVICE_CODE)
                HPX_ASSERT(false);
                typename util::detail::algorithm_result<ExPolicy,
                    util::in_out_result<FwdIter1, FwdIter2>>::type* dummy =
                    nullptr;
                return std::move(*dummy);
#else
                typedef hpx::util::zip_iterator<FwdIter1, FwdIter2>
                    zip_iterator;

                return util::detail::get_in_out_result(
                    util::foreach_partitioner<ExPolicy>::call(
                        std::forward<ExPolicy>(policy),
                        hpx::util::make_zip_iterator(first, dest),
                        detail::distance(first, last), copy_iteration(),
                        [](zip_iterator&& last) -> zip_iterator {
                            using hpx::util::get;
                            auto iters = last.get_iterator_tuple();
                            util::copy_synchronize(
                                get<0>(iters), get<1>(iters));
                            return std::move(last);
                        }));
#endif
            }
        };

#if defined(HPX_COMPUTE_DEVICE_CODE)
        template <typename FwdIter1, typename FwdIter2, typename Enable = void>
        struct copy_iter : public copy<util::in_out_result<FwdIter1, FwdIter2>>
        {
        };
#else
        ///////////////////////////////////////////////////////////////////////
        template <typename FwdIter1, typename FwdIter2, typename Enable = void>
        struct copy_iter;

        template <typename FwdIter1, typename FwdIter2>
        struct copy_iter<FwdIter1, FwdIter2,
            typename std::enable_if<
                iterators_are_segmented<FwdIter1, FwdIter2>::value>::type>
          : public copy<util::in_out_result<
                typename hpx::traits::segmented_iterator_traits<
                    FwdIter1>::local_iterator,
                typename hpx::traits::segmented_iterator_traits<
                    FwdIter2>::local_iterator>>
        {
        };

        template <typename FwdIter1, typename FwdIter2>
        struct copy_iter<FwdIter1, FwdIter2,
            typename std::enable_if<
                iterators_are_not_segmented<FwdIter1, FwdIter2>::value>::type>
          : public copy<util::in_out_result<FwdIter1, FwdIter2>>
        {
        };
#endif

        ///////////////////////////////////////////////////////////////////////
        template <typename I, typename O>
        O get_second_element(util::in_out_result<I, O>&& p)
        {
            return p.out;
        }

        template <typename I, typename O>
        hpx::future<O> get_second_element(
            hpx::future<util::in_out_result<I, O>>&& f)
        {
            return lcos::make_future<O>(std::move(f),
                [](util::in_out_result<I, O>&& p) { return p.out; });
        }
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        HPX_CONCEPT_REQUIRES_(
            execution::is_execution_policy<ExPolicy>::value&&
            hpx::traits::is_iterator<FwdIter1>::value&&
            hpx::traits::is_iterator<FwdIter2>::value)>
    // clang-format on
    HPX_DEPRECATED_V(
        1, 6, "hpx::parallel::copy is deprecated, use hpx::copy instead")
        typename util::detail::algorithm_result<ExPolicy,
            util::in_out_result<FwdIter1, FwdIter2>>::type
        copy(ExPolicy&& policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest)
    {
        return detail::transfer<detail::copy_iter<FwdIter1, FwdIter2>>(
            std::forward<ExPolicy>(policy), first, last, dest);
    }

    /////////////////////////////////////////////////////////////////////////////
    // copy_n
    namespace detail {

        // sequential copy_n
        template <typename IterPair>
        struct copy_n : public detail::algorithm<copy_n<IterPair>, IterPair>
        {
            copy_n()
              : copy_n::algorithm("copy_n")
            {
            }

            template <typename ExPolicy, typename InIter, typename OutIter>
            static util::in_out_result<InIter, OutIter> sequential(
                ExPolicy, InIter first, std::size_t count, OutIter dest)
            {
                return util::copy_n(first, count, dest);
            }

            template <typename ExPolicy, typename FwdIter1, typename FwdIter2>
            static typename util::detail::algorithm_result<ExPolicy,
                util::in_out_result<FwdIter1, FwdIter2>>::type
            parallel(ExPolicy&& policy, FwdIter1 first, std::size_t count,
                FwdIter2 dest)
            {
                typedef hpx::util::zip_iterator<FwdIter1, FwdIter2>
                    zip_iterator;

                return util::detail::get_in_out_result(
                    util::foreach_partitioner<ExPolicy>::call(
                        std::forward<ExPolicy>(policy),
                        hpx::util::make_zip_iterator(first, dest), count,
                        [](zip_iterator part_begin, std::size_t part_size,
                            std::size_t) {
                            using hpx::util::get;

                            auto iters = part_begin.get_iterator_tuple();
                            util::copy_n(
                                get<0>(iters), part_size, get<1>(iters));
                        },
                        [](zip_iterator&& last) -> zip_iterator {
                            using hpx::util::get;
                            auto iters = last.get_iterator_tuple();
                            util::copy_synchronize(
                                get<0>(iters), get<1>(iters));
                            return std::move(last);
                        }));
            }
        };
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename Size,
        typename FwdIter2,
        HPX_CONCEPT_REQUIRES_(
            execution::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter1>::value &&
            hpx::traits::is_iterator<FwdIter2>::value)>
    // clang-format on
    HPX_DEPRECATED_V(
        1, 6, "hpx::parallel::copy_n is deprecated, use hpx::copy_n instead")
        typename util::detail::algorithm_result<ExPolicy,
            util::in_out_result<FwdIter1, FwdIter2>>::type
        copy_n(ExPolicy&& policy, FwdIter1 first, Size count, FwdIter2 dest)
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Required at least forward iterator.");
        static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;

        // if count is representing a negative value, we do nothing
        if (detail::is_negative(count))
        {
            return util::detail::algorithm_result<ExPolicy,
                util::in_out_result<FwdIter1, FwdIter2>>::
                get(util::in_out_result<FwdIter1, FwdIter2>{first, dest});
        }

        return detail::copy_n<util::in_out_result<FwdIter1, FwdIter2>>().call(
            std::forward<ExPolicy>(policy), is_seq(), first, std::size_t(count),
            dest);
    }

    /////////////////////////////////////////////////////////////////////////////
    // copy_if
    namespace detail {

        // sequential copy_if with projection function
        template <typename InIter1, typename InIter2, typename OutIter,
            typename Pred, typename Proj>
        inline util::in_out_result<InIter1, OutIter> sequential_copy_if(
            InIter1 first, InIter2 last, OutIter dest, Pred&& pred, Proj&& proj)
        {
            while (first != last)
            {
                if (hpx::util::invoke(pred, hpx::util::invoke(proj, *first)))
                    *dest++ = *first;
                first++;
            }
            return util::in_out_result<InIter1, OutIter>{
                std::move(first), std::move(dest)};
        }

        template <typename IterPair>
        struct copy_if : public detail::algorithm<copy_if<IterPair>, IterPair>
        {
            copy_if()
              : copy_if::algorithm("copy_if")
            {
            }

            template <typename ExPolicy, typename InIter1, typename InIter2,
                typename OutIter, typename Pred,
                typename Proj = util::projection_identity>
            static util::in_out_result<InIter1, OutIter> sequential(ExPolicy,
                InIter1 first, InIter2 last, OutIter dest, Pred&& pred,
                Proj&& proj /* = Proj()*/)
            {
                return sequential_copy_if(first, last, dest,
                    std::forward<Pred>(pred), std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
                typename FwdIter3, typename Pred,
                typename Proj = util::projection_identity>
            static typename util::detail::algorithm_result<ExPolicy,
                util::in_out_result<FwdIter1, FwdIter3>>::type
            parallel(ExPolicy&& policy, FwdIter1 first, FwdIter2 last,
                FwdIter3 dest, Pred&& pred, Proj&& proj /* = Proj()*/)
            {
                typedef hpx::util::zip_iterator<FwdIter1, bool*> zip_iterator;
                typedef util::detail::algorithm_result<ExPolicy,
                    util::in_out_result<FwdIter1, FwdIter3>>
                    result;
                typedef typename std::iterator_traits<FwdIter1>::difference_type
                    difference_type;

                if (first == last)
                {
                    return result::get(util::in_out_result<FwdIter1, FwdIter3>{
                        std::move(first), std::move(dest)});
                }

                difference_type count = detail::distance(first, last);

#if defined(HPX_HAVE_CXX17_SHARED_PTR_ARRAY)
                std::shared_ptr<bool[]> flags(new bool[count]);
#else
                boost::shared_array<bool> flags(new bool[count]);
#endif
                std::size_t init = 0;

                using hpx::util::get;
                using hpx::util::make_zip_iterator;
                typedef util::scan_partitioner<ExPolicy,
                    util::in_out_result<FwdIter1, FwdIter3>, std::size_t>
                    scan_partitioner_type;

                auto f1 = [pred = std::forward<Pred>(pred),
                              proj = std::forward<decltype(proj)>(proj)](
                              zip_iterator part_begin,
                              std::size_t part_size) -> std::size_t {
                    std::size_t curr = 0;

                    // MSVC complains if proj is captured by ref below
                    util::loop_n<ExPolicy>(part_begin, part_size,
                        [&pred, proj, &curr](zip_iterator it) mutable {
                            bool f = hpx::util::invoke(
                                pred, hpx::util::invoke(proj, get<0>(*it)));

                            if ((get<1>(*it) = f))
                                ++curr;
                        });

                    return curr;
                };
                auto f3 = [dest, flags](zip_iterator part_begin,
                              std::size_t part_size,
                              hpx::shared_future<std::size_t> curr,
                              hpx::shared_future<std::size_t> next) mutable {
                    HPX_UNUSED(flags);

                    next.get();    // rethrow exceptions

                    std::advance(dest, curr.get());
                    util::loop_n<ExPolicy>(part_begin, part_size,
                        [&dest](zip_iterator it) mutable {
                            if (get<1>(*it))
                                *dest++ = get<0>(*it);
                        });
                };

                auto f4 =
                    [first, dest, flags](
                        std::vector<hpx::shared_future<std::size_t>>&& items,
                        std::vector<hpx::future<void>>&&) mutable
                    -> util::in_out_result<FwdIter1, FwdIter3> {
                    HPX_UNUSED(flags);

                    std::advance(dest, items.back().get());
                    std::advance(first, items.back().get());
                    return util::in_out_result<FwdIter1, FwdIter3>{
                        std::move(first), std::move(dest)};
                };

                return scan_partitioner_type::call(
                    std::forward<ExPolicy>(policy),
                    make_zip_iterator(first, flags.get()), count, init,
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
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred,
        HPX_CONCEPT_REQUIRES_(
            hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter1>::value &&
            hpx::traits::is_iterator<FwdIter2>::value &&
            hpx::traits::is_invocable<Pred,
                typename std::iterator_traits<FwdIter1>::value_type
            >::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(
        1, 6, "hpx::parallel::copy_if is deprecated, use hpx::copy_if instead")
        typename util::detail::algorithm_result<ExPolicy,
            util::in_out_result<FwdIter1, FwdIter2>>::type
        copy_if(ExPolicy&& policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest,
            Pred&& pred)
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Required at least forward iterator.");
        static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;

        return detail::copy_if<util::in_out_result<FwdIter1, FwdIter2>>().call(
            std::forward<ExPolicy>(policy), is_seq(), first, last, dest,
            std::forward<Pred>(pred), util::projection_identity{});
    }
}}}    // namespace hpx::parallel::v1

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::copy
    HPX_INLINE_CONSTEXPR_VARIABLE struct copy_t final
      : hpx::functional::tag<copy_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter2>::type
        tag_invoke(hpx::copy_t, ExPolicy&& policy, FwdIter1 first,
            FwdIter1 last, FwdIter2 dest)
        {
            return parallel::v1::detail::get_second_element(
                parallel::v1::detail::transfer<
                    parallel::v1::detail::copy_iter<FwdIter1, FwdIter2>>(
                    std::forward<ExPolicy>(policy), first, last, dest));
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value&&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend FwdIter2 tag_invoke(
            hpx::copy_t, FwdIter1 first, FwdIter1 last, FwdIter2 dest)
        {
            return parallel::v1::detail::get_second_element(
                parallel::v1::detail::transfer<
                    parallel::v1::detail::copy_iter<FwdIter1, FwdIter2>>(
                    hpx::parallel::execution::seq, first, last, dest));
        }
    } copy{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::copy_n
    HPX_INLINE_CONSTEXPR_VARIABLE struct copy_n_t final
      : hpx::functional::tag<copy_n_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename Size,
            typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter2>::type
        tag_invoke(hpx::copy_n_t, ExPolicy&& policy, FwdIter1 first, Size count,
            FwdIter2 dest)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Required at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::v1::detail::is_negative(count))
            {
                return hpx::parallel::util::detail::algorithm_result<ExPolicy,
                    FwdIter2>::get(std::move(dest));
            }

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::get_second_element(
                hpx::parallel::v1::detail::copy_n<
                    hpx::parallel::util::in_out_result<FwdIter1, FwdIter2>>()
                    .call(std::forward<ExPolicy>(policy), is_seq(), first,
                        std::size_t(count), dest));
        }

        // clang-format off
        template <typename FwdIter1, typename Size, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend FwdIter2 tag_invoke(
            hpx::copy_n_t, FwdIter1 first, Size count, FwdIter2 dest)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Required at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::v1::detail::is_negative(count))
            {
                return hpx::parallel::util::detail::algorithm_result<
                    hpx::parallel::execution::sequenced_policy,
                    FwdIter2>::get(std::move(dest));
            }

            return hpx::parallel::v1::detail::get_second_element(
                hpx::parallel::v1::detail::copy_n<
                    hpx::parallel::util::in_out_result<FwdIter1, FwdIter2>>()
                    .call(hpx::parallel::execution::seq, std::true_type{},
                        first, std::size_t(count), dest));
        }
    } copy_n{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::copy_if
    HPX_INLINE_CONSTEXPR_VARIABLE struct copy_if_t final
      : hpx::functional::tag<copy_if_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Pred,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::traits::is_invocable<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter2>::type
        tag_invoke(hpx::copy_if_t, ExPolicy&& policy, FwdIter1 first,
            FwdIter1 last, FwdIter2 dest, Pred&& pred)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Required at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::get_second_element(
                hpx::parallel::v1::detail::copy_if<
                    hpx::parallel::util::in_out_result<FwdIter1, FwdIter2>>()
                    .call(std::forward<ExPolicy>(policy), is_seq(), first, last,
                        dest, std::forward<Pred>(pred),
                        hpx::parallel::util::projection_identity{}));
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2, typename Pred,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::traits::is_invocable<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type
                >::value
            )>
        // clang-format on
        friend FwdIter2 tag_invoke(hpx::copy_if_t, FwdIter1 first,
            FwdIter1 last, FwdIter2 dest, Pred&& pred)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Required at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::get_second_element(
                hpx::parallel::v1::detail::copy_if<
                    hpx::parallel::util::in_out_result<FwdIter1, FwdIter2>>()
                    .call(hpx::parallel::execution::seq, std::true_type{},
                        first, last, dest, std::forward<Pred>(pred),
                        hpx::parallel::util::projection_identity{}));
        }
    } copy_if{};

}    // namespace hpx

#endif    // DOXYGEN
