//  Copyright (c) 2017 Taeguk Kwon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_ALGORITHM_UNIQUE_JUL_07_2017_1805PM)
#define HPX_PARALLEL_ALGORITHM_UNIQUE_JUL_07_2017_1805PM

#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/tagged_pair.hpp>
#include <hpx/util/unused.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/is_negative.hpp>
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/algorithms/detail/transfer.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/traits/projected.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/foreach_partitioner.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/scan_partitioner.hpp>
#include <hpx/parallel/util/transfer.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/shared_array.hpp>

namespace hpx { namespace parallel { inline namespace v1
{
    /////////////////////////////////////////////////////////////////////////////
    // unique_copy
    namespace detail
    {
        /// \cond NOINTERNAL

        // sequential unique_copy with projection function
        template <typename FwdIter, typename OutIter, typename Pred, typename Proj>
        std::pair<FwdIter, OutIter>
        sequential_unique_copy(FwdIter first, FwdIter last, OutIter dest,
            Pred && pred, Proj && proj, std::true_type)
        {
            if (first == last)
                return std::make_pair(std::move(last), std::move(dest));

            FwdIter base = first;

            *dest++ = *first;

            while (++first != last)
            {
                if (!hpx::util::invoke(pred,
                    hpx::util::invoke(proj, *base),
                    hpx::util::invoke(proj, *first)))
                {
                    base = first;
                    *dest++ = *first;
                }
            }
            return std::make_pair(std::move(last), std::move(dest));
        }

        // sequential unique_copy with projection function
        template <typename InIter, typename OutIter, typename Pred, typename Proj>
        std::pair<InIter, OutIter>
        sequential_unique_copy(InIter first, InIter last, OutIter dest,
            Pred && pred, Proj && proj, std::false_type)
        {
            if (first == last)
                return std::make_pair(std::move(last), std::move(dest));

            auto base_val = *first;

            *dest++ = base_val;

            while (++first != last)
            {
                if (!hpx::util::invoke(pred,
                    hpx::util::invoke(proj, base_val),
                    hpx::util::invoke(proj, *first)))
                {
                    base_val = *first;
                    *dest++ = base_val;
                }
            }
            return std::make_pair(std::move(last), std::move(dest));
        }

        template <typename IterPair>
        struct unique_copy : public detail::algorithm<unique_copy<IterPair>, IterPair>
        {
            unique_copy()
              : unique_copy::algorithm("unique_copy")
            {}

            template <typename ExPolicy, typename InIter, typename OutIter,
                typename Pred, typename Proj = util::projection_identity>
            static std::pair<InIter, OutIter>
            sequential(ExPolicy, InIter first, InIter last, OutIter dest,
                Pred && pred, Proj && proj/* = Proj()*/)
            {
                return sequential_unique_copy(first, last, dest,
                    std::forward<Pred>(pred), std::forward<Proj>(proj),
                    hpx::traits::is_forward_iterator<InIter>());
            }

            template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
                typename Pred, typename Proj = util::projection_identity>
            static typename util::detail::algorithm_result<
                ExPolicy, std::pair<FwdIter1, FwdIter2>
            >::type
            parallel(ExPolicy && policy, FwdIter1 first, FwdIter1 last,
                FwdIter2 dest, Pred && pred, Proj && proj/* = Proj()*/)
            {
                typedef hpx::util::zip_iterator<FwdIter1, bool*> zip_iterator;
                typedef util::detail::algorithm_result<
                    ExPolicy, std::pair<FwdIter1, FwdIter2>
                > result;
                typedef typename std::iterator_traits<FwdIter1>::difference_type
                    difference_type;

                if (first == last)
                    return result::get(
                        std::make_pair(std::move(last), std::move(dest)));

                difference_type count = std::distance(first, last);

                *dest++ = *first;

                if (count == 1)
                    return result::get(
                        std::make_pair(std::move(last), std::move(dest)));

                boost::shared_array<bool> flags(new bool[count - 1]);
                std::size_t init = 0;

                using hpx::util::get;
                using hpx::util::make_zip_iterator;
                typedef util::scan_partitioner<
                        ExPolicy, std::pair<FwdIter1, FwdIter2>, std::size_t
                    > scan_partitioner_type;

                auto f1 =
                    [pred, proj, flags, policy]
                    (
                       zip_iterator part_begin, std::size_t part_size
                    )   -> std::size_t
                    {
                        HPX_UNUSED(flags);
                        HPX_UNUSED(policy);

                        FwdIter1 base = get<0>(part_begin.get_iterator_tuple());
                        std::size_t curr = 0;

                        // MSVC complains if pred or proj is captured by ref below
                        util::loop_n<ExPolicy>(
                            ++part_begin, part_size,
                            [base, pred, proj, &curr](zip_iterator it) mutable
                            {
                                using hpx::util::invoke;

                                bool f = invoke(pred,
                                    invoke(proj, *base),
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
                    [dest, flags, policy](
                        zip_iterator part_begin, std::size_t part_size,
                        hpx::shared_future<std::size_t> curr,
                        hpx::shared_future<std::size_t> next
                    ) mutable -> void
                    {
                        HPX_UNUSED(flags);
                        HPX_UNUSED(policy);

                        next.get();     // rethrow exceptions

                        std::advance(dest, curr.get());
                        util::loop_n<ExPolicy>(
                            ++part_begin, part_size,
                            [&dest](zip_iterator it) mutable
                            {
                                if(!get<1>(*it))
                                    *dest++ = get<0>(*it);
                            });
                    };

                return scan_partitioner_type::call(
                    std::forward<ExPolicy>(policy),
                    make_zip_iterator(first, flags.get()),
                    count - 1, init,
                    // step 1 performs first part of scan algorithm
                    std::move(f1),
                    // step 2 propagates the partition results from left
                    // to right
                    hpx::util::unwrapped(std::plus<std::size_t>()),
                    // step 3 runs final accumulation on each partition
                    std::move(f3),
                    // step 4 use this return value
                    [last, dest, flags](
                        std::vector<hpx::shared_future<std::size_t> > && items,
                        std::vector<hpx::future<void> > &&) mutable
                    ->  std::pair<FwdIter1, FwdIter2>
                    {
                        HPX_UNUSED(flags);

                        std::advance(dest, items.back().get());
                        return std::make_pair(std::move(last), std::move(dest));
                    });
            }
        };
        /// \endcond
    }

    /// Copies the elements from the range [first, last),
    /// to another range beginning at \a dest in such a way that
    /// there are no consecutive equal elements. Only the first element of
    /// each group of equal elements is copied.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first - 1 applications of
    ///         the predicate \a pred.
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
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<FwdIter1>::value &&
        hpx::traits::is_iterator<FwdIter2>::value &&
        traits::is_projected<Proj, FwdIter1>::value &&
        traits::is_indirect_callable<
            ExPolicy, Pred,
            traits::projected<Proj, FwdIter1>,
            traits::projected<Proj, FwdIter1>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy, hpx::util::tagged_pair<
        tag::in(FwdIter1), tag::out(FwdIter2)>
    >::type
    unique_copy(ExPolicy&& policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest,
        Pred && pred = Pred(), Proj && proj = Proj())
    {
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
        static_assert(
            (hpx::traits::is_input_iterator<FwdIter1>::value),
            "Required at least input iterator.");
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
            "Required at least forward iterator.");
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;
#endif

        typedef std::pair<FwdIter1, FwdIter2> result_type;

        return hpx::util::make_tagged_pair<tag::in, tag::out>(
            detail::unique_copy<result_type>().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, dest, std::forward<Pred>(pred),
                std::forward<Proj>(proj)));
    }
}}}

#endif
