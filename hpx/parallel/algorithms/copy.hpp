//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2015 Daniel Bourgeois
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/copy.hpp

#if !defined(HPX_PARALLEL_DETAIL_COPY_MAY_30_2014_0317PM)
#define HPX_PARALLEL_DETAIL_COPY_MAY_30_2014_0317PM

#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/tagged_pair.hpp>

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
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/shared_array.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // copy
    namespace detail
    {
        /// \cond NOINTERNAL

        struct copy_iteration
        {
            template <typename Iter>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            void operator()(std::size_t, Iter part_begin, std::size_t part_size)
            {
                using hpx::util::get;
                auto const& iters = part_begin.get_iterator_tuple();
                util::copy_n_helper(get<0>(iters), part_size, get<1>(iters));
            }
        };

        template <typename IterPair>
        struct copy
          : public detail::algorithm<copy<IterPair>, IterPair>
        {
            copy()
              : copy::algorithm("copy")
            {}

            template <typename ExPolicy, typename InIter, typename OutIter>
            static std::pair<InIter, OutIter>
            sequential(ExPolicy, InIter first, InIter last, OutIter dest)
            {
                std::pair<InIter, OutIter> result =
                    util::copy_helper(first, last, dest);
                util::copy_synchronize(first, dest);
                return result;
            }

            template <typename ExPolicy, typename FwdIter, typename OutIter>
            static typename util::detail::algorithm_result<
                ExPolicy, std::pair<FwdIter, OutIter>
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last,
                OutIter dest)
            {
                typedef hpx::util::zip_iterator<FwdIter, OutIter> zip_iterator;

                return get_iter_pair(
                    util::foreach_partitioner<ExPolicy>::call(
                        std::forward<ExPolicy>(policy),
                        hpx::util::make_zip_iterator(first, dest),
                        std::distance(first, last),
                        copy_iteration(),
                        [](zip_iterator && last) -> zip_iterator
                        {
                            using hpx::util::get;
                            auto const& iters = last.get_iterator_tuple();
                            util::copy_synchronize(get<0>(iters), get<1>(iters));
                            return std::move(last);
                        }));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template<typename InIter, typename OutIter, typename Enable = void>
        struct copy_iter;

        template <typename InIter, typename OutIter>
        struct copy_iter<
            InIter, OutIter,
            typename std::enable_if<
                iterators_are_segmented<InIter, OutIter>::value
            >::type>
          : public copy<std::pair<
                typename hpx::traits::segmented_iterator_traits<InIter>
                    ::local_iterator,
                typename hpx::traits::segmented_iterator_traits<OutIter>
                    ::local_iterator
            > >
        {};

        template<typename InIter, typename OutIter>
        struct copy_iter<
            InIter, OutIter,
            typename std::enable_if<
                iterators_are_not_segmented<InIter, OutIter>::value
            >::type>
          : public copy<std::pair<InIter, OutIter> >
        {};

        /// \endcond
    }

    /// Copies the elements in the range, defined by [first, last), to another
    /// range beginning at \a dest.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
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
    /// execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a copy algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy algorithm returns a
    ///           \a hpx::future<tagged_pair<tag::in(InIter), tag::out(OutIter)> >
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a tagged_pair<tag::in(InIter), tag::out(OutIter)>
    ///           otherwise.
    ///           The \a copy algorithm returns the pair of the input iterator
    ///           \a last and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename InIter, typename OutIter,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<InIter>::value &&
        hpx::traits::is_iterator<OutIter>::value)>
    typename util::detail::algorithm_result<
        ExPolicy, hpx::util::tagged_pair<tag::in(InIter), tag::out(OutIter)>
    >::type
    copy(ExPolicy && policy, InIter first, InIter last, OutIter dest)
    {
        return detail::transfer<detail::copy_iter<InIter, OutIter> >(
            std::forward<ExPolicy>(policy), first, last, dest);
    }

    /////////////////////////////////////////////////////////////////////////////
    // copy_n
    namespace detail
    {
        /// \cond NOINTERNAL

        // sequential copy_n
        template <typename IterPair>
        struct copy_n : public detail::algorithm<copy_n<IterPair>, IterPair>
        {
            copy_n()
              : copy_n::algorithm("copy_n")
            {}

            template <typename ExPolicy, typename InIter, typename OutIter>
            static std::pair<InIter, OutIter>
            sequential(ExPolicy, InIter first, std::size_t count, OutIter dest)
            {
                return util::copy_n_helper(first, count, dest);
            }

            template <typename ExPolicy, typename FwdIter, typename OutIter>
            static typename util::detail::algorithm_result<
                ExPolicy, std::pair<FwdIter, OutIter>
            >::type
            parallel(ExPolicy && policy, FwdIter first, std::size_t count,
                OutIter dest)
            {
                typedef hpx::util::zip_iterator<FwdIter, OutIter> zip_iterator;

                return get_iter_pair(
                    util::foreach_partitioner<ExPolicy>::call(
                        std::forward<ExPolicy>(policy),
                        hpx::util::make_zip_iterator(first, dest), count,
                        [](std::size_t, zip_iterator part_begin,
                            std::size_t part_size)
                        {
                            using hpx::util::get;

                            auto const& iters = part_begin.get_iterator_tuple();
                            util::copy_n_helper(get<0>(iters), part_size,
                                get<1>(iters));
                        },
                        [](zip_iterator && last) -> zip_iterator
                        {
                            using hpx::util::get;
                            auto const& iters = last.get_iterator_tuple();
                            util::copy_synchronize(get<0>(iters), get<1>(iters));
                            return std::move(last);
                        }));
            }
        };
        /// \endcond
    }

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
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
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
    /// an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a copy_n algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy_n algorithm returns a
    ///           \a hpx::future<tagged_pair<tag::in(InIter), tag::out(OutIter)> >
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a tagged_pair<tag::in(InIter), tag::out(OutIter)>
    ///           otherwise.
    ///           The \a copy algorithm returns the pair of the input iterator
    ///           forwarded to the first element after the last in the input
    ///           sequence and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename InIter, typename Size,
        typename OutIter,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<InIter>::value &&
        hpx::traits::is_iterator<OutIter>::value)>
    typename util::detail::algorithm_result<
        ExPolicy, hpx::util::tagged_pair<tag::in(InIter), tag::out(OutIter)>
    >::type
    copy_n(ExPolicy && policy, InIter first, Size count, OutIter dest)
    {
        static_assert(
            (hpx::traits::is_input_iterator<InIter>::value),
            "Required at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<OutIter>::value ||
                hpx::traits::is_forward_iterator<OutIter>::value),
            "Requires at least output iterator.");

        using hpx::util::tagged_pair;
        using hpx::util::make_tagged_pair;

        // if count is representing a negative value, we do nothing
        if (detail::is_negative(count))
        {
            return util::detail::algorithm_result<
                    ExPolicy, tagged_pair<tag::in(InIter), tag::out(OutIter)>
                >::get(make_tagged_pair<tag::in, tag::out>(first, dest));
        }

        typedef std::integral_constant<bool,
                is_sequential_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<InIter>::value ||
               !hpx::traits::is_forward_iterator<OutIter>::value
            > is_seq;

        return make_tagged_pair<tag::in, tag::out>(
            detail::copy_n<std::pair<InIter, OutIter> >().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, std::size_t(count), dest));
    }

    /////////////////////////////////////////////////////////////////////////////
    // copy_if
    namespace detail
    {
        /// \cond NOINTERNAL

        // sequential copy_if with projection function
        template <typename InIter, typename OutIter, typename Pred, typename Proj>
        inline std::pair<InIter, OutIter>
        sequential_copy_if(InIter first, InIter last, OutIter dest,
            Pred && pred, Proj && proj)
        {
            while (first != last)
            {
                if (hpx::util::invoke(pred, hpx::util::invoke(proj, *first)))
                    *dest++ = *first;
                first++;
            }
            return std::make_pair(first, dest);
        }

        template <typename IterPair>
        struct copy_if : public detail::algorithm<copy_if<IterPair>, IterPair>
        {
            copy_if()
              : copy_if::algorithm("copy_if")
            {}

            template <typename ExPolicy, typename InIter, typename OutIter,
                typename Pred, typename Proj = util::projection_identity>
            static std::pair<InIter, OutIter>
            sequential(ExPolicy, InIter first, InIter last, OutIter dest,
                Pred && pred, Proj && proj/* = Proj()*/)
            {
                return sequential_copy_if(first, last, dest,
                    std::forward<Pred>(pred), std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename FwdIter, typename OutIter,
                typename Pred, typename Proj = util::projection_identity>
            static typename util::detail::algorithm_result<
                ExPolicy, std::pair<FwdIter, OutIter>
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last,
                OutIter dest, Pred && pred, Proj && proj/* = Proj()*/)
            {
                typedef hpx::util::zip_iterator<FwdIter, bool*> zip_iterator;
                typedef util::detail::algorithm_result<
                    ExPolicy, std::pair<FwdIter, OutIter>
                > result;
                typedef typename std::iterator_traits<FwdIter>::difference_type
                    difference_type;

                if (first == last)
                    return result::get(std::make_pair(last, dest));

                difference_type count = std::distance(first, last);

                boost::shared_array<bool> flags(new bool[count]);
                std::size_t init = 0;

                using hpx::util::get;
                using hpx::util::make_zip_iterator;
                typedef util::scan_partitioner<
                        ExPolicy, std::pair<FwdIter, OutIter>, std::size_t
                    > scan_partitioner_type;
                return scan_partitioner_type::call(
                    std::forward<ExPolicy>(policy),
                    make_zip_iterator(first, flags.get()), count, init,
                    // step 1 performs first part of scan algorithm
                    [pred, proj](zip_iterator part_begin, std::size_t part_size)
                        -> std::size_t
                    {
                        std::size_t curr = 0;

                        // MSVC complains if proj is captured by ref below
                        util::loop_n(
                            part_begin, part_size,
                            [&pred, proj, &curr](zip_iterator it) mutable
                            {
                                using hpx::util::invoke;
                                bool f = invoke(pred, invoke(proj, get<0>(*it)));

                                if ((get<1>(*it) = f))
                                    ++curr;
                            });
                        return curr;
                    },
                    // step 2 propagates the partition results from left
                    // to right
                    hpx::util::unwrapped(std::plus<std::size_t>()),
                    // step 3 runs final accumulation on each partition
                    [dest, flags](
                        zip_iterator part_begin, std::size_t part_size,
                        hpx::shared_future<std::size_t> f_accu) mutable
                    {
                        std::advance(dest, f_accu.get());
                        util::loop_n(part_begin, part_size,
                            [&dest](zip_iterator it) mutable
                            {
                                if(get<1>(*it))
                                    *dest++ = get<0>(*it);
                            });
                    },
                    // step 4 use this return value
                    [last, dest, flags](
                        std::vector<hpx::shared_future<std::size_t> > && items,
                        std::vector<hpx::future<void> > &&) mutable
                    ->  std::pair<FwdIter, OutIter>
                    {
                        std::advance(dest, items.back().get());
                        return std::make_pair(last, dest);
                    });
            }
        };
        /// \endcond
    }

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
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a copy_if requires \a F to meet the
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
    ///                     unary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
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
    /// The assignments in the parallel \a copy_if algorithm invoked with
    /// an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a copy_if algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy_if algorithm returns a
    ///           \a hpx::future<tagged_pair<tag::in(InIter), tag::out(OutIter)> >
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a tagged_pair<tag::in(InIter), tag::out(OutIter)>
    ///           otherwise.
    ///           The \a copy algorithm returns the pair of the input iterator
    ///           forwarded to the first element after the last in the input
    ///           sequence and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename InIter, typename OutIter, typename F,
        typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<InIter>::value &&
        hpx::traits::is_iterator<OutIter>::value &&
        traits::is_projected<Proj, InIter>::value &&
        traits::is_indirect_callable<
            F, traits::projected<Proj, InIter>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy, hpx::util::tagged_pair<tag::in(InIter), tag::out(OutIter)>
    >::type
    copy_if(ExPolicy&& policy, InIter first, InIter last, OutIter dest, F && f,
        Proj && proj = Proj())
    {
        static_assert(
            (hpx::traits::is_input_iterator<InIter>::value),
            "Required at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<OutIter>::value ||
                hpx::traits::is_forward_iterator<OutIter>::value),
            "Requires at least output iterator.");

        typedef std::integral_constant<bool,
                is_sequential_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<InIter>::value ||
               !hpx::traits::is_forward_iterator<OutIter>::value
            > is_seq;

        return hpx::util::make_tagged_pair<tag::in, tag::out>(
            detail::copy_if<std::pair<InIter, OutIter> >().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, dest, std::forward<F>(f),
                std::forward<Proj>(proj)));
    }
}}}

#endif
