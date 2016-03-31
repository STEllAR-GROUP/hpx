//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2015 Daniel Bourgeois
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

#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/algorithms/detail/is_negative.hpp>
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/scan_partitioner.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>
#include <hpx/parallel/traits/projected.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/shared_array.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // copy
    namespace detail
    {
        /// \cond NOINTERNAL

        // sequential copy
        template <typename InIter, typename OutIter>
        inline std::pair<InIter, OutIter>
        sequential_copy(InIter first, InIter last, OutIter dest)
        {
            while (first != last)
            {
                *dest++ = *first++;
            }
            return std::make_pair(first, dest);
        }

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
                return sequential_copy(first, last, dest);
            }

            template <typename ExPolicy, typename FwdIter, typename OutIter>
            static typename util::detail::algorithm_result<
                ExPolicy, std::pair<FwdIter, OutIter>
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last,
                OutIter dest)
            {
                typedef hpx::util::zip_iterator<FwdIter, OutIter> zip_iterator;
                typedef typename zip_iterator::reference reference;

                return get_iter_pair(
                    for_each_n<zip_iterator>().call(
                        std::forward<ExPolicy>(policy), boost::mpl::false_(),
                        hpx::util::make_zip_iterator(first, dest),
                        std::distance(first, last),
                        [](reference t)
                        {
                            using hpx::util::get;
                            get<1>(t) = get<0>(t); //-V573
                        }));
            }
        };

        template <typename ExPolicy, typename InIter, typename OutIter>
        typename util::detail::algorithm_result<
            ExPolicy, std::pair<InIter, OutIter>
        >::type
        copy_(ExPolicy && policy, InIter first, InIter last, OutIter dest,
            std::false_type)
        {
            typedef typename std::iterator_traits<InIter>::iterator_category
                input_iterator_category;
            typedef typename std::iterator_traits<OutIter>::iterator_category
                output_iterator_category;

            typedef typename boost::mpl::or_<
                parallel::is_sequential_execution_policy<ExPolicy>,
                boost::is_same<std::input_iterator_tag, input_iterator_category>,
                boost::is_same<std::output_iterator_tag, output_iterator_category>
            >::type is_seq;

            return detail::copy<std::pair<InIter, OutIter> >().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, dest);
        }

        // forward declare the segmented version of this algorithm
        template <typename ExPolicy, typename InIter, typename OutIter>
        typename util::detail::algorithm_result<
            ExPolicy, std::pair<InIter, OutIter>
        >::type
        copy_(ExPolicy && policy, InIter first, InIter last, OutIter dest,
            std::true_type);

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
        typedef typename std::iterator_traits<InIter>::iterator_category
            input_iterator_category;
        typedef typename std::iterator_traits<OutIter>::iterator_category
            output_iterator_category;

        static_assert(
            (boost::is_base_of<
                std::input_iterator_tag, input_iterator_category>::value),
            "Required at least input iterator.");

        static_assert(
            (boost::mpl::or_<
                boost::is_base_of<
                    std::forward_iterator_tag, output_iterator_category>,
                boost::is_same<
                    std::output_iterator_tag, output_iterator_category>
            >::value),
            "Requires at least output iterator.");

        typedef hpx::traits::segmented_iterator_traits<InIter> iterator_traits;
        typedef typename iterator_traits::is_segmented_iterator is_segmented;

        return hpx::util::make_tagged_pair<tag::in, tag::out>(
            detail::copy_(
                std::forward<ExPolicy>(policy), first, last, dest,
                is_segmented()));
    }

    /////////////////////////////////////////////////////////////////////////////
    // copy_n
    namespace detail
    {
        /// \cond NOINTERNAL

        // sequential copy_n
        template <typename InIter, typename OutIter>
        inline std::pair<InIter, OutIter>
        sequential_copy_n(InIter first, std::size_t count, OutIter dest)
        {
            if (count > 0)
            {
                *dest++ = *first;
                for (std::size_t i = 1; i != count; ++i)
                {
                    *dest++ = *++first;
                }
            }
            return std::make_pair(first, dest);
        }

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
                return sequential_copy_n(first, count, dest);
            }

            template <typename ExPolicy, typename FwdIter, typename OutIter>
            static typename util::detail::algorithm_result<
                ExPolicy, std::pair<FwdIter, OutIter>
            >::type
            parallel(ExPolicy && policy, FwdIter first, std::size_t count,
                OutIter dest)
            {
                typedef hpx::util::zip_iterator<FwdIter, OutIter> zip_iterator;
                typedef typename zip_iterator::reference reference;

                return get_iter_pair(
                    for_each_n<zip_iterator>().call(
                        std::forward<ExPolicy>(policy), boost::mpl::false_(),
                        hpx::util::make_zip_iterator(first, dest),
                        count,
                        [](reference t)
                        {
                            using hpx::util::get;
                            get<1>(t) = get<0>(t); //-V573
                        }
                    ));
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
        typedef typename std::iterator_traits<InIter>::iterator_category
            input_iterator_category;
        typedef typename std::iterator_traits<OutIter>::iterator_category
            output_iterator_category;

        static_assert(
            (boost::is_base_of<
                std::input_iterator_tag, input_iterator_category>::value),
            "Required at least input iterator.");

        static_assert(
            (boost::mpl::or_<
                boost::is_base_of<
                    std::forward_iterator_tag, output_iterator_category>,
                boost::is_same<
                    std::output_iterator_tag, output_iterator_category>
            >::value),
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

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, input_iterator_category>,
            boost::is_same<std::output_iterator_tag, output_iterator_category>
        >::type is_seq;

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
                Pred && pred, Proj && proj = Proj())
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
                OutIter dest, Pred && pred, Proj && proj = Proj())
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
        typedef typename std::iterator_traits<InIter>::iterator_category
            input_iterator_category;
        typedef typename std::iterator_traits<OutIter>::iterator_category
            output_iterator_category;

        static_assert(
            (boost::is_base_of<
                std::input_iterator_tag, input_iterator_category>::value),
            "Required at least input iterator.");

        static_assert(
            (boost::mpl::or_<
                boost::is_base_of<
                    std::forward_iterator_tag, output_iterator_category>,
                boost::is_same<
                    std::output_iterator_tag, output_iterator_category>
            >::value),
            "Requires at least output iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, input_iterator_category>,
            boost::is_same<std::output_iterator_tag, output_iterator_category>
        >::type is_seq;

        return hpx::util::make_tagged_pair<tag::in, tag::out>(
            detail::copy_if<std::pair<InIter, OutIter> >().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, dest, std::forward<F>(f),
                std::forward<Proj>(proj)));
    }
}}}

#endif
