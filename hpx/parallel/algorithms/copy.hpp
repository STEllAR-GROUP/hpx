//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/copy.hpp

#if !defined(HPX_PARALLEL_DETAIL_COPY_MAY_30_2014_0317PM)
#define HPX_PARALLEL_DETAIL_COPY_MAY_30_2014_0317PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/algorithm_result.hpp>
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/scan_partitioner.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // copy
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename OutIter>
        struct copy
          : public detail::algorithm<copy<OutIter>, OutIter>
        {
            copy()
              : copy::algorithm("copy")
            {}

            template <typename ExPolicy, typename InIter, typename OutIter_>
            static OutIter_
            sequential(ExPolicy const&, InIter first, InIter last, OutIter_ dest)
            {
                return std::copy(first, last, dest);
            }

            template <typename ExPolicy, typename FwdIter, typename OutIter_>
            static typename detail::algorithm_result<ExPolicy, OutIter_>::type
            parallel(ExPolicy const& policy, FwdIter first, FwdIter last,
                OutIter_ dest)
            {
                typedef hpx::util::zip_iterator<FwdIter, OutIter_> zip_iterator;
                typedef typename zip_iterator::reference reference;
                typedef
                    typename detail::algorithm_result<ExPolicy, OutIter_>::type
                result_type;

                return get_iter<1, result_type>(
                    for_each_n<zip_iterator>().call(
                        policy, boost::mpl::false_(),
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
        typename detail::algorithm_result<ExPolicy, OutIter>::type
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

            return detail::copy<OutIter>().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, dest);
        }

        // forward declare the segmented version of this algorithm
        template <typename ExPolicy, typename InIter, typename OutIter>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
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
    /// \returns  The \a copy algorithm returns a \a hpx::future<OutIter> if the
    ///           execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a OutIter otherwise.
    ///           The \a copy algorithm returns the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename InIter, typename OutIter>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    copy(ExPolicy && policy, InIter first, InIter last, OutIter dest)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            input_iterator_category;
        typedef typename std::iterator_traits<OutIter>::iterator_category
            output_iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::input_iterator_tag, input_iterator_category>::value),
            "Required at least input iterator.");

        BOOST_STATIC_ASSERT_MSG(
            (boost::mpl::or_<
                boost::is_base_of<
                    std::forward_iterator_tag, output_iterator_category>,
                boost::is_same<
                    std::output_iterator_tag, output_iterator_category>
            >::value),
            "Requires at least output iterator.");

        typedef hpx::traits::segmented_iterator_traits<InIter> iterator_traits;
        typedef typename iterator_traits::is_segmented_iterator is_segmented;

        return detail::copy_(
            std::forward<ExPolicy>(policy), first, last, dest,
            is_segmented());
    }

    /////////////////////////////////////////////////////////////////////////////
    // copy_n
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename Iter>
        struct copy_n : public detail::algorithm<copy_n<Iter>, Iter>
        {
            copy_n()
              : copy_n::algorithm("copy_n")
            {}

            template <typename ExPolicy, typename InIter>
            static Iter
            sequential(ExPolicy const&, InIter first, std::size_t count,
                Iter dest)
            {
                return std::copy_n(first, count, dest);
            }

            template <typename ExPolicy, typename FwdIter>
            static typename detail::algorithm_result<ExPolicy, Iter>::type
            parallel(ExPolicy const& policy, FwdIter first, std::size_t count,
                Iter dest)
            {
                typedef hpx::util::zip_iterator<FwdIter, Iter> zip_iterator;
                typedef typename zip_iterator::reference reference;
                typedef
                    typename detail::algorithm_result<ExPolicy, Iter>::type
                result_type;

                return get_iter<1, result_type>(
                    for_each_n<zip_iterator>().call(
                        policy, boost::mpl::false_(),
                        hpx::util::make_zip_iterator(first, dest),
                        count,
                        [](reference t) {
                            hpx::util::get<1>(t) = hpx::util::get<0>(t); //-V573
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
    /// \returns  The \a copy_n algorithm returns a \a hpx::future<OutIter> if
    ///           the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a OutIter otherwise.
    ///           The \a copy_n algorithm returns the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename InIter, typename Size, typename OutIter>
    typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    copy_n(ExPolicy && policy, InIter first, Size count, OutIter dest)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            input_iterator_category;
        typedef typename std::iterator_traits<OutIter>::iterator_category
            output_iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::input_iterator_tag, input_iterator_category>::value),
            "Required at least input iterator.");

        BOOST_STATIC_ASSERT_MSG(
            (boost::mpl::or_<
                boost::is_base_of<
                    std::forward_iterator_tag, output_iterator_category>,
                boost::is_same<
                    std::output_iterator_tag, output_iterator_category>
            >::value),
            "Requires at least output iterator.");

        // if count is representing a negative value, we do nothing
        if (detail::is_negative<Size>::call(count))
        {
            return detail::algorithm_result<ExPolicy, OutIter>::get(
                std::move(dest));
        }

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, input_iterator_category>,
            boost::is_same<std::output_iterator_tag, output_iterator_category>
        >::type is_seq;

        return detail::copy_n<OutIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, std::size_t(count), dest);
    }

    /////////////////////////////////////////////////////////////////////////////
    // copy_if
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename Iter>
        struct copy_if : public detail::algorithm<copy_if<Iter>, Iter>
        {
            copy_if()
              : copy_if::algorithm("copy_if")
            {}

            template <typename ExPolicy, typename InIter, typename F>
            static Iter
            sequential(ExPolicy const&, InIter first, InIter last, Iter dest,
                F && f)
            {
                return std::copy_if(first, last, dest, std::forward<F>(f));
            }

            template <typename ExPolicy, typename FwdIter, typename F>
            static typename detail::algorithm_result<ExPolicy, Iter>::type
            parallel(ExPolicy const& policy, FwdIter first, FwdIter last,
                Iter dest, F && f)
            {
                typedef hpx::util::zip_iterator<FwdIter, char*> zip_iterator;
                typedef detail::algorithm_result<ExPolicy, Iter> result;
                typedef typename std::iterator_traits<FwdIter>::difference_type
                    difference_type;
                typedef typename detail::algorithm_result<ExPolicy, Iter>::type
                    result_type;

                if (first == last)
                    return result::get(std::move(dest));

                difference_type count = std::distance(first, last);

                boost::shared_array<char> flags(new char[count]);
                std::size_t init = 0;

                typedef util::scan_partitioner<ExPolicy, Iter, std::size_t>
                    scan_partitioner_type;
                return scan_partitioner_type::call(
                    policy,
                    hpx::util::make_zip_iterator(first, flags.get()),
                    count,
                    init,
                    // Flag the elements to be copied
                    [f](zip_iterator part_begin, std::size_t part_size)
                        -> std::size_t
                    {
                        std::size_t curr = 0;
                        util::loop_n(part_begin, part_size,
                            [&f, &curr](zip_iterator d) mutable
                            {
                                using hpx::util::get;
                                get<1>(*d) = (f(get<0>(*d)) != 0) ? 1 : 0;
                                curr += get<1>(*d);
                            });
                        return curr;
                    },
                    // Determine how far to advance the  dest iterator for each
                    // partition
                    hpx::util::unwrapped(
                        [](std::size_t const& prev, std::size_t const& curr)
                        {
                            return prev + curr;
                        }),
                    // Copy the elements into dest in parallel
                    [=](std::vector<hpx::shared_future<std::size_t> >&& r,
                        std::vector<std::size_t> const& chunk_sizes) mutable
                            -> result_type
                    {
                        HPX_ASSERT(!r.empty());
                        std::size_t last_index = r[r.size()-1].get();

                        typedef util::partitioner<ExPolicy, Iter, void>
                            partitioner_type;
                        return partitioner_type::call_with_data(
                            policy,
                            hpx::util::make_zip_iterator(first, flags.get()),
                            count,
                            [dest](hpx::shared_future<std::size_t>&& pos,
                                zip_iterator part_begin, std::size_t part_count)
                            {
                                Iter iter = dest;
                                std::size_t next_pos = pos.get();
                                std::advance(iter, next_pos);
                                util::loop_n(part_begin, part_count,
                                    [&iter](zip_iterator d)
                                    {
                                        using hpx::util::get;
                                        if(get<1>(*d))
                                            *iter++ = get<0>(*d);
                                    });
                            },
                            [=](std::vector<hpx::future<void> >&&) mutable
                                -> Iter
                            {
                                std::advance(dest, last_index);
                                return dest;
                            },
                            chunk_sizes,
                            std::move(r)
                        );
                    }
                );
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
    /// \returns  The \a copy_if algorithm returns a \a hpx::future<OutIter> if the
    ///           execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a OutIter otherwise.
    ///           The \a copy_if algorithm returns the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename InIter, typename OutIter, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    copy_if(ExPolicy&& policy, InIter first, InIter last, OutIter dest, F && f)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            input_iterator_category;
        typedef typename std::iterator_traits<OutIter>::iterator_category
            output_iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::input_iterator_tag, input_iterator_category>::value),
            "Required at least input iterator.");

        BOOST_STATIC_ASSERT_MSG(
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

        return detail::copy_if<OutIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, dest, std::forward<F>(f));
    }
}}}

#endif
