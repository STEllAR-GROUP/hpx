//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or uninitialized_copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/detail/uninitialized_copy.hpp

#if !defined(HPX_PARALLEL_DETAIL_UNINITIALIZED_COPY_OCT_02_2014_1145AM)
#define HPX_PARALLEL_DETAIL_UNINITIALIZED_COPY_OCT_02_2014_1145AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/detail/algorithm_result.hpp>
#include <hpx/parallel/detail/dispatch.hpp>
#include <hpx/parallel/detail/is_negative.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner_with_cleanup.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // uninitialized_copy
    namespace detail
    {
        /// \cond NOINTERNAL
        ///////////////////////////////////////////////////////////////////////
        template <typename Iter, typename FwdIter>
        FwdIter
        sequential_uninitialized_copy_n(Iter first, std::size_t count,
            FwdIter dest)
        {
            typedef hpx::util::zip_iterator<Iter, FwdIter> zip_iterator;
            typedef typename std::iterator_traits<FwdIter>::value_type
                value_type;

            return util::loop_with_cleanup_n(
                hpx::util::make_zip_iterator(first, dest), count,
                [](zip_iterator t) {
                    ::new (&hpx::util::get<1>(*t))
                        value_type(hpx::util::get<0>(*t));
                },
                [](zip_iterator t) {
                    hpx::util::get<1>(*t)->~value_type();
                });
        }

        template <typename Iter, typename FwdIter>
        FwdIter
        sequential_uninitialized_copy_n(Iter first, std::size_t count,
            FwdIter dest, util::cancellation_token<no_data>& tok)
        {
            typedef hpx::util::zip_iterator<Iter, FwdIter> zip_iterator;
            typedef typename std::iterator_traits<FwdIter>::value_type
                value_type;

            return util::loop_with_cleanup_n(
                hpx::util::make_zip_iterator(first, dest), count, tok,
                [](zip_iterator t) {
                    ::new (&hpx::util::get<1>(*t))
                        value_type(hpx::util::get<0>(*t));
                },
                [](zip_iterator t) {
                    hpx::util::get<1>(*t)->~value_type();
                });
        }

        template <typename FwdIter>
        struct uninitialized_copy
          : public detail::algorithm<uninitialized_copy<FwdIter>, FwdIter>
        {
            uninitialized_copy()
              : uninitialized_copy::algorithm("uninitialized_copy")
            {}

            template <typename ExPolicy, typename Iter, typename FwdIter>
            static FwdIter
            sequential(ExPolicy const&, Iter first, Iter last, FwdIter dest)
            {
                return sequential_uninitialized_copy_n(first,
                    std::distance(first, last), dest);
            }

            template <typename ExPolicy, typename Iter, typename FwdIter>
            static typename detail::algorithm_result<ExPolicy, Iter>::type
            parallel(ExPolicy const& policy, Iter first, Iter last, FwdIter dest)
            {
                if (count == 0)
                {
                    return detail::algorithm_result<ExPolicy, Iter>::get(
                        std::move(first));
                }

                typedef std::pair<Iter, Iter> partition_result_type;

                util::cancellation_token<no_data> tok;
                return util::partitioner_with_cleanup<
                        ExPolicy, Iter, FwdIter, partition_result_type
                    >::call(
                        first, count,
                        [dest, tok](Iter part_begin, std::size_t part_size)
                        {
                            return std::make_pair(part_begin,
                                sequential_uninitialized_copy_n(
                                    part_begin, part_size, dest, tok));
                        },
                        // finalize, called once if no error occurred
                        [=](std::vector<hpx::future<partition_result_type> > && v)
                            mutable -> FwdIter
                        {
                            return dest;
                        },
                        // cleanup function, called for each partition which
                        // didn't fail
                        [](partition_result_type const& r)
                        {
                        });
            }
        };
        /// \endcond
    }

    /// Copies the elements in the range, defined by [first, last), to an
    /// uninitialized memory area beginning at \a dest. If an exception is
    /// thrown during the initialization, the function has no effects.
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
    /// The assignments in the parallel \a uninitialized_copy algorithm invoked with an
    /// execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_copy algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_copy algorithm returns a \a hpx::future<OutIter> if the
    ///           execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a OutIter otherwise.
    ///           The \a uninitialized_copy algorithm returns the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename InIter, typename OutIter>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    uninitialized_copy(ExPolicy && policy, InIter first, InIter last, OutIter dest)
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

        return detail::uninitialized_copy<OutIter>().call(
            std::forward<ExPolicy>(policy),
            first, last, dest, is_seq());
    }

    /////////////////////////////////////////////////////////////////////////////
    // uninitialized_copy_n
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename OutIter>
        struct uninitialized_copy_n
          : public detail::algorithm<uninitialized_copy_n<OutIter>, OutIter>
        {
            uninitialized_copy_n()
              : uninitialized_copy_n::algorithm("uninitialized_copy_n")
            {}

            template <typename ExPolicy, typename InIter>
            static OutIter
            sequential(ExPolicy const&, InIter first, std::size_t count,
                OutIter dest)
            {
                return std::uninitialized_copy_n(first, count, dest);
            }

            template <typename ExPolicy, typename FwdIter>
            static typename detail::algorithm_result<ExPolicy, OutIter>::type
            parallel(ExPolicy const& policy, FwdIter first, std::size_t count,
                OutIter dest)
            {
                typedef hpx::util::zip_iterator<FwdIter, OutIter> zip_iterator;
                typedef typename zip_iterator::reference reference;
                typedef
                    typename detail::algorithm_result<ExPolicy, OutIter>::type
                result_type;

                return get_iter<1, result_type>(
                    for_each_n<zip_iterator>().call(policy,
                        hpx::util::make_zip_iterator(first, dest),
                        count,
                        [](reference t) {
                            hpx::util::get<1>(t) = hpx::util::get<0>(t); //-V573
                        },
                        boost::mpl::false_()));
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
    /// The assignments in the parallel \a uninitialized_copy_n algorithm invoked with
    /// an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_copy_n algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_copy_n algorithm returns a \a hpx::future<OutIter> if
    ///           the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a OutIter otherwise.
    ///           The \a uninitialized_copy_n algorithm returns the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename InIter, typename Size, typename OutIter>
    typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    uninitialized_copy_n(ExPolicy && policy, InIter first, Size count,
        OutIter dest)
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

        return detail::uninitialized_copy_n<OutIter>().call(
            std::forward<ExPolicy>(policy),
            first, std::size_t(count), dest, is_seq());
    }
}}}

#endif
