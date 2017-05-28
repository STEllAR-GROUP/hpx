//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/adjacent_find.hpp

#if !defined(HPX_PARALLEL_DETAIL_ADJACENT_FIND_SEP_20_2014_0731PM)
#define HPX_PARALLEL_DETAIL_ADJACENT_FIND_SEP_20_2014_0731PM

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1
{
    ///////////////////////////////////////////////////////////////////////////
    // adjacent_find
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename FwdIter>
        struct adjacent_find
          : public detail::algorithm<adjacent_find<FwdIter>, FwdIter>
        {
            adjacent_find()
              : adjacent_find::algorithm("adjacent_find")
            {}

            template <typename ExPolicy, typename Pred>
            static FwdIter
            sequential(ExPolicy, FwdIter first, FwdIter last, Pred && op)
            {
                return std::adjacent_find(first, last, op);
            }

            template <typename ExPolicy, typename Pred>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last,
                Pred && op)
            {
                typedef hpx::util::zip_iterator<FwdIter, FwdIter> zip_iterator;
                typedef typename zip_iterator::reference reference;
                typedef typename std::iterator_traits<FwdIter>::difference_type
                    difference_type;

                if(first == last)
                {
                    return util::detail::algorithm_result<
                            ExPolicy, FwdIter
                        >::get(std::move(last));
                }

                FwdIter next = first;
                ++next;
                difference_type count = std::distance(first, last);
                util::cancellation_token<difference_type> tok(count);

                return util::partitioner<ExPolicy, FwdIter, void>::
                    call_with_index(
                        std::forward<ExPolicy>(policy),
                        hpx::util::make_zip_iterator(first, next), count-1, 1,
                        [op, tok](zip_iterator it, std::size_t part_size,
                            std::size_t base_idx) mutable
                        {
                            util::loop_idx_n(
                                base_idx, it, part_size, tok,
                                [&op, &tok](reference t, std::size_t i)
                                {
                                    using hpx::util::get;
                                    if(op(get<0>(t), get<1>(t)))
                                        tok.cancel(i);
                                });
                        },
                        [=](std::vector<hpx::future<void> > &&) mutable -> FwdIter
                        {
                            difference_type adj_find_res = tok.get_data();
                            if(adj_find_res != count)
                                std::advance(first, adj_find_res);
                            else
                                first = last;

                            return std::move(first);
                        });
            }
        };
        /// \endcond
    }

    /// Searches the range [first, last) for two consecutive identical elements.
    /// This version uses the given binary predicate op
    ///
    /// \note   Complexity: Exactly the smaller of (result - first) + 1 and
    ///                     (last - first) - 1 application of the predicate
    ///                     where \a result is the value returned
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a adjacent_find requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the range the algorithm will be applied to.
    /// \param op           The binary predicate which returns \a true
    ///                     if the elements should be treated as equal. The
    ///                     signature should be equivalent to the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 must be such
    ///                     that objects of type \a FwdIter
    ///                     can be dereferenced and then implicitly converted
    ///                     to \a Type1 .
    ///
    /// The comparison operations in the parallel \a adjacent_find invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a adjacent_find invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a adjacent_find algorithm returns a \a hpx::future<InIter>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a InIter otherwise.
    ///           The \a adjacent_find algorithm returns an iterator to the
    ///           first of the identical elements. If no such elements are
    ///           found, \a last is returned.
    ///
    ///           This overload of \a adjacent_find is available if the user
    ///           decides to provide their algorithm their own binary
    ///           predicate \a op.
    ///
    template <typename ExPolicy, typename FwdIter,
        typename Pred = detail::equal_to>
    inline typename std::enable_if<
        execution::is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    >::type
    adjacent_find(ExPolicy && policy, FwdIter first, FwdIter last,
        Pred && op = Pred())
    {
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter>::value),
            "Requires at least a forward iterator");

        typedef execution::is_sequential_execution_policy<ExPolicy> is_seq;

        return detail::adjacent_find<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, std::forward<Pred>(op));
    }
}}}

#endif
