//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/detail/search.hpp

#if !defined(HPX_PARALLEL_DETAIL_COPY_MAY_30_2014_0317PM)
#define HPX_PARALLEL_DETAIL_COPY_MAY_30_2014_0317PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/detail/algorithm_result.hpp>
#include <hpx/parallel/detail/dispatch.hpp>
#include <hpx/parallel/detail/predicates.hpp>
#include <hpx/parallel/detail/for_each.hpp>
#include <hpx/parallel/detail/is_negative.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx {namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
	///////////////////////////////////////////////////////////////////////////
	// search 
	namespace detail
	{
		/// \cond NOINTERNAL
		template<typename FwdIter>
		struct search: public detail::algorithm<search<FwdIter>, FwdIter>
		{
			search()
				: search::algorithm("search")
			{}

			template <typename ExPolicy, typename FwdIter2, typename Pred>
			static FwdIter
			sequential(ExPolicy const&, FwdIter first, FwdIter last, FwdIter2 s_first,
				FwdIter2 s_last, Pred && op)
			{
				return std::search(first, last, s_first, s_last, op);
			}

			template <typename ExPolicy, typename FwdIter2, typename Pred>
			static typename detail::algorithm_result<ExPolicy, FwdIter>::type
			parallel(ExPolicy const& policy, FwdIter first, FwdIter last,
				FwdIter2 s_first, FwdIter2 s_last, Pred && op)
			{
				typedef typename std::iterator_traits<FwdIter>::reference reference;
				typedef typename std::iterator_traits<FwdIter>::difference_type
					difference_type;
				typedef typename std::iterator_traits<FwdIter2>::difference_type
					s_difference_type;

				s_difference_type diff = std::distance(s_first, s_last);
				if(diff <= 0) 
				{
					return detail::algorithm_result<ExPolicy, FwdIter>::get(
						std::move(last));
				}

				difference_type count = std::distance(first, last);
				if(diff > count)
				{
					return detail::algorithm_result<ExPolicy, FwdIter>::get(
						std::move(last));
				}

				util::cancellation_token<difference_type> tok(count);

				return util::partitioner<ExPolicy, FwdIter, void>::call_with_index(
					policy, first, count-(diff-1),
					[=](std::size_t base_idx, FwdIter it, std::size_t part_size) mutable
				{

					FwdIter curr = it;

					util::loop_idx_n(
						base_idx, it, part_size, tok,
						[=, &tok, &curr](reference v, std::size_t i)
						{
							++curr;
							if (op(v, *s_first))
							{
								difference_type local_count = 1;
								FwdIter2 needle = s_first;
								FwdIter mid = curr;

								for(difference_type len = 0;
									local_count != diff && len != count;
									++local_count, ++len, ++mid)
								{
									if(*mid != *++needle)
										break;
								}

								if(local_count == diff)
									tok.cancel(i);
							}
						});
				},
				[=](std::vector<hpx::future<void> > &&) mutable
				{
					difference_type search_res = tok.get_data();
					if( search_res != count)
						std::advance(first, search_res);
					else
						first = last;

					return std::move(first);
				});
			}
		};
	}

		    /// Searches the range [first, last) for any elements in the range [s_first, s_last).
    /// Uses opeartor== to compare elements.
    ///
    /// \note   Complexity: at most (S*N) comparisons where
    ///         \a S = distance(s_first, s_last) and
    ///         \a N = distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter      The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam FwdIter2     The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param s_first      Refers to the beginning of the sequence of elements
    ///                     the algorithm will be searching for.
    /// \param s_last       Refers to the end of the sequence of elements of
    ///                     the algorithm will be searching for.
    ///
    /// The comparison operations in the parallel \a find_first_of algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find_first_of algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find_first_of algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type \a task_execution_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a find_first_of algorithm returns an iterator to the beginning of
    ///           the last subsequence [s_first, s_last) in range [first, last).
    ///           If the length of the subsequence [s_first, s_last) is greater
    ///           than the length of the range [first, last), \a last is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, \a last is also returned.
    ///           
	template <typename ExPolicy, typename FwdIter, typename FwdIter2>
	inline typename boost::enable_if<
		is_execution_policy<ExPolicy>,
		typename detail::algorithm_result<ExPolicy, FwdIter>::type
	>::type
	search(ExPolicy && policy, FwdIter first, FwdIter last,
		FwdIter2 s_first, FwdIter2 s_last)
	{
		typedef typename std::iterator_traits<FwdIter>::iterator_category
			iterator_category;
		typedef typename std::iterator_traits<FwdIter2>::iterator_category
			s_iterator_category;

		BOOST_STATIC_ASSERT_MSG(
			(boost::is_base_of<
				std::input_iterator_tag, iterator_category
			>::value),
			"Requires at least input iterator.");

		BOOST_STATIC_ASSERT_MSG(
			(boost::is_base_of<
				std::forward_iterator_tag, s_iterator_category
			>::value),
			"Subsequence requires at least forward iterator.");
		
		typedef typename boost::mpl::or_<
			is_sequential_execution_policy<ExPolicy>,
			boost::is_same<std::input_iterator_tag, iterator_category>
		>::type is_seq;

		return detail::search<FwdIter>().call(
			std::forward<ExPolicy>(policy),
			first, last, s_first, s_last,
			detail::equal_to(), is_seq());
	}
}}}

#endif