
#if !define(HPX_STL_DETAIL_COPY_MAY_30_2014_0317PM)
#define HPX_STL_DETAIL_COPY_MAY_30_2014_0317PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/stl/execution_policy.hpp>
#include <hpx/stl/detail/algorithm_result.hpp>
#include <hpx/stl/util/partitioner.hpp>
#include <hpx/stl/util/loop.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/util/void_guard.hpp>
#include <hpx/util/decay.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel
{
	///////////////////////////////////////////////////////////////////////////
    // copy
	namespace detail
	{
		template<typename ExPolicy, typename InIter, typename OutIter
			typename IterTag>
		typename detail::algorithm_result<ExPolicy, OutIter>::type
		copy_seq(ExPolicy const&, InIter first, InIter last, OutIter dest,
			IterTag)
		{
			try {
				return detail::algorithm_result<ExPolicy, OutIter>::get(
					std::copy(first,last,dest));
			}
			catch(std::bad_alloc const& e) {
				throw e;
			}
			catch (...) {
				throw hpx::exception_list(boost::current_exception());
			}
		}
	
		template<typename ExPolicy, typename InIter, typename OutIter,
			typename IterTag>
		typename detail::algorithm_result<ExPolicy, OutIter>::type
		copy(ExPolicy const& policy, InIter first, InIter last, OutIter dest, 
			IterTag category)
		{
			typedef
				typename detail::algorithm_result<ExPolicy, void>::type
			result_type;

			return hpx::util::void_guard<result_type>(),
				detail::for_each_n(policy, first, std::distance(first,last),
				[](){}, category);
		}

		template<typename ExPolicy, typename InIter, typename OutIter>
		typename boost::enable_if<
			is_parallel_execution_policy<ExPolicy>,
			typename detail::algorithm_resutl<ExPolicy, void>::type
		>::type
		for_each(ExPolicy const& policy, InIter first, InIter last, OutIter dest,
			std::input_iterator_tag category)
		{
			return detail::copy_seq(policy, first, last, desk, category);
		}

		template<typename InIter, typename OutIter, typename IterTag>
		OutIter transform
	
	
	
	
	}