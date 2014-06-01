
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
#include <hpx/stl/detail/zip_iterator.hpp>

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

		        template <int N, typename R, typename ZipIter>
        R get_iter(ZipIter&& zipiter)
        {
            return boost::get<N>(*zipiter);
        }

        template <int N, typename R, typename ZipIter>
        R get_iter(hpx::future<ZipIter>&& zipiter)
        {
            return zipiter.then(
                [](hpx::future<ZipIter>&& f) {
                    typename std::iterator_traits<ZipIter>::value_type t =
                        *f.get();
                    return boost::get<N>(t);
                });
        }
	
		template<typename ExPolicy, typename InIter, typename OutIter,
			typename IterTag>
		typename detail::algorithm_result<ExPolicy, OutIter>::type
		copy(ExPolicy const& policy, InIter first, InIter last, OutIter dest, 
			IterTag category)
		{
			typedef boost::tuple<InIter, OutIter> iterator_tuple;
			typedef detail::zip_iterator<iterator_tuple> zip_iterator;
			typedef typename zip_iterator::reference reference;
			typedef
				typename detail::algorithm_result<ExPolicy, void>::type
			result_type;

			return get_iter<1, result_type>(
				for_each_n(policy,
					detail::make_zip_iterator(boost::make_tuple(first,dest)),
					std::distance(first,last),
					[](reference it) {
						*boost::get<1>(it) = *boost::get<0>(it);
					},
					category));
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
		OutIter copy(sequential_execution_policy const& policy,
			InIter first, InIter last, OutIter dest, IterTag category)
		{
			return detail::copy_seq(policy, first, last, dest, category);
		}

		template<typename InIter, typename OutIter, typename IterTag>
		OutIter copy(execution_policy const& policy,
			InIter first, InIter last, OutIter dest, IterTag category)
		{
			switch (detail::which(policy))
			{
			case detail::execution_policy_enum::sequential:
				return detail::copy(
					*policy.get<sequential_execution_policy>(),
					first, last, dest, category);

			case detail::execution_policy_enum::parallel:
				return detail::copy(
					*policy.get<parallel_execution_policy>(),
					first, last, dest, category);

			case detail::execution::policy_enum::task:
				return detail::copy(par,
					first, last, desk, category);

			default:
				HPX_THROW_EXCEPTION(hpx::bad_parameter,
					"hpx::parallel::detail::transform",
					"Not supported execution policy");
				break;
			}
		}
	}

	template<typename ExPolicy, typename InIter, typename OutIter>
	typename boost::enable_if<
		is_execution_policy<ExPolicy>,
		typename detail::algorith_result<ExPolicy, OutIter>::type
	>::type
	transform(ExPolicy&& policy, InIter first, InIter last, OutIter dest)
	{
		BOOST_STATIC_ASSERT_MSG(
			boost::is_base_of<std::input_iterator_tag,
				typename std::iterator_traits<InIter>::iterator_category::value,
			"Required at least input iterator.");

		std::iterator_traits<InIter>::iterator_category category;
		return detail::copy(policy, first, last, dest, category);
	}
	
}