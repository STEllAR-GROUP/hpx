//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#if !defined(HPX_STL_DETAIL_SWAP_RANGES_JUNE_20_2014_1006AM)
#define HPX_STL_DETAIL_SWAP_RANGES_JUNE_20_2014_1006AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/stl/execution_policy.hpp>
#include <hpx/stl/detail/algorithm_result.hpp>
#include <hpx/stl/detail/zip_iterator.hpp>
#include <hpx/stl/detail/synchronize.hpp>
#include <hpx/stl/util/partitioner.hpp>
#include <hpx/stl/util/loop.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel
{
    ///////////////////////////////////////////////////////////////////////////
    // swap ranges
    namespace detail
    {
        template <typename ExPolicy, typename ForwardIter1, typename ForwardIter2>
        typename detail::algorithm_result<ExPolicy, ForwardIter2>::type
        swap_ranges(ExPolicy const&, ForwardIter1 first1, ForwardIter1 last1,
            ForwardIter2 first2, boost::mpl::true_)
        {
            try {
                return detail::algorithm_result<ExPolicy, ForwardIter2>::get(
                    std::swap_ranges(first1, last1, first2));
            }
            catch(...) {
                detail::handle_exception<ExPolicy>::call();
            }
        }

        template <typename ExPolicy, typename ForwardIter1, typename ForwardIter2>
        typename detail::algorithm_result<ExPolicy, ForwardIter2>::type
        swap_ranges(ExPolicy const& policy, ForwardIter1 first1, ForwardIter1 last1,
            ForwardIter2 first2, boost::mpl::false_ f)
        {
            typedef boost::tuple<ForwardIter1, ForwardIter2> iterator_tuple;
            typedef detail::zip_iterator<iterator_tuple> zip_iterator;
            typedef typename zip_iterator::reference reference;
            typedef
                typename detail::algorithm_result<ExPolicy, ForwardIter2>::type
            result_type;

            return get_iter<1, result_type>(
                plain_for_each_n(policy,
                    detail::make_zip_iterator(boost::make_tuple(first1, first2)),
                    std::distance(first1, last1),
                    [](reference it) {
                        std::swap(*boost::get<0>(it), *boost::get<1>(it));
                    },
                    f));
        }

        template <typename ForwardIter1, typename ForwardIter2>
        ForwardIter2 swap_ranges(execution_policy const& policy,ForwardIter1 first1,
            ForwardIter1 last1, ForwardIter2 first2, boost::mpl::false_ f)
        {
            switch (detail::which(policy))
            {
            case detail::execution_policy_enum::sequential:
                return detail::swap_ranges(
                    *policy.get<sequential_execution_policy>(),
                    first1, last1, first2, boost::mpl::true_());

            case detail::execution_policy_enum::parallel:
                return detail::swap_ranges(
                    *policy.get<parallel_execution_policy>(),
                    first1, last1, first2, f);

            case detail::execution_policy_enum::vector:
                return detail::swap_ranges(
                    *policy.get<vector_execution_policy>(),
                    first1, last1, first2, f);

            case detail::execution_policy_enum::task:
                return detail::swap_ranges(par,
                    first1, last1, first2, f);

            default:
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "hpx::parallel::detail::swap_ranges",
                    "Not supported execution policy");
                break;
            }
        }

        template <typename ForwardIter1, typename ForwardIter2>
        ForwardIter2 swap_ranges(execution_policy const& policy, ForwardIter1 first1,
             ForwardIter1 last1, ForwardIter2 first2, boost::mpl::true_ t)
        {
            return detail::swap_ranges(sequential_execution_policy(),
                first1, last1, first2, t);
        }
    }

    template <typename ExPolicy, typename ForwardIter1, typename ForwardIter2>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, ForwardIter2>::type
    >::type
    swap_ranges(ExPolicy && policy, ForwardIter1 first1, ForwardIter1 last1,
        ForwardIter2 first2)
    {
        typedef typename std::iterator_traits<ForwardIter1>::iterator_category
            iter1_category;
        typedef typename std::iterator_traits<ForwardIter2>::iterator_category
            iter2_category;

        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<
                std::forward_iterator_tag, iter1_category>::value,
            "Required at least forward iterator tag.");
        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<
                std::forward_iterator_tag, iter2_category>::value,
            "Required at least forward iterator tag.");

        typedef typename is_sequential_execution_policy<ExPolicy>::type is_seq;
        return detail::swap_ranges( std::forward<ExPolicy>(policy),
            first1, last1, first2, is_seq());
    }
}}
#endif
