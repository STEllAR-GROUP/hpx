//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_STL_DETAIL_MOVE_JUNE_16_2014_1106AM)
#define HPX_STL_DETAIL_MOVE_JUNE_16_2014_1106AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/stl/execution_policy.hpp>
#include <hpx/stl/detail/algorithm_result.hpp>
#include <hpx/stl/detail/zip_iterator.hpp>
#include <hpx/stl/detail/synchronize.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel
{
    ///////////////////////////////////////////////////////////////////////////
    // move
    namespace detail
    {
        template <typename ExPolicy, typename InIter, typename OutIter>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
        move(ExPolicy const&, InIter first, InIter last, OutIter dest,
            boost::mpl::true_)
        {
            try {
                return detail::algorithm_result<ExPolicy, OutIter>::get(
                    std::move(first, last, dest));
            }
            catch(std::bad_alloc const& e) {
               boost::throw_exception(e);
            }
            catch (...) {
                boost::throw_exception(
                    hpx::exception_list(boost::current_exception())
                );
            }
        }

        template <typename ExPolicy, typename FwdIter, typename OutIter>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
        move(ExPolicy const& policy, FwdIter first, FwdIter last, OutIter dest,
            boost::mpl::false_ fls)
        {
            typedef boost::tuple<FwdIter, OutIter> iterator_tuple;
            typedef detail::zip_iterator<iterator_tuple> zip_iterator;
            typedef typename zip_iterator::reference reference;
            typedef
                typename detail::algorithm_result<ExPolicy, OutIter>::type
            result_type;

            return get_iter<1, result_type>(
                plain_for_each_n(policy,
                    detail::make_zip_iterator(boost::make_tuple(first,dest)),
                    std::distance(first,last),
                    [](reference it) {
                        *boost::get<1>(it) = std::move(*boost::get<0>(it));
                    },
                    fls));
        }

        template <typename InIter, typename OutIter>
        OutIter move(execution_policy const& policy,
            InIter first, InIter last, OutIter dest, boost::mpl::false_ fls)
        {
            switch (detail::which(policy))
            {
            case detail::execution_policy_enum::sequential:
                return detail::move(
                    *policy.get<sequential_execution_policy>(),
                    first, last, dest, boost::mpl::true_());

            case detail::execution_policy_enum::parallel:
                return detail::move(
                    *policy.get<parallel_execution_policy>(),
                    first, last, dest, fls);

            case detail::execution_policy_enum::vector:
                return detail::move(
                    *policy.get<vector_execution_policy>(),
                    first, last, dest, fls);

            case detail::execution_policy_enum::task:
                return detail::move(par,
                    first, last, dest, fls);

            default:
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "hpx::parallel::detail::move",
                    "Not supported execution policy");
                break;
            }
        }

        template <typename InIter, typename OutIter>
        OutIter move(execution_policy const& policy,
            InIter first, InIter last, OutIter dest, boost::mpl::true_ t)
        {
            return detail::move(sequential_execution_policy(),
                first, last, dest, t);
        }
    }

    template <typename ExPolicy, typename InIter, typename OutIter>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    move(ExPolicy && policy, InIter first, InIter last, OutIter dest)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            input_iterator_category;
        typedef typename std::iterator_traits<OutIter>::iterator_category
            output_iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<
                std::input_iterator_tag, input_iterator_category>::value,
            "Required at least input iterator.");

        BOOST_STATIC_ASSERT_MSG(
            boost::mpl::or_<
                boost::is_base_of<
                    std::forward_iterator_tag, output_iterator_category>,
                boost::is_same<
                    std::output_iterator_tag, output_iterator_category>
            >::value,
            "Requires at least output iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, input_iterator_category>,
            boost::is_same<std::output_iterator_tag, output_iterator_category>
        >::type is_seq;

        return detail::move( std::forward<ExPolicy>(policy),
            first, last, dest, is_seq());
    }
}}

#endif
