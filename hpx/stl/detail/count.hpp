//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DETAIL_COUNT_JUNE_17_2014_1154AM)
#define HPX_PARALLEL_DETAIL_COUNT_JUNE_17_2014_1154AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/stl/execution_policy.hpp>
#include <hpx/stl/detail/algorithm_result.hpp>
#include <hpx/stl/detail/zip_iterator.hpp>
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
    // count
    namespace detail
    {
        template <typename ExPolicy, typename InIter, typename T>
        typename detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<InIter>::difference_type>::type
        count(ExPolicy const&, InIter first, InIter last, const T& value,
        boost::mpl::true_)
        {
            typedef typename std::iterator_traits<InIter>::difference_type
                difference;
            try {
                return detail::algorithm_result<ExPolicy, difference>::get(
                    std::count(first, last, value));
            }
            catch (...) {
                detail::handle_exception<ExPolicy>::call();
            }
        }

        template <typename ExPolicy, typename FwdIter, typename T>
        typename detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<FwdIter>::difference_type
        >::type
        count(ExPolicy const& policy, FwdIter first, FwdIter last,
            T const& value, boost::mpl::false_ f)
        {
            typedef typename std::iterator_traits<FwdIter>::value_type type;
            typedef typename std::iterator_traits<FwdIter>::difference_type
                difference_type;

            if (first == last)
            {
                return detail::algorithm_result<ExPolicy, difference_type>
                    ::get( difference_type(0));
            }
            return util::partitioner<ExPolicy, difference_type>::call(
                policy, first, std::distance(first, last),
                [&value](FwdIter part_begin, std::size_t part_count)
                {
                    difference_type ret = std::count(part_begin,
                        std::next(part_begin, part_count), value);
                    return ret;
                },
                hpx::util::unwrapped([](std::vector<difference_type>&& results)
                {
                    return util::accumulate_n(boost::begin(results),
                        boost::size(results), difference_type(0),
                        [](difference_type v1, difference_type v2)
                        {
                            return v1 + v2;
                        });
                }));
        }

        template <typename InIter, typename T>
        typename std::iterator_traits<InIter>::difference_type
        count(execution_policy const& policy, InIter first, InIter last,
            T const& value, boost::mpl::false_)
        {
            HPX_PARALLEL_DISPATCH(policy, detail::count, first, last, value);
        }

        template <typename InIter, typename T>
        typename std::iterator_traits<InIter>::difference_type count(
            execution_policy const& policy, InIter first, InIter last,
            const T& value, boost::mpl::true_ t)
        {
            return detail::count(sequential_execution_policy(),
                first, last, value, t);
        }
    }

    template <typename ExPolicy, typename InIter, typename T>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<InIter>::difference_type
        >::type
    >::type
    count(ExPolicy && policy, InIter first, InIter last, T const& value)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            category;

        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag,
                typename std::iterator_traits<InIter>::iterator_category>::value,
            "Required at least input iterator .");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, category>
        >::type is_seq;

        return detail::count(std::forward<ExPolicy>(policy),
            first, last, value, is_seq());
    }

    ///////////////////////////////////////////////////////////////////////////
    // count_if
    namespace detail
    {
        template <typename ExPolicy, typename InIter, typename Pred>
        typename detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<InIter>::difference_type
        >::type
        count_if(ExPolicy const&, InIter first, InIter last, Pred && op,
            boost::mpl::true_)
        {
            typedef typename std::iterator_traits<InIter>::difference_type
                difference_type;

            try {
                return detail::algorithm_result<ExPolicy, difference_type>
                    ::get(std::count_if(first, last, std::forward<Pred>(op)));
            }
            catch(...) {
                detail::handle_exception<ExPolicy>::call();
            }
        }

        template <typename ExPolicy, typename FwdIter, typename Pred>
        typename detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<FwdIter>::difference_type
        >::type
        count_if(ExPolicy const& policy, FwdIter first, FwdIter last,
            Pred && op, boost::mpl::false_ f)
        {
            typedef typename std::iterator_traits<FwdIter>::value_type type;
            typedef typename std::iterator_traits<FwdIter>::difference_type
                difference_type;

            if (first == last)
            {
                return detail::algorithm_result<ExPolicy, difference_type>
                    ::get( (difference_type)0);
            }
            return util::partitioner<ExPolicy, difference_type>::call(
                policy, first, std::distance(first, last),
                [op](FwdIter part_begin, std::size_t part_count)
                {
                    difference_type ret = std::count_if(part_begin,
                        std::next(part_begin, part_count), op);
                    return ret;
                },
                hpx::util::unwrapped([](std::vector<difference_type>&& results)
                {
                    return util::accumulate_n(boost::begin(results),
                        boost::size(results), difference_type(0),
                        [](difference_type v1, difference_type v2)
                        {
                            return v1 + v2;
                        });
                }));
        }

        template <typename InIter, typename Pred>
        typename std::iterator_traits<InIter>::difference_type
        count_if(execution_policy const& policy, InIter first, InIter last,
            Pred && op, boost::mpl::false_)
        {
            HPX_PARALLEL_DISPATCH(policy, detail::count_if, first, last,
                std::forward<Pred>(op));
        }

        template <typename InIter, typename Pred>
        typename std::iterator_traits<InIter>::difference_type
        count_if(execution_policy const& policy, InIter first, InIter last,
            Pred && op, boost::mpl::true_ t)
        {
            return detail::count_if(sequential_execution_policy(),
                first, last, std::forward<Pred>(op), t);
        }
    }

    template <typename ExPolicy, typename InIter, typename Pred>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<InIter>::difference_type
        >::type
    >::type
    count_if(ExPolicy && policy, InIter first, InIter last, Pred && op)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            category;

        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag, category>::value,
            "Required at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, category>
        >::type is_seq;

        return detail::count_if(std::forward<ExPolicy>(policy),
            first, last,
            std::forward<Pred>(op), is_seq());
    }
}}

#endif
