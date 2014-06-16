//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_STL_DETAILREUCE_JUN_01_2014_0903AM)
#define HPX_STL_DETAILREUCE_JUN_01_2014_0903AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/unwrapped.hpp>
#include <hpx/stl/execution_policy.hpp>
#include <hpx/stl/detail/algorithm_result.hpp>
#include <hpx/stl/util/partitioner.hpp>

#include <algorithm>
#include <numeric>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel
{
    ///////////////////////////////////////////////////////////////////////////
    // reduce
    namespace detail
    {
        template <typename ExPolicy, typename InIter, typename T, typename Pred>
        typename detail::algorithm_result<ExPolicy, T>::type 
        reduce(ExPolicy const&, InIter first, InIter last, T && init,
            Pred && op, boost::mpl::true_)
        {
            try {
                return detail::algorithm_result<ExPolicy, T>::get(
                    std::accumulate(first, last, std::forward<T>(init),
                        std::forward<Pred>(op)));
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

        template <typename ExPolicy, typename InIter, typename T, typename Pred>
        typename detail::algorithm_result<ExPolicy, T>::type
        reduce(ExPolicy const& policy, InIter first, InIter last, T && init,
            Pred && op, boost::mpl::false_)
        {
            if (first == last)
            {
                return detail::algorithm_result<ExPolicy, T>::get(
                    std::forward<T>(init));
            }

            typedef typename std::iterator_traits<InIter>::iterator_category
                category;

            return util::partitioner<ExPolicy, T>::call(
                first, std::distance(first, last),
                [op](InIter part_begin, std::size_t part_count)
                {
                    T val = *part_begin;
                    return util::accumulate_n<category>::call(
                        ++part_begin, --part_count, val, op);
                },
                hpx::util::unwrapped([init, op](std::vector<T>&& results)
                {
                    return util::accumulate_n<category>::call(
                        boost::begin(results), boost::size(results), init, op);
                }));
        }


        template <typename InIter, typename T, typename Pred>
        T reduce(execution_policy const& policy, InIter first, InIter last,
            T && init, Pred && op, boost::mpl::false_ fls)
        {
            switch (detail::which(policy))
            {
            case detail::execution_policy_enum::sequential:
                return detail::reduce(
                    *policy.get<sequential_execution_policy>(),
                    first, last, std::forward<T>(init), std::forward<Pred>(op),
                    boost::mpl::true_());

            case detail::execution_policy_enum::parallel:
                return detail::reduce(
                    *policy.get<parallel_execution_policy>(),
                    first, last, std::forward<T>(init), std::forward<Pred>(op),
                    fls);

            case detail::execution_policy_enum::vector:
                return detail::reduce(
                    *policy.get<vector_execution_policy>(),
                    first, last, std::forward<T>(init), std::forward<Pred>(op),
                    fls);

            case detail::execution_policy_enum::task:
                // the dynamic case will never return a future
                return detail::reduce(par,
                    first, last, std::forward<T>(init), std::forward<Pred>(op),
                    fls);

            default:
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "hpx::parallel::detail::reduce",
                    "Not supported execution policy");
                break;
            }
        }

        template<typename InIter, typename T, typename Pred>
        T reduce(execution_policy const& policy, InIter first, InIter last,
            T init, Pred && op, boost::mpl::true_ t)
        {
            return detail::reduce(sequential_execution_policy(),
                first, last, std::forward<T>(init), std::forward<Pred>(op), t);
        }
    }

    template <typename ExPolicy, typename InIter, typename T, typename Pred>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, T>::type
    >::type
    reduce(ExPolicy&& policy, InIter first, InIter last, T init, Pred && op)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag, iterator_category>::value,
            "Requires at least input iterator.");

        typedef boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::reduce(std::forward<ExPolicy>(policy), first, last,
            std::move(init), std::forward<Pred>(op), is_seq());
    }

    template <typename ExPolicy, typename InIter, typename T>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, T>::type
    >::type
    reduce(ExPolicy&& policy, InIter first, InIter last, T init)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag, iterator_category>::value,
            "Requires at least input iterator.");

        typedef boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::reduce(std::forward<ExPolicy>(policy), first, last,
            std::move(init), std::plus<T>(), is_seq());
    }

#if !defined(BOOST_NO_CXX11_FUNCTION_TEMPLATE_DEFAULT_ARGS)
    template <typename ExPolicy, typename InIter,
        typename T = std::iterator_traits<InIter>::value_type>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, T>::type
    >::type
    reduce(ExPolicy&& policy, InIter first, InIter last)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag, iterator_category>::value,
            "Requires at least input iterator.");

        typedef boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::reduce(std::forward<ExPolicy>(policy), first, last,
            T(), std::plus<T>(), is_seq());
    }
#endif
}}

#endif
