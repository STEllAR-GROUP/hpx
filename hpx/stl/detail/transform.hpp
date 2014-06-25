//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_STL_DETAIL_TRANSFORM_MAY_29_2014_0932PM)
#define HPX_STL_DETAIL_TRANSFORM_MAY_29_2014_0932PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/stl/execution_policy.hpp>
#include <hpx/stl/detail/algorithm_result.hpp>
#include <hpx/stl/detail/zip_iterator.hpp>
#include <hpx/stl/detail/synchronize.hpp>
#include <hpx/util/move.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel
{
    ///////////////////////////////////////////////////////////////////////////
    // transform
    namespace detail
    {
        template <typename ExPolicy, typename InIter, typename OutIter,
            typename F>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
        transform(ExPolicy const&, InIter first, InIter last, OutIter dest,
            F && f, boost::mpl::true_)
        {
            try {
                detail::synchronize(first, last);
                return detail::algorithm_result<ExPolicy, OutIter>::get(
                    std::transform(first, last, dest, std::forward<F>(f)));
            }
            catch (...) {
                detail::handle_exception<ExPolicy>::call();
            }
        }

        template <typename ExPolicy, typename FwdIter, typename OutIter,
            typename F>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
        transform(ExPolicy const& policy, FwdIter first, FwdIter last,
            OutIter dest, F && f, boost::mpl::false_ fls)
        {
            typedef boost::tuple<FwdIter, OutIter> iterator_tuple;
            typedef detail::zip_iterator<iterator_tuple> zip_iterator;
            typedef typename zip_iterator::reference reference;
            typedef
                typename detail::algorithm_result<ExPolicy, OutIter>::type
            result_type;

            return get_iter<1, result_type>(
                for_each_n(policy,
                    detail::make_zip_iterator(boost::make_tuple(first, dest)),
                    std::distance(first, last),
                    [f](reference it) {
                        *boost::get<1>(it) = f(*boost::get<0>(it));
                    },
                    fls));
        }

        template <typename InIter, typename OutIter, typename F>
        OutIter transform(execution_policy const& policy,
            InIter first, InIter last, OutIter dest, F && f, boost::mpl::false_ fls)
        {
            switch (detail::which(policy))
            {
            case detail::execution_policy_enum::sequential:
                return detail::transform(
                    *policy.get<sequential_execution_policy>(),
                    first, last, dest, std::forward<F>(f), boost::mpl::true_());

            case detail::execution_policy_enum::parallel:
                return detail::transform(
                    *policy.get<parallel_execution_policy>(),
                    first, last, dest, std::forward<F>(f), fls);

            case detail::execution_policy_enum::vector:
                return detail::transform(
                    *policy.get<vector_execution_policy>(),
                    first, last, dest, std::forward<F>(f), fls);

            case detail::execution_policy_enum::task:
                // the dynamic case will never return a future
                return detail::transform(par,
                    first, last, dest, std::forward<F>(f), fls);

            default:
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "hpx::parallel::detail::transform",
                    "Not supported execution policy");
                break;
            }
        }

        template <typename InIter, typename OutIter, typename F>
        OutIter transform(execution_policy const& policy,
            InIter first, InIter last, OutIter dest, F && f, boost::mpl::true_ t)
        {
            return detail::transform(sequential_execution_policy(),
                first, last, dest, std::forward<F>(f), t);
        }
    }

    template <typename ExPolicy, typename InIter, typename OutIter, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    transform(ExPolicy&& policy, InIter first, InIter last, OutIter dest, F && f)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag,
                typename std::iterator_traits<InIter>::iterator_category>::value,
            "Required at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::transform(std::forward<ExPolicy>(policy),
            first, last, dest,
            std::forward<F>(f), is_seq());
    }

    ///////////////////////////////////////////////////////////////////////////
    // transform binary predicate
    namespace detail
    {
        template <typename ExPolicy, typename InIter1, typename InIter2,
            typename OutIter, typename F>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
        transform_binary(ExPolicy const&, InIter1 first1, InIter1 last1,
            InIter2 first2, OutIter dest, F && f, boost::mpl::true_)
        {
            try {
                detail::synchronize_binary(first1, last1, first2);
                return detail::algorithm_result<ExPolicy, OutIter>::get(
                    std::transform(first1, last1, first2, dest,
                        std::forward<F>(f)));
            }
            catch (...) {
                detail::handle_exception<ExPolicy>::call();
            }
        }

        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename OutIter, typename F>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
        transform_binary(ExPolicy const& policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, OutIter dest, F && f, boost::mpl::false_ fls)
        {
            typedef boost::tuple<FwdIter1, FwdIter2, OutIter> iterator_tuple;
            typedef detail::zip_iterator<iterator_tuple> zip_iterator;
            typedef typename zip_iterator::reference reference;
            typedef
                typename detail::algorithm_result<ExPolicy, OutIter>::type
            result_type;

            return get_iter<2, result_type>(
                for_each_n(policy,
                    detail::make_zip_iterator(boost::make_tuple(first1, first2, dest)),
                    std::distance(first1, last1),
                    [f](reference it) {
                        *boost::get<2>(it) =
                            f(*boost::get<0>(it), *boost::get<1>(it));
                    },
                    fls));
        }

        template <typename InIter1, typename InIter2, typename OutIter,
            typename F>
        OutIter transform_binary(execution_policy const& policy,
            InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest,
            F && f, boost::mpl::false_ fls)
        {
            switch (detail::which(policy))
            {
            case detail::execution_policy_enum::sequential:
                return detail::transform_binary(
                    *policy.get<sequential_execution_policy>(),
                    first1, last1, first2, dest, std::forward<F>(f),
                    boost::mpl::true_());

            case detail::execution_policy_enum::parallel:
                return detail::transform_binary(
                    *policy.get<parallel_execution_policy>(),
                    first1, last1, first2, dest, std::forward<F>(f),
                    fls);

            case detail::execution_policy_enum::vector:
                return detail::transform_binary(
                    *policy.get<vector_execution_policy>(),
                    first1, last1, first2, dest, std::forward<F>(f),
                    fls);

            case detail::execution_policy_enum::task:
                // the dynamic case will never return a future
                return detail::transform_binary(par,
                    first1, last1, first2, dest, std::forward<F>(f),
                    fls);

            default:
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "hpx::parallel::detail::transform_binary",
                    "Not supported execution policy");
                break;
            }
        }

        template<typename InIter1, typename InIter2, typename OutIter,
            typename F>
        OutIter transform_binary(execution_policy const& policy,
            InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest,
            F && f, boost::mpl::true_ t)
        {
            return detail::transform_binary(sequential_execution_policy(),
                first1, last1, first2, dest, std::forward<F>(f),
                t);
        }
    }

    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename OutIter, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    transform(ExPolicy && policy, InIter1 first1, InIter1 last1, InIter2 first2,
        OutIter dest, F && f)
    {
        typedef typename std::iterator_traits<InIter1>::iterator_category category1;
        typedef typename std::iterator_traits<InIter2>::iterator_category category2;

        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag, category1>::value,
            "Required at least input iterator.");
        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag, category2>::value,
            "Required at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, category1>,
            boost::is_same<std::input_iterator_tag, category2>
        >::type is_seq;

        return detail::transform_binary(
            std::forward<ExPolicy>(policy),
            first1, last1, first2,
            dest, std::forward<F>(f), is_seq());
    }
}}

#endif
