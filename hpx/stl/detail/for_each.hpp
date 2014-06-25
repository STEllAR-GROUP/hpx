//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DETAIL_FOR_EACH_MAY_29_2014_0932PM)
#define HPX_PARALLEL_DETAIL_FOR_EACH_MAY_29_2014_0932PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/stl/execution_policy.hpp>
#include <hpx/stl/detail/algorithm_result.hpp>
#include <hpx/stl/detail/synchronize.hpp>
#include <hpx/stl/util/partitioner.hpp>
#include <hpx/stl/util/loop.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/util/void_guard.hpp>
#include <hpx/util/move.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel
{
    ///////////////////////////////////////////////////////////////////////////
    // for_each_n
    namespace detail
    {
        template <typename ExPolicy, typename InIter, typename F>
        typename detail::algorithm_result<ExPolicy, InIter>::type
        for_each_n(ExPolicy const&, InIter first,
            std::size_t count, F && f, boost::mpl::true_)
        {
            try {
                return detail::algorithm_result<ExPolicy, InIter>::get(
                    util::loop_n(first, count, std::forward<F>(f)));
            }
            catch (...) {
                detail::handle_exception<ExPolicy>::call();
            }
        }

        template <typename ExPolicy, typename FwdIter, typename F>
        typename detail::algorithm_result<ExPolicy, FwdIter>::type
        for_each_n(ExPolicy const& policy, FwdIter first, std::size_t count,
            F && f, boost::mpl::false_)
        {
            if (count > 0)
            {
                return util::partitioner<ExPolicy>::call(
                    policy, first, count,
                    [f](FwdIter part_begin, std::size_t part_count)
                    {
                        util::loop_n(part_begin, part_count, f);
                    });
            }

            return detail::algorithm_result<ExPolicy, FwdIter>::get(
                std::move(first));
        }

        template <typename ExPolicy, typename InIter, typename F>
        typename detail::algorithm_result<ExPolicy, InIter>::type
        plain_for_each_n(ExPolicy const&, InIter first,
            std::size_t count, F && f, boost::mpl::true_)
        {
            try {
                return detail::algorithm_result<ExPolicy, InIter>::get(
                    util::plain_loop_n(first, count, std::forward<F>(f)));
            }
            catch (...) {
                detail::handle_exception<ExPolicy>::call();
            }
        }

        template <typename ExPolicy, typename FwdIter, typename F>
        typename detail::algorithm_result<ExPolicy, FwdIter>::type
        plain_for_each_n(ExPolicy const& policy, FwdIter first,
            std::size_t count, F && f, boost::mpl::false_)
        {
            if (count > 0)
            {
                return util::partitioner<ExPolicy>::call(
                    policy, first, count,
                    [f](FwdIter part_begin, std::size_t part_count)
                    {
                        util::plain_loop_n(part_begin, part_count, f);
                    });
            }

            return detail::algorithm_result<ExPolicy, FwdIter>::get(
                std::move(first));
        }

        template <typename InIter, typename F>
        InIter for_each_n(execution_policy const& policy,
            InIter first, std::size_t count, F && f, boost::mpl::false_)
        {
            HPX_PARALLEL_DISPATCH(policy, detail::for_each_n, first, count,
                std::forward<F>(f));
        }

        template <typename InIter, typename F>
        InIter for_each_n(execution_policy const& policy,
            InIter first, std::size_t count, F && f, boost::mpl::true_ t)
        {
            return detail::for_each_n(sequential_execution_policy(),
                first, count, std::forward<F>(f), t);
        }
    }

    template <typename ExPolicy, typename InIter, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, InIter>::type
    >::type
    for_each_n(ExPolicy && policy, InIter first, std::size_t count, F && f)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag, iterator_category>::value,
            "Requires at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::for_each_n(
            std::forward<ExPolicy>(policy),
            first, count, std::forward<F>(f), is_seq());
    }

    ///////////////////////////////////////////////////////////////////////////
    // for_each
    namespace detail
    {
        template <typename ExPolicy, typename InIter, typename F>
        typename detail::algorithm_result<ExPolicy, void>::type
        for_each(ExPolicy const&, InIter first, InIter last, F && f,
            boost::mpl::true_)
        {
            try {
                detail::synchronize(first, last);
                std::for_each(first, last, std::forward<F>(f));
                return detail::algorithm_result<ExPolicy, void>::get();
            }
            catch (...) {
                detail::handle_exception<ExPolicy>::call();
            }
        }

        template <typename ExPolicy, typename FwdIter, typename F>
        typename detail::algorithm_result<ExPolicy, void>::type
        for_each(ExPolicy const& policy, FwdIter first, FwdIter last, F && f,
            boost::mpl::false_ fls)
        {
            typedef
                typename detail::algorithm_result<ExPolicy, void>::type
            result_type;

            return hpx::util::void_guard<result_type>(),
                detail::for_each_n(policy, first, std::distance(first, last),
                    std::forward<F>(f), boost::mpl::false_());
        }

        template <typename InIter, typename F>
        void for_each(execution_policy const& policy,
            InIter first, InIter last, F && f, boost::mpl::false_)
        {
            HPX_PARALLEL_DISPATCH(policy, detail::for_each, first, last,
                std::forward<F>(f));
        }

        template <typename InIter, typename F>
        void for_each(execution_policy const& policy,
            InIter first, InIter last, F && f, boost::mpl::true_ t)
        {
            detail::for_each(sequential_execution_policy(),
                first, last, std::forward<F>(f), t);
        }
    }

    template <typename ExPolicy, typename InIter, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, void>::type
    >::type
    for_each(ExPolicy && policy, InIter first, InIter last, F && f)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag, iterator_category>::value,
            "Requires at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::for_each(
            std::forward<ExPolicy>(policy),
            first, last, std::forward<F>(f), is_seq());
    }
}}

#endif
