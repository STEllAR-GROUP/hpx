//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DETAIL_FILL_JUNE_12_2014_0405PM)
#define HPX_PARALLEL_DETAIL_FILL_JUNE_12_2014_0405PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/detail/algorithm_result.hpp>
#include <hpx/parallel/detail/synchronize.hpp>
#include <hpx/parallel/util/loop.hpp>
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
    // fill
    namespace detail
    {
        template <typename ExPolicy, typename InIter, typename T>
        typename detail::algorithm_result<ExPolicy, void>::type
        fill(ExPolicy const&, InIter first, InIter last, T val,
            boost::mpl::true_)
        {
            try {
                std::fill(first, last, val);
                return detail::algorithm_result<ExPolicy, void>::get();
            }
            catch(...) {
                detail::handle_exception<ExPolicy>::call();
            }
        }

        template <typename ExPolicy, typename FwdIter, typename T>
        typename detail::algorithm_result<ExPolicy, void>::type
        fill(ExPolicy const& policy, FwdIter first, FwdIter last, T val,
            boost::mpl::false_ f)
        {
            typedef typename detail::algorithm_result<ExPolicy, void>::type
                result_type;
            typedef typename std::iterator_traits<FwdIter>::value_type type;

            return hpx::util::void_guard<result_type>(),
                plain_for_each_n(policy, first,
                    std::distance(first, last),
                    [val](type& v){
                        v = val;
                    }, f);
        }

        template <typename InIter, typename T>
        void fill(execution_policy const& policy,
            InIter first, InIter last, T val, boost::mpl::false_)
        {
            HPX_PARALLEL_DISPATCH(policy, detail::fill, first, last, val);
        }

       template <typename InIter, typename T>
        void fill(execution_policy const& policy,
            InIter first, InIter last, T val, boost::mpl::true_ t)
        {
            detail::fill(sequential_execution_policy(),
                first, last, val, t);
        }
    }

    template <typename ExPolicy, typename InIter, typename T>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, void>::type
    >::type
    fill(ExPolicy && policy, InIter first, InIter last, T val)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
             iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::forward_iterator_tag, iterator_category>::value),
            "Required at least forward iterator.");

        typedef typename is_sequential_execution_policy<ExPolicy>::type is_seq;

        return detail::fill( std::forward<ExPolicy>(policy),
            first, last, val, is_seq());
    }
    ///////////////////////////////////////////////////////////////////////////
    // fill_n
    namespace detail
    {
        template <typename ExPolicy, typename OutIter, typename T>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
        fill_n(ExPolicy const&, OutIter first, std::size_t count, T val,
        boost::mpl::true_)
        {
            try {
                return detail::algorithm_result<ExPolicy, OutIter>::get(
                    std::fill_n(first, count, val));
            }
            catch (...) {
                detail::handle_exception<ExPolicy>::call();
            }
        }

        template <typename ExPolicy, typename OutIter, typename T>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
        fill_n(ExPolicy const& policy, OutIter first, std::size_t count, T val,
            boost::mpl::false_ f)
        {
            typedef typename std::iterator_traits<OutIter>::iterator_category
                category;
            typedef typename std::iterator_traits<OutIter>::value_type type;

            return plain_for_each_n(policy, first, count,
                        [val](type& v) {
                            v = val;
                        }, f);

        }

        template <typename OutIter, typename T>
        OutIter fill_n(execution_policy const& policy,
            OutIter first, std::size_t count, T val, boost::mpl::false_)
        {
            HPX_PARALLEL_DISPATCH(policy, detail::fill_n, first, count, val);
        }

        template <typename OutIter, typename T>
        OutIter fill_n(execution_policy const& policy,
            OutIter first, std::size_t count, T val, boost::mpl::true_ t)
        {
            return detail::fill_n(sequential_execution_policy(),
                first, count, val, t);
        }
    }

    template <typename ExPolicy, typename OutIter, typename T>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    fill_n(ExPolicy && policy, OutIter first, std::size_t count, T val)
    {
        typedef typename std::iterator_traits<OutIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::mpl::or_<
                boost::is_base_of<
                    std::forward_iterator_tag, iterator_category>,
                boost::is_same<
                    std::output_iterator_tag, iterator_category>
            >::value),
            "Requires at least output iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::output_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::fill_n(
            std::forward<ExPolicy>(policy),
            first, count, val, is_seq());
    }
}}

#endif
