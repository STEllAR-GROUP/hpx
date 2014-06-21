//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_STL_DETAIL_COUNT_JUNE_17_2014_1154AM)
#define HPX_STL_DETAIL_COUNT_JUNE_17_2014_1154AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/stl/execution_policy.hpp>
#include <hpx/stl/detail/algorithm_result.hpp>
#include <hpx/stl/detail/zip_iterator.hpp>

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
            catch(std::bad_alloc const& e) {
                boost::throw_exception(e);
            }
            catch (...){
                boost::throw_exception(
                    hpx::exception_list(boost::current_exception())
                );
            }
        }

        template <typename ExPolicy, typename InIter, typename T>
        typename detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<InIter>::difference_type>::type
        count(ExPolicy const& policy, InIter first, InIter last, const T& value,
            boost::mpl::false_ f)
        {
            typedef typename std::iterator_traits<InIter>::value_type type;
            typedef typename std::iterator_traits<InIter>::difference_type
                difference_type;

            // FIXME: this is not thread safe! I'd suggest implementing this
            // similarly to reduce
            difference_type ret = 0;
            for_each_n(policy,
                first, std::distance(first, last),
                [&value, &ret](type const& v) {
                    if (v == value)
                        ++ret;
                }, f);

            return detail::algorithm_result<ExPolicy, difference_type>::get(
                std::move(ret));
        }

        template <typename InIter, typename T>
        typename std::iterator_traits<InIter>::difference_type count(
            execution_policy const& policy, InIter first, InIter last,
            const T& value, boost::mpl::false_ f)
        {
            switch (detail::which(policy))
            {
            case detail::execution_policy_enum::sequential:
                return detail::count(
                    *policy.get<sequential_execution_policy>(),
                    first, last, value, boost::mpl::true_());

            case detail::execution_policy_enum::parallel:
                return detail::count(
                    *policy.get<parallel_execution_policy>(),
                    first, last, value, f);

            case detail::execution_policy_enum::vector:
                return detail::count(
                    *policy.get<vector_execution_policy>(),
                    first, last, value, f);

            case detail::execution_policy_enum::task:
                return detail::count( par,
                    first, last, value, f);

            default:
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "hpx::parallel::detail::count",
                    "Not supported execution policy");
                break;
            }
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
        typename std::iterator_traits<InIter>::difference_type>::type
    >::type
    count(ExPolicy && policy, InIter first, InIter last, const T& value)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            category;

        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag,
                typename std::iterator_traits<InIter>::iterator_category>::value,
            "Required at least input iterator .");

        typedef boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, category>
        >::type is_seq;

        return detail::count( std::forward<ExPolicy>(policy),
            first, last, value, is_seq());
    }
    ///////////////////////////////////////////////////////////////////////////
    // count_if
    namespace detail
    {
        template <typename ExPolicy, typename InIter, typename Pred>
        typename detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<InIter>::difference_type>::type
        count_if(ExPolicy const&, InIter first, InIter last, Pred && op,
            boost::mpl::true_)
        {
            try {
                return detail::algorithm_result<ExPolicy,
                    std::iterator_traits<InIter>::difference_type>::get(
                    std::count_if(first, last, std::forward<Pred>(op)));
            }
            catch(std::bad_alloc const& e) {
                boost::throw_exception(e);
            }
            catch(...) {
                boost::throw_exception(
                    hpx::exception_list(boost::current_exception())
                );
            }
        }

        template <typename ExPolicy, typename InIter, typename Pred>
        typename detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<InIter>::difference_type>::type
        count_if(ExPolicy const& policy, InIter first, InIter last,
            Pred && op, boost::mpl::false_ f)
        {
            typedef typename std::iterator_traits<InIter>::value_type type;
<<<<<<< HEAD
    
            plain_for_each_n(policy,
=======
            typedef typename std::iterator_traits<InIter>::difference_type
                difference_type;

            // FIXME: this is not thread safe! I'd suggest implementing this
            // similarly to reduce
            difference_type ret = 0;

            for_each_n(policy,
>>>>>>> 88b9df4dcacc7e13297d2a090931cf260616a78a
                first, std::distance(first, last),
                [op, &ret](type const& v) {
                    if (op(v))
                        ++ret;
                }, f);

            return detail::algorithm_result<ExPolicy, difference_type>::get(
                std::move(ret));
        }

        template <typename InIter, typename Pred>
        typename std::iterator_traits<InIter>::difference_type count_if(
            execution_policy const& policy, InIter first, InIter last,
            Pred && op, boost::mpl::false_ f)
        {
            switch (detail::which(policy))
            {
            case detail::execution_policy_enum::sequential:
                return detail::count_if(
                    *policy.get<sequential_execution_policy>(),
                    first, last, std::forward<Pred>(op), boost::mpl::true_());

            case detail::execution_policy_enum::parallel:
                return detail::count_if(
                    *policy.get<parallel_execution_policy>(),
                    first, last, std::forward<Pred>(op), f);

            case detail::execution_policy_enum::vector:
                return detail::count_if(
                    *policy.get<vector_execution_policy>(),
                    first, last, std::forward<Pred>(op), f);

            case detail::execution_policy_enum::task:
                return detail::count_if( par,
                    first, last, std::forward<Pred>(op), f);

            default:
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "hpx::parallel::detail::count_if",
                    "Not supported execution policy");
                break;
            }
        }

        template <typename InIter, typename Pred>
        typename std::iterator_traits<InIter>::difference_type count_if(
            execution_policy const& policy, InIter first, InIter last,
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
            typename std::iterator_traits<InIter>::difference_type>::type
    >::type
    count_if(ExPolicy && policy, InIter first, InIter last, Pred && op)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            category;

        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag, category>::value,
            "Required at least input iterator.");

        typedef boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, category>
        >::type is_seq;

        return detail::count_if( std::forward<ExPolicy>(policy),
            first, last,
            std::forward<Pred>(op), is_seq());
    }
}}

#endif
