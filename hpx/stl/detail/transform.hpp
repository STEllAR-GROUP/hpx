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
            typename F, typename IterTag>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
        transform_seq(ExPolicy const&, InIter first, InIter last, OutIter dest,
            F && f, IterTag)
        {
            try {
                return detail::algorithm_result<ExPolicy, OutIter>::get(
                    std::transform(first, last, dest, std::forward<F>(f)));
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
        R get_iter(hpx::future<ZipIter> && zipiter)
        {
            return zipiter.then(
                [](hpx::future<ZipIter> f) {
                    typename std::iterator_traits<ZipIter>::value_type t =
                        *f.get();
                    return boost::get<N>(t);
                });
        }

        template <typename ExPolicy, typename InIter, typename OutIter,
            typename F, typename IterTag>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
        transform(ExPolicy const& policy, InIter first, InIter last,
            OutIter dest, F && f, IterTag category)
        {
            typedef boost::tuple<InIter, OutIter> iterator_tuple;
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
                    category));
        }

        template <typename ExPolicy, typename InIter, typename OutIter, typename F>
        typename boost::enable_if<
            is_parallel_execution_policy<ExPolicy>,
            typename detail::algorithm_result<ExPolicy, OutIter>::type
        >::type
        transform (ExPolicy const& policy, InIter first, InIter last,
            OutIter dest, F && f, std::input_iterator_tag category)
        {
            return transform_seq(policy, first, last, dest,
                std::forward<F>(f), category);
        }

        template <typename InIter, typename OutIter, typename F, typename IterTag>
        OutIter transform(execution_policy const& policy,
            InIter first, InIter last, OutIter dest, F && f, IterTag category)
        {
            switch (detail::which(policy))
            {
            case detail::execution_policy_enum::sequential:
                return detail::transform(
                    *policy.get<sequential_execution_policy>(),
                    first, last, dest, std::forward<F>(f), category);

            case detail::execution_policy_enum::parallel:
                return detail::transform(
                    *policy.get<parallel_execution_policy>(),
                    first, last, dest, std::forward<F>(f), category);

            case detail::execution_policy_enum::vector:
                return detail::transform(
                    *policy.get<vector_execution_policy>(),
                    first, last, dest, std::forward<F>(f), category);

            case detail::execution_policy_enum::task:
                // the dynamic case will never return a future
                return detail::transform(par,
                    first, last, dest, std::forward<F>(f), category);

            default:
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "hpx::parallel::detail::transform",
                    "Not supported execution policy");
                break;
            }
        }
    }

    template <typename ExPolicy, typename InIter, typename OutIter, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    transform(ExPolicy&& policy, InIter first, InIter last, OutIter dest, F && f)
    {
        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag,
                typename std::iterator_traits<InIter>::iterator_category>::value,
            "Required at least input iterator.");

        std::iterator_traits<InIter>::iterator_category category;
        return detail::transform(policy, first, last, dest,
            std::forward<F>(f), category);
    }

    ///////////////////////////////////////////////////////////////////////////
    // transform binary predicate
    namespace detail
    {
        template <typename ExPolicy, typename InIter1, typename InIter2,
            typename OutIter, typename F, typename IterTag>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
        transform_binary_seq(ExPolicy const &, InIter1 first1, InIter1 last1,
            InIter2 first2, OutIter dest, F && f, IterTag)
        {
            try {
                return detail::algorithm_result<ExPolicy, OutIter>::get(
                    std::transform(first1, last1, first2, dest,
                        std::forward<F>(f)));
            }
            catch(std::bad_alloc const& e) {
                throw e;
            }
            catch (...) {
                throw hpx::exception_list(boost::current_exception());
            }
        }

        template <typename ExPolicy, typename InIter1, typename InIter2,
            typename OutIter, typename F, typename IterTag>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
        transform_binary(ExPolicy const& policy, InIter1 first1, InIter1 last1,
            InIter2 first2, OutIter dest, F && f, IterTag category)
        {
            typedef boost::tuple<InIter1, InIter2, OutIter> iterator_tuple;
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
                    category));
        }

        template <typename ExPolicy, typename InIter1, typename InIter2,
             typename OutIter, typename F>
        inline typename boost::enable_if<
            is_parallel_execution_policy<ExPolicy>,
            typename detail::algorithm_result<ExPolicy, OutIter>::type
        >::type
        transform_binary(ExPolicy const& policy, InIter1 first1, InIter1 last1,
            InIter2 first2, OutIter dest, F && f, std::input_iterator_tag category)
        {
            return transform_binary_seq(policy, first1, last1, first2, dest,
                std::forward<F>(f), category);
        }

        template <typename ExPolicy, typename InIter1, typename InIter2,
            typename OutIter, typename F, typename IterTag>
        OutIter transform_binary(execution_policy const& policy,
            InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest,
            F && f, IterTag category)
        {
            switch (detail::which(policy))
            {
            case detail::execution_policy_enum::sequential:
                return detail::transform_binary_seq(
                    *policy.get<sequential_execution_policy>(),
                    first1, last1, first2, dest, std::forward<F>(f), category);

            case detail::execution_policy_enum::parallel:
                return detail::transform_binary(
                    *policy.get<parallel_execution_policy>(),
                    first1, last1, first2, dest, std::forward<F>(f), category);

            case detail::execution_policy_enum::vector:
                return detail::transform_binary(
                    *policy.get<vector_execution_policy>(),
                    first1, last1, first2, dest, std::forward<F>(f), category);

            case detail::execution_policy_enum::task:
                // the dynamic case will never return a future
                return detail::transform_binary(par,
                    first1, last1, first2, dest, std::forward<F>(f), category);

            default:
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "hpx::parallel::detail::transform_binary",
                    "Not supported execution policy");
                break;
            }
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
        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag,
                typename std::iterator_traits<InIter1>::iterator_category>::value,
            "Required at least input iterator.");
        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag,
                typename std::iterator_traits<InIter2>::iterator_category>::value,
            "Required at least input iterator.");

        detail::zip_iterator_category_helper<InIter1, InIter2>::iterator_category category;
        return detail::transform_binary(policy, first1, last1, first2,
            dest, std::forward<F>(f), category);
    }
}}

#endif
