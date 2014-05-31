//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_STL_DETAIL_FOR_EACH_MAY_29_2014_0932PM)
#define HPX_STL_DETAIL_FOR_EACH_MAY_29_2014_0932PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/stl/execution_policy.hpp>
#include <hpx/stl/detail/algorithm_result.hpp>
#include <hpx/stl/util/partitioner.hpp>
#include <hpx/stl/util/loop.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/util/void_guard.hpp>
#include <hpx/util/decay.hpp>

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
        template <typename ExPolicy, typename InIter, typename F, typename IterTag>
        typename detail::algorithm_result<ExPolicy, InIter>::type
        for_each_n_seq(ExPolicy const&, InIter first,
            std::size_t count, F && f, IterTag)
        {
            try {
                util::loop_n<IterTag>::call(first, count, std::forward<F>(f));
                return detail::algorithm_result<ExPolicy, InIter>::get(
                    std::move(first));
            }
            catch(std::bad_alloc const& e) {
                throw e;
            }
            catch (...) {
                boost::throw_exception(
                    hpx::exception_list(boost::current_exception())
                );
            }
        }

        template <typename ExPolicy, typename InIter, typename F,
            typename IterTag>
        typename detail::algorithm_result<ExPolicy, InIter>::type
        for_each_n(ExPolicy const&, InIter first, std::size_t count, F && f, IterTag)
        {
            typedef typename hpx::util::decay<ExPolicy>::type execution_policy_type;
            BOOST_STATIC_ASSERT_MSG(
                is_execution_policy<execution_policy_type>::value,
                "Given type is not a execution policy");

            if (count > 0)
            {
                return util::partitioner<execution_policy_type>::call(
                    first, count,
                    [f](InIter part_begin, std::size_t part_count)
                    {
                        util::loop_n<IterTag>::call(part_begin, part_count, f);
                    });
            }

            return detail::algorithm_result<ExPolicy, InIter>::get(
                std::move(first));
        }

        template <typename ExPolicy, typename InIter, typename F>
        typename boost::enable_if<
            is_parallel_execution_policy<ExPolicy>,
            typename detail::algorithm_result<ExPolicy, InIter>::type
        >::type
        for_each_n(ExPolicy const& policy, InIter first, std::size_t count,
            F && f, std::input_iterator_tag category)
        {
            return detail::for_each_n_seq(policy, first, count,
                std::forward<F>(f), category);
        }

        template <typename InIter, typename F, typename IterTag>
        InIter for_each_n(sequential_execution_policy const& policy,
            InIter first, std::size_t count, F && f, IterTag category)
        {
            return detail::for_each_n_seq(policy, first, count,
                std::forward<F>(f), category);
        }

        template <typename InIter, typename F, typename IterTag>
        InIter for_each_n(execution_policy const& policy,
            InIter first, std::size_t count, F && f, IterTag category)
        {
            switch (detail::which(policy))
            {
            case detail::execution_policy_enum::sequential:
                return detail::for_each_n_seq(
                    *policy.get<sequential_execution_policy>(),
                    first, count, std::forward<F>(f), category);

            case detail::execution_policy_enum::parallel:
                return detail::for_each_n(
                    *policy.get<parallel_execution_policy>(),
                    first, count, std::forward<F>(f), category);

            case detail::execution_policy_enum::vector:
                return detail::for_each_n(
                    *policy.get<vector_execution_policy>(),
                    first, count, std::forward<F>(f), category);

            case detail::execution_policy_enum::task:
                // the dynamic case will never return a future
                return detail::for_each_n(
                    par, first, count, std::forward<F>(f), category);

            default:
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "hpx::parallel::detail::for_each_n",
                    "Not supported execution policy");
                break;
            }
        }
    }

    template <typename ExPolicy, typename InIter, typename F>
    typename boost::enable_if<
        is_execution_policy<typename hpx::util::decay<ExPolicy>::type>,
        typename detail::algorithm_result<ExPolicy, InIter>::type
    >::type
    for_each_n(ExPolicy && policy, InIter first, std::size_t count, F && f)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag, iterator_category>::value,
            "Requires at least input iterator.");

        return detail::for_each_n(
            std::forward<ExPolicy>(policy),
            first, count, std::forward<F>(f), iterator_category());
    }

    ///////////////////////////////////////////////////////////////////////////
    // for_each
    namespace detail
    {
        template <typename ExPolicy, typename InIter, typename F, typename IterTag>
        typename detail::algorithm_result<ExPolicy, void>::type
        for_each_seq(ExPolicy const&, InIter first, InIter last, F && f, IterTag)
        {
            try {
                std::for_each(first, last, std::forward<F>(f));
                return detail::algorithm_result<ExPolicy, void>::get();
            }
            catch(std::bad_alloc const& e) {
                throw e;
            }
            catch (...) {
                boost::throw_exception(
                    hpx::exception_list(boost::current_exception())
                );
            }
        }

        template <typename ExPolicy, typename InIter, typename F, typename IterTag>
        typename detail::algorithm_result<ExPolicy, void>::type
        for_each(ExPolicy const& policy, InIter first, InIter last, F && f,
            IterTag category)
        {
            typedef
                typename detail::algorithm_result<ExPolicy, void>::type
            result_type;

            return hpx::util::void_guard<result_type>(),
                detail::for_each_n(policy, first, std::distance(first, last),
                    std::forward<F>(f), category);
        }

        template <typename ExPolicy, typename InIter, typename F>
        typename boost::enable_if<
            is_parallel_execution_policy<ExPolicy>,
            typename detail::algorithm_result<ExPolicy, void>::type
        >::type
        for_each(ExPolicy const& policy, InIter first, InIter last, F && f,
            std::input_iterator_tag category)
        {
            return detail::for_each_seq(policy, first, last,
                std::forward<F>(f), category);
        }

        template <typename InIter, typename F, typename IterTag>
        void for_each(sequential_execution_policy const& policy,
            InIter first, InIter last, F && f, IterTag category)
        {
            detail::for_each_seq(policy, first, last,
                std::forward<F>(f), category);
        }

        template <typename InIter, typename F, typename IterTag>
        void for_each(execution_policy const& policy,
            InIter first, InIter last, F && f, IterTag category)
        {
            switch (detail::which(policy))
            {
            case detail::execution_policy_enum::sequential:
                detail::for_each_seq(
                    *policy.get<sequential_execution_policy>(),
                    first, last, std::forward<F>(f), category);
                break;

            case detail::execution_policy_enum::parallel:
                detail::for_each(
                    *policy.get<parallel_execution_policy>(),
                    first, last, std::forward<F>(f), category);
                break;

            case detail::execution_policy_enum::vector:
                detail::for_each(
                    *policy.get<vector_execution_policy>(),
                    first, last, std::forward<F>(f), category);
                break;

            case detail::execution_policy_enum::task:
                // the dynamic case will never return a future
                detail::for_each(par, first, last, std::forward<F>(f),
                    category);
                break;

            default:
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "hpx::parallel::detail::for_each",
                    "Not supported execution policy");
                break;
            }
        }
    }

    template <typename ExPolicy, typename InIter, typename F>
    typename boost::enable_if<
        is_execution_policy<typename hpx::util::decay<ExPolicy>::type>,
        typename detail::algorithm_result<ExPolicy, void>::type
    >::type
    for_each(ExPolicy && policy, InIter first, InIter last, F && f)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag, iterator_category>::value,
            "Requires at least input iterator.");

        return detail::for_each(
            std::forward<ExPolicy>(policy),
            first, last, std::forward<F>(f), iterator_category());
    }
}}

#endif
