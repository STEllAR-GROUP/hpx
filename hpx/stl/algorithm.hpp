//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_STL_PARALLEL_ALGORITHM_MAY_27_2014_0904PM)
#define HPX_STL_PARALLEL_ALGORITHM_MAY_27_2014_0904PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/stl/execution_policy.hpp>
#include <hpx/stl/util/partitioner.hpp>
#include <hpx/stl/util/loop.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/declval.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel
{
    ///////////////////////////////////////////////////////////////////////////
    // for_each_n
    namespace detail
    {
        template <typename InIter, typename Diff, typename F, typename IterTag>
        inline InIter for_each_n(sequential_execution_policy const &,
            InIter first, Diff count, F && func, IterTag)
        {
            try {
                for (/**/; count != 0; --count, ++first)
                    func(*first);
            }
            catch(std::bad_alloc const& e) {
                throw e;
            }
            catch (...) {
                // throw exception_list
            }
            return first;
        }

        template <typename ExPolicy, typename InIter, typename Diff,
            typename F, typename IterTag>
        inline InIter for_each_n(ExPolicy const&, InIter first, Diff count,
            F && func, IterTag)
        {
            typedef typename util::decay<ExPolicy>::type execution_policy_type;
            BOOST_STATIC_ASSERT_MSG(
                is_execution_policy<execution_policy_type>::value,
                "Given type is not a execution policy");

            if (count > 0)
            {
                typedef std::iterator_traits<InIter>::reference reference;

                return util::partitioner<execution_policy_type>::call(
                    first, count, [func](InIter part_begin, std::size_t count)
                    {
                        util::loop<InIter>::call(
                            part_begin, count, [&func](reference it)
                            {
                                func(it);
                            });
                    });
            }
            return first;
        }

        template <typename ExPolicy, typename InIter, typename Diff, typename F>
        inline typename detail::enable_if_parallel<ExPolicy, InIter>::type
        for_each_n(ExPolicy const&, InIter first, Diff count, F && func,
            std::input_iterator_tag category)
        {
            return detail::for_each_n(seq, first, count, std::forward<F>(func),
                category);
        }

        template <typename InIter, typename Diff, typename F, typename IterTag>
        inline InIter For_each_n(execution_policy cosnt& policy, InIter first,
            Diff count, F && func, IterTag category)
        {
            std::type_info const& t = policy.type();
            if (t == typeid(parallel_execution_policy)) {
                return detail::for_each_n(
                    *policy.get<parallel_execution_policy>(),
                    first, count, std::forward<F>(func), category);
            }
            else if (t == typeid(sequential_execution_policy)) {
                return detail::for_each_n(
                    *policy.get<sequential_execution_policy>(),
                    first, count, std::forward<F>(func), category);
            }
//             else if (t == typeid(vector_execution_policy)) {
//                 return detail::for_each_n(
//                     *policy.get<vector_execution_policy>(),
//                     first, count, std::forward<F>(func), category);
//             }
            else {
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "hpx::parallel::detail::for_each_n"
                    "Not supported execution policy");
            }
        }
    }

    template <typename ExPolicy, typename InIter, typename Diff, typename F>
    inline typename detail::enable_if_policy<ExPolicy, InIter>::type
    for_each_n(ExPolicy && policy, InIter first, Diff count, F && func)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag, iterator_category>::value,
            "Requires at least input iterator.");

        return detail::for_each_n(
            std::forward<ExPolicy>(policy), first, count,
            std::forward<F>(func), boost::declval<iterator_category>());
    }

    ///////////////////////////////////////////////////////////////////////////
    // for_each
    namespace detail
    {
        template <typename InIter, typename F, typename IterTag>
        inline void for_each(sequential_execution_policy const&,
            InIter first, InIter last, F && func, IterTag)
        {
            try {
                std::for_each(first, last, std::forward<F>(func));
            }
            catch(std::bad_alloc const& e) {
                throw e;
            }
            catch (...) {
                // throw exception_list
            }
        }

        template <typename ExPolicy, typename InIter, typename F, typename IterTag>
        inline void for_each(ExPolicy const& policy, InIter first, InIter last,
            F && func, IterTag category)
        {
            detail::for_each_n(policy, first, std::distance(first, last),
                std::forward<F>(func), category);
        }

        template <typename ExPolicy, typename InIter, typename F>
        inline typename detail::enable_if_parallel<ExPolicy, void>::type
        for_each(ExPolicy const&, InIter first, InIter last, F && func,
            std::input_iterator_tag category)
        {
            detail::for_each(seq, first, last, std::forward<F>(func), category);
        }

        template <typename InIter, typename F, typename IterTag>
        inline void for_each(execution_policy const& policy,
            InIter first, InIter last, F && func, IterTag category)
        {
            std::type_info const& t = policy.type();
            if (t == typeid(parallel_execution_policy)) {
                return detail::for_each(*policy.get<parallel_execution_policy>(),
                    first, last, std::forward<F>(func), category);
            }
            else if (t == typeid(sequential_execution_policy)) {
                return detail::for_each(*policy.get<sequential_execution_policy>(),
                    first, last, std::forward<F>(func), category);
            }
//             else if (t == typeid(vector_execution_policy)) {
//                 return detail::for_each(*policy.get<vector_execution_policy>(),
//                     first, last, std::forward<F>(func), category);
//             }
            else {
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "hpx::parallel::detail::for_each"
                    "Not supported execution policy");
            }
        }
    }

    template <typename ExPolicy, typename InIter, typename F>
    inline typename detail::enable_if_policy<ExPolicy, void>::type
    for_each(ExPolicy && policy, InIter first, InIter last, F && func)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag, iterator_category>::value,
            "Requires at least input iterator.");

        return detail::for_each(
            std::forward<ExPolicy>(policy), first, last,
            std::forward<F>(func), boost::declval<iterator_category>());
    }
}}

#endif

