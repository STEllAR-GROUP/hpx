//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_STL_DETAIL_COPY_MAY_30_2014_0317PM)
#define HPX_STL_DETAIL_COPY_MAY_30_2014_0317PM

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
    // copy
    namespace detail
    {
        template<typename ExPolicy, typename InIter, typename OutIter,
            typename IterTag>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
        copy_seq(ExPolicy const&, InIter first, InIter last, OutIter dest,
            IterTag)
        {
            try {
                return detail::algorithm_result<ExPolicy, OutIter>::get(
                    std::copy(first,last,dest));
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
    
        template<typename ExPolicy, typename InIter, typename OutIter,
            typename IterTag>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
        copy(ExPolicy const& policy, InIter first, InIter last, OutIter dest, 
            IterTag category)
        {
            typedef boost::tuple<InIter, OutIter> iterator_tuple;
            typedef detail::zip_iterator<iterator_tuple> zip_iterator;
            typedef typename zip_iterator::reference reference;
            typedef
                typename detail::algorithm_result<ExPolicy, OutIter>::type
            result_type;

            return get_iter<1, result_type>(
                for_each_n(policy,
                    detail::make_zip_iterator(boost::make_tuple(first,dest)),
                    std::distance(first,last),
                    [](reference it) {
                        *boost::get<1>(it) = *boost::get<0>(it);
                    },
                    category));
        }

        template<typename ExPolicy, typename InIter, typename OutIter>
        typename boost::enable_if<
            is_parallel_execution_policy<ExPolicy>,
            typename detail::algorithm_result<ExPolicy, OutIter>::type
        >::type
        copy(ExPolicy const& policy, InIter first, InIter last, OutIter dest,
            std::input_iterator_tag category)
        {
            return detail::copy_seq(policy, first, last, dest, category);
        }

        template<typename InIter, typename OutIter, typename IterTag>
        OutIter copy(sequential_execution_policy const& policy,
            InIter first, InIter last, OutIter dest, IterTag category)
        {
            return detail::copy_seq(policy, first, last, dest, category);
        }

        template<typename InIter, typename OutIter, typename IterTag>
        OutIter copy(execution_policy const& policy,
            InIter first, InIter last, OutIter dest, IterTag category)
        {
            switch (detail::which(policy))
            {
            case detail::execution_policy_enum::sequential:
                return detail::copy(
                    *policy.get<sequential_execution_policy>(),
                    first, last, dest, category);

            case detail::execution_policy_enum::parallel:
                return detail::copy(
                    *policy.get<parallel_execution_policy>(),
                    first, last, dest, category);

            case detail::execution_policy_enum::vector:
                return detail::copy(
                    *policy.get<vector_execution_policy>(),
                    first, last, dest, category);

            case detail::execution_policy_enum::task:
                return detail::copy(par,
                    first, last, dest, category);

            default:
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "hpx::parallel::detail::copy",
                    "Not supported execution policy");
                break;
            }
        }
    }

    template<typename ExPolicy, typename InIter, typename OutIter>
    typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    copy(ExPolicy&& policy, InIter first, InIter last, OutIter dest)
    {
        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag,
                typename std::iterator_traits<InIter>::iterator_category>::value,
            "Required at least input iterator.");

        std::iterator_traits<InIter>::iterator_category category;
        return detail::copy(policy, first, last, dest, category);
    }
    
    /////////////////////////////////////////////////////////////////////////////
    // copy_n
    namespace detail
    {
        template<typename ExPolicy, typename InIter, typename OutIter,
            typename IterTag>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
        copy_n_seq(ExPolicy const&, InIter first, std::size_t count, OutIter dest,
            IterTag)
        {
            try{
                return detail::algorithm_result<ExPolicy, OutIter>::get(
                    std::copy_n(first,count,dest));
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

        template<typename ExPolicy, typename InIter, typename OutIter,
            typename IterTag>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
        copy_n(ExPolicy const& policy, InIter first, std::size_t count, OutIter dest,
            IterTag category)
        {
            typedef boost::tuple<InIter,OutIter> iterator_tuple;
            typedef detail::zip_iterator<iterator_tuple> zip_iterator;
            typedef typename zip_iterator::reference reference;
            typedef
                typename detail::algorithm_result<ExPolicy, OutIter>::type
            result_type;

            return get_iter<1, result_type>(
                for_each_n(policy,
                    detail::make_zip_iterator(boost::make_tuple(first,dest)),
                    count,
                    [](reference it) {
                        *boost::get<1>(it) = *boost::get<0>(it);
                }, 
                category));
        }
        
        template<typename ExPolicy, typename InIter, typename OutIter>
        typename boost::enable_if<
            is_parallel_execution_policy<ExPolicy>,
            typename detail::algorithm_result<ExPolicy, OutIter>::type
        >::type
        copy_n(ExPolicy const& policy, InIter first, std::size_t count, 
            OutIter dest, std::input_iterator_tag category)
        {
            return detail::copy_n_seq(policy, first, count, dest, category);
        }

        template<typename InIter, typename OutIter, typename IterTag>
        OutIter copy_n(sequential_execution_policy const& policy,
            InIter first, std::size_t count, OutIter dest, IterTag category)
        {
            return detail::copy_n_seq(policy, first, count, dest, category);
        }

        template<typename InIter, typename OutIter, typename IterTag>
        OutIter copy_n(execution_policy const& policy,
            InIter first, std::size_t count, OutIter dest, IterTag category)
        {
            switch (detail::which(policy))
            {
            case detail::execution_policy_enum::sequential:
                return detail::copy_n(
                    *policy.get<sequential_execution_policy>(),
                    first, count, dest, category);

            case detail::execution_policy_enum::parallel:
                return detail::copy_n(
                    *policy.get<parallel_execution_policy>(),
                    first, count, dest, category);

            case detail::execution_policy_enum::vector:
                return detail::copy_n(
                    *policy.get<parallel_execution_policy>(),
                    first, count, dest, category);

            case detail::execution_policy_enum::task:
                return detail::copy_n(par,
                    first, count, dest, category);

            default:
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "hpx::parallel::detail::copy_n",
                    "Not supported execution policy");
                break;
            }
        }
    }

    template<typename ExPolicy, typename InIter, typename OutIter>
    typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    copy_n(ExPolicy && policy, InIter first, std::size_t count, OutIter dest)
    {
        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag,
                typename std::iterator_traits<InIter>::iterator_category>::value,
            "Required at least input iterator.");

        std::iterator_traits<InIter>::iterator_category category;
        return detail::copy_n(policy, first, count, dest, category);
    }
    
    /////////////////////////////////////////////////////////////////////////////
    // copy_if
    namespace detail
    {
        template<typename ExPolicy, typename InIter, typename OutIter,
            typename F, typename IterTag>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
        copy_if_seq(ExPolicy const&, InIter first, InIter last, OutIter dest,
            F && f, IterTag)
        {
            try{
                return detail::algorithm_result<ExPolicy, OutIter>::get(
                    std::copy_if(first,last,dest,std::forward<F>(f)));
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

        template <typename ExPolicy, typename InIter, typename OutIter,
            typename F, typename IterTag>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
        copy_if(ExPolicy const& policy, InIter first, InIter last, OutIter dest,
            F && f, IterTag category)
        {
            typedef boost::tuple<InIter, OutIter> iterator_tuple;
            typedef detail::zip_iterator<iterator_tuple> zip_iterator;
            typedef typename zip_iterator::reference reference;
            typedef
                typename detail::algorithm_result<ExPolicy, OutIter>::type
            result_type;

            return get_iter<1,result_type>(
                for_each_n(policy,
                detail::make_zip_iterator(boost::make_tuple(first, dest)),
                std::distance(first,last),
                [f](reference it) {
                    if(f(*boost::get<0>(it)))
                        *boost::get<1>(it)=*boost::get<0>(it);
                },
                category));
        }

        template <typename ExPolicy, typename InIter, typename OutIter, typename F>
        typename boost::enable_if<
            is_parallel_execution_policy<ExPolicy>,
            typename detail::algorithm_result<ExPolicy, OutIter>::type
        >::type
        copy_if (ExPolicy const& policy, InIter first, InIter last,
            OutIter dest, F && f, std::input_iterator_tag category)
        {
            return copy_if_seq(policy, first, last, dest,
                std::forward<F>(f), category);
        }

        template<typename InIter, typename OutIter, typename F, typename IterTag>
        OutIter copy_if(sequential_execution_policy const& policy,
            InIter first, InIter last, OutIter dest, F && f, IterTag category)
        {
            return detail::copy_if_seq(policy, first, last, dest,
                std::forward<F>(f), category);
        }

        template <typename InIter, typename OutIter, typename F, typename IterTag>
        OutIter copy_if(execution_policy const& policy,
            InIter first, InIter last, OutIter dest, F && f, IterTag category)
        {
            switch(detail::which(policy))
            {
            case detail::execution_policy_enum::sequential:
                return detail::copy_if(
                    *policy.get<sequential_execution_policy>(),
                    first, last, dest, std::forward<F>(f), category);

            case detail::execution_policy_enum::parallel:
                return detail::copy_if(
                    *policy.get<parallel_execution_policy>(),
                    first, last, dest, std::forward<F>(f), category);

            case detail::execution_policy_enum::vector:
                return detail::copy_if(
                    *policy.get<vector_execution_policy>(),
                    first, last, dest, std::forward<F>(f), category);

            case detail::execution_policy_enum::task:
                return detail::copy_if(par,
                    first, last, dest, std::forward<F>(f), category);

            default:
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "hpx::parallel::detail::copy_if",
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
    copy_if(ExPolicy&& policy, InIter first, InIter last, OutIter dest, F && f)
    {
        BOOST_STATIC_ASSERT_MSG(
            boost::is_base_of<std::input_iterator_tag,
                typename std::iterator_traits<InIter>::iterator_category>::value,
            "Required at least input iterator.");

        std::iterator_traits<InIter>::iterator_category category;
        return detail::copy_if(policy, first, last, dest,
            std::forward<F>(f), category);
    }
}}

#endif
