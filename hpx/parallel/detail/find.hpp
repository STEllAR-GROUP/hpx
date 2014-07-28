//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/detail/find.hpp

#if !defined(HPX_PARALLEL_DETAIL_FIND_JULY_16_2014_0213PM)
#define HPX_PARALLEL_DETAIL_FIND_JULY_16_2014_0213PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/detail/algorithm_result.hpp>
#include <hpx/parallel/detail/dispatch.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>
#include <hpx/parallel/detail/transform_reduce.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // find
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename InIter>
        struct find : public detail::algorithm<find<InIter>, InIter>
        {
            find()
                : find::algorithm("find")
            {}

            template <typename ExPolicy, typename T>
            static InIter
            sequential(ExPolicy const&, InIter first, InIter last, const T& val)
            {
                return std::find(first, last, val);
            }

            template <typename ExPolicy, typename T>
            static typename detail::algorithm_result<ExPolicy, InIter>::type
            parallel(ExPolicy const& policy, InIter first, InIter last,
                T const& val)
            {
                typedef typename std::iterator_traits<InIter>::iterator_category
                    category;
                typedef typename std::iterator_traits<InIter>::value_type type;

                std::size_t count = std::distance(first, last);

                util::cancellation_token<std::size_t> tok(count);

                return util::partitioner<ExPolicy, InIter, void>::call_with_index(
                    policy, first, count,
                    [val, tok](std::size_t base_idx, InIter it,
                        std::size_t part_size) mutable
                    {
                        util::loop_idx_n(
                            base_idx, it, part_size, tok,
                            [&val, &tok](type& v, std::size_t i)
                            {
                                if (v == val)
                                    tok.cancel(i);
                            });
                    },
                    [=](std::vector<hpx::future<void> > &&) mutable
                    {
                        std::size_t find_res = tok.get_data();
                        if(find_res != count)
                            std::advance(first, find_res);
                        else
                            first = last;

                        return std::move(first);
                    });
            }
        };
        /// \endcond
    }

    template <typename ExPolicy, typename InIter, typename T>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, InIter>::type
    >::type
    find(ExPolicy && policy, InIter first, InIter last, T const& val)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::input_iterator_tag, iterator_category
            >::value),
            "Requires at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::find<InIter>().call(
            std::forward<ExPolicy>(policy),
            first, last, val, is_seq());
    }

    namespace detail 
    {
        /// \cond NOINTERNAL
        template <typename InIter>
        struct find_if : public detail::algorithm<find_if<InIter>, InIter>
        {
            find_if()
                : find_if::algorithm("find_if")
            {}

            template <typename ExPolicy, typename F>
            static InIter
            sequential(ExPolicy const&, InIter first, InIter last, F && f)
            {
                return std::find_if(first, last, f);
            }

            template <typename ExPolicy, typename FwdIter, typename F>
            static typename detail::algorithm_result<ExPolicy, FwdIter>::type
            parallel(ExPolicy const& policy, FwdIter first, FwdIter last, F && f)
            {
                typedef typename std::iterator_traits<FwdIter>::iterator_category
                    category;
                typedef typename std::iterator_traits<FwdIter>::value_type type;

                std::size_t count = std::distance(first, last);

                util::cancellation_token<std::size_t> tok(count);

                return util::partitioner<ExPolicy, FwdIter, void>::call_with_index(
                    policy, first, count,
                    [f, tok](std::size_t base_idx, FwdIter it,
                        std::size_t part_size) mutable
                {
                    util::loop_idx_n(
                        base_idx, it, part_size, tok,
                        [&f, &tok](type& v, std::size_t i)
                    {
                        if ( f(v) )
                            tok.cancel(i);
                    });
                },
                [=](std::vector<hpx::future<void> > &&) mutable
                {
                    std::size_t find_res = tok.get_data();
                    if(find_res != count)
                        std::advance(first, find_res);
                    else
                        first = last;

                    return std::move(first);
                });
            }
        };
        /// \endcond
    }

    template <typename ExPolicy, typename InIter, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, InIter>::type
    >::type
    find_if(ExPolicy && policy, InIter first, InIter last, F && f)
    {

        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::input_iterator_tag, iterator_category
            >::value),
            "Requires at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::find_if<InIter>().call(
            std::forward<ExPolicy>(policy),
            first, last, std::forward<F>(f), is_seq());
    } 

    namespace detail {
        /// \cond NOINTERNAL
        template <typename InIter>
        struct find_if_not : public detail::algorithm<find_if_not<InIter>, InIter>
        {
            find_if_not()
                : find_if_not::algorithm("find_if_not")
            {}

            template <typename ExPolicy, typename F>
            static InIter
            sequential(ExPolicy const&, InIter first, InIter last, F && f)
            {
                for (; first != last; ++first) {
                    if (!f(*first)) {
                        return first;
                    }
                }
                return last;
            }

            template <typename ExPolicy, typename FwdIter, typename F>
            static typename detail::algorithm_result<ExPolicy, FwdIter>::type
            parallel(ExPolicy const& policy, FwdIter first, FwdIter last, F && f)
            {
                typedef typename std::iterator_traits<FwdIter>::iterator_category
                    category;
                typedef typename std::iterator_traits<FwdIter>::value_type type;

                std::size_t count = std::distance(first, last);

                util::cancellation_token<std::size_t> tok(count);

                return util::partitioner<ExPolicy, FwdIter, void>::call_with_index(
                    policy, first, count,
                    [f, tok](std::size_t base_idx, FwdIter it, std::size_t part_size) mutable
                {
                    util::loop_idx_n(
                        base_idx, it, part_size, tok,
                        [&f, &tok](type& v, std::size_t i)
                    {
                        if ( !f(v) )
                            tok.cancel(i);
                    });

                },
                [=](std::vector<hpx::future<void> > &&) mutable
                {
                    std::size_t find_res = tok.get_data();
                    if(find_res != count)
                        std::advance(first, find_res);
                    else
                        first = last;
                    
                    return std::move(first);
                });
            }

        };    
    }

    template <typename ExPolicy, typename InIter, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, InIter>::type
    >::type
    find_if_not(ExPolicy && policy, InIter first, InIter last, F && f)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::input_iterator_tag, iterator_category
            >::value),
            "Requires at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::find_if_not<InIter>().call(
            std::forward<ExPolicy>(policy),
            first, last, std::forward<F>(f), is_seq());
    }

    namespace detail {
        /// \cond NOINTERNAL
        template <typename FwdIter>
        struct find_end : public detail::algorithm<find_end<FwdIter>, FwdIter>
        {
            find_end()
                : find_end::algorithm("find_end")
            {}

            template <typename ExPolicy, typename FwdIter2>
            static FwdIter
            sequential(ExPolicy const&, FwdIter first1, FwdIter last1, 
                FwdIter2 first2, FwdIter2 last2)
            {
                return
                    std::find_end(first1, last1, first2, last2);
            }

            template <typename ExPolicy, typename FwdIter2>
            static typename detail::algorithm_result<ExPolicy, FwdIter>::type
            parallel(ExPolicy const& policy, FwdIter first1, FwdIter last1, 
                FwdIter2 first2, FwdIter2 last2)
            {
                std::size_t count = std::distance(first1, last1);
                std::size_t diff = std::distance(first2, last2);

                typedef typename std::iterator_traits<FwdIter>::iterator_category
                    category;
                typedef typename std::iterator_traits<FwdIter>::reference value;
                typedef typename std::iterator_traits<FwdIter>::difference_type diff_type;

                util::cancellation_token<diff_type, std::greater_equal<diff_type>> tok(-1);

                return util::partitioner<ExPolicy, FwdIter, void>::call_with_index(
                    policy, first1, count-(diff-1),
                    [tok,first2,diff,count]( std::size_t base_idx, FwdIter it,
                        std::size_t part_count) mutable
                {
                    util::loop_idx_n(
                        base_idx, it, part_count, tok,
                        [&tok,first2,diff,count,it,base_idx](value t, std::size_t i)
                        {
                            if(t == *first2) {
                                diff_type local_count = 1;
                                FwdIter2 needle = first2;
                                FwdIter mid = next(it,i-base_idx);
                                for(; std::size_t(local_count) < diff; ++local_count) {
                                    ++needle;
                                    ++mid;

                                    if( *mid != *needle )
                                        break;
                                }
                                
                                if(local_count == diff) {
                                    tok.cancel(i);
                                }
                            }

                        });
                },
                [=](std::vector<hpx::future<void> > &&) mutable
                {
                    std::size_t find_end_res = tok.get_data();
                    if(find_end_res != count) {
                        std::advance(first1, find_end_res);
                    }else{
                        first1 = last1;
                    }
                    return std::move(first1);
                });
            }
        };
        /// \endcond
    }

    template <typename ExPolicy, typename FwdIter1, typename FwdIter2>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, FwdIter1>::type
    >::type
    find_end(ExPolicy && policy, FwdIter1 first1, FwdIter1 last1, FwdIter2 first2,
        FwdIter2 last2)
    {
        typedef typename std::iterator_traits<FwdIter1>::iterator_category
            iterator_category1;

        typedef typename std::iterator_traits<FwdIter2>::iterator_category
            iterator_category2;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::forward_iterator_tag, iterator_category1
            >::value),
            "Requires at least forward iterator.");

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::forward_iterator_tag, iterator_category2
            >::value),
            "Requires at least forward iterator.");

        typedef is_sequential_execution_policy<ExPolicy> is_seq;

        return detail::find_end<FwdIter1>().call(
            std::forward<ExPolicy>(policy),
            first1, last1, first2, last2, is_seq());
    }
}}}

#endif
