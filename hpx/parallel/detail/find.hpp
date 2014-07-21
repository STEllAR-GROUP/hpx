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
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // find
    namespace detail {
        /// \cond NOINTERNAL
        template <typename InIter>
        struct find : public detail::algorithm<find<InIter>, InIter>
        {
            find()
                : detail::algorithm<find<InIter>, InIter>("find")
            {}

            template <typename ExPolicy, typename T>
            static typename detail::algorithm_result<ExPolicy, InIter>::type
            sequential(ExPolicy const&, InIter first, InIter last, const T& val)
            {
                return detail::algorithm_result<ExPolicy, InIter>::get(
                    std::find(first, last, val));
            }

            template <typename ExPolicy, typename T>
            static typename detail::algorithm_result<ExPolicy, InIter>::type
            parallel(ExPolicy const& policy, InIter first, InIter last,
                const T& val)
            {
                typedef typename std::iterator_traits<InIter>::iterator_category
                    category;
                typedef typename std::iterator_traits<InIter>::value_type type;

                std::size_t count = std::distance(first, last);

                util::cancellation_token<std::size_t> tok(count);

                util::partitioner<ExPolicy>::call(policy,
                    first, count,
                    [val, &tok, first](InIter it, std::size_t part_count) mutable
                    {
                        std::size_t base_idx =
                            std::distance(first, it);

                        util::loop_idx_n(
                            base_idx, it, part_count, tok,
                            [val, &tok](type& v, std::size_t i)
                            {
                                if(v == val){
                                    tok.cancel(i);
                                }
                            });
                    });

                std::size_t findres = tok.get_data();
                if(findres != count)
                    std::advance(first, findres);
                else
                    first = last;

                std::cout << *first;
                return first;
                //return detail::algorithm_result<ExPolicy, InIter>::get(std::move(res));
            }
        };
        /// \endcond
    }

        template <typename ExPolicy, typename InIter, typename T>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, InIter>::type
    >::type
    find(ExPolicy && policy, InIter first, InIter last, const T& val)
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
            first, last, val,
            is_seq());
    }
}}}

#endif