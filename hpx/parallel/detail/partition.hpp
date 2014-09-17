//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/detail/partition.hpp

#if !defined(HPX_PARALLEL_DETAIL_PARTITION_JUL_28_2014_0752PM)
#define HPX_PARALLEL_DETAIL_PARTITION_JUL_28_2014_0752PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/detail/algorithm_result.hpp>
#include <hpx/parallel/detail/dispatch.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // partition
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename FwdIter>
        struct partition
          : public detail::algorithm<partition<FwdIter>, FwdIter>
        {
            partition()
              : partition::algorithm("partition")
            {}

            template <typename ExPolicy, typename F>
            static FwdIter
            sequential(ExPolicy const&, FwdIter first, FwdIter last, F && f)
            {
                return std::partition(first, last, std::forward<F>(f));
            }

            template <typename ExPolicy, typename F>
            static typename detail::algorithm_result<ExPolicy, FwdIter>::type
            parallel(ExPolicy const& policy, FwdIter first, FwdIter last,
                F && f)
            {

                return detail::algorithm_result<ExPolicy, FwdIter>::get(
                    std::move(first));
            }
        };
        /// \endcond
    }

    template <typename ExPolicy, typename FwdIter, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, FwdIter>::type
    >::type
    partition(ExPolicy && policy, FwdIter first, FwdIter last, F && f)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<std::forward_iterator_tag, iterator_category>::value),
            "Requires at least forward iterator.");

        // FIXME: currently forward iterators cause sequential execution
        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::forward_iterator_tag, iterator_category>
        >::type is_seq;

        return details::partition().call(
            std::forward<ExPolicy>(policy),
            first, last, std::forward<F>(f), is_seq());
    }
}}}

#endif

