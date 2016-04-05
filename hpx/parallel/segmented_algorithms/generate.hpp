//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHM_GENERATE_JAN_24_2016_0749PM)
#define HPX_PARALLEL_SEGMENTED_ALGORITHM_GENERATE_JAN_24_2016_0749PM

#include <hpx/config.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/generate.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <list>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // segmented_generate
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F>
        static typename util::detail::algorithm_result<ExPolicy, SegIter>::type
        segmented_generate(Algo && algo, ExPolicy const& policy,
            SegIter first, SegIter last, F && f, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, SegIter> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    local_iterator_type out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f);
                    last = traits::compose(send, out);
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                local_iterator_type out = traits::local(last);

                if (beg != end)
                {
                    out = dispatch(traits::get_id(sit), algo, policy,
                        std::true_type(), beg, end, f);
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        out = dispatch(traits::get_id(sit), algo, policy,
                            std::true_type(), beg, end, f);
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    out = dispatch(traits::get_id(sit), algo, policy,
                        std::true_type(), beg, end, f);
                }

                last = traits::compose(send, out);
            }

            return result::get(std::move(last));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F>
        static typename util::detail::algorithm_result<ExPolicy, SegIter>::type
        segmented_generate(Algo && algo, ExPolicy const& policy,
            SegIter first, SegIter last, F && f, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;

            typedef std::integral_constant<bool,
                    !hpx::traits::is_forward_iterator<SegIter>::value
                > forced_seq;
            typedef util::detail::algorithm_result<ExPolicy, SegIter> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<shared_future<local_iterator_type> > segments;
            segments.reserve(std::distance(sit, send));

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(), beg, end, f));
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(), beg, end, f));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        segments.push_back(dispatch_async(traits::get_id(sit),
                            algo, policy, forced_seq(), beg, end, f));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(), beg, end, f));
                }
            }

            return result::get(
                dataflow(
                    [=](std::vector<hpx::shared_future<local_iterator_type> > && r)
                        ->  SegIter
                    {
                        // handle any remote exceptions, will throw on error
                        std::list<boost::exception_ptr> errors;
                        parallel::util::detail::handle_remote_exceptions<
                            ExPolicy
                        >::call(r, errors);
                        return traits::compose(send, r.back().get());
                    },
                    std::move(segments)));
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename FwdIter, typename F>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        generate_(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
            std::true_type)
        {
            typedef parallel::is_sequential_execution_policy<ExPolicy> is_seq;

            if (first == last)
            {
                typedef util::detail::algorithm_result<ExPolicy, FwdIter> result;
                return result::get(std::move(last));
            }

            typedef hpx::traits::segmented_iterator_traits<FwdIter>
                iterator_traits;

            return segmented_generate(
                generate<typename iterator_traits::local_iterator>(),
                std::forward<ExPolicy>(policy), first, last,
                std::forward<F>(f), is_seq());
        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename FwdIter, typename F>
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        generate_(ExPolicy&& policy, FwdIter first, FwdIter last, F && f,
            std::false_type);

        /// \endcond
    }
}}}

#endif
