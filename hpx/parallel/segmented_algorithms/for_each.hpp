//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHM_FOR_EACH_OCT_15_2014_0839PM)
#define HPX_PARALLEL_SEGMENTED_ALGORITHM_FOR_EACH_OCT_15_2014_0839PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>

#include <boost/mpl/bool.hpp>
#include <boost/type_traits/is_same.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // segmented_for_each
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F, typename Proj>
        static typename util::detail::algorithm_result<ExPolicy, SegIter>::type
        segmented_for_each(Algo && algo, ExPolicy const& policy,
            SegIter first, SegIter last, F && f, Proj && proj,
            boost::mpl::true_)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, SegIter> result;

            using boost::mpl::true_;

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
                        algo, policy, true_(), beg, end, f, proj
                    );
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
                    out = dispatch(traits::get_id(sit),
                        algo, policy, true_(), beg, end, f, proj
                    );
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    out = traits::begin(send);

                    if (beg != end)
                    {
                        out = dispatch(traits::get_id(sit),
                            algo, policy, true_(), beg, end, f, proj
                        );
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    out = dispatch(traits::get_id(sit),
                        algo, policy, true_(), beg, end, f, proj
                    );
                }

                last = traits::compose(send, out);
            }

            return result::get(std::move(last));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F, typename Proj>
        static typename util::detail::algorithm_result<ExPolicy, SegIter>::type
        segmented_for_each(Algo && algo, ExPolicy const& policy,
            SegIter first, SegIter last, F && f, Proj && proj,
            boost::mpl::false_)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, SegIter> result;

            typedef typename std::iterator_traits<SegIter>::iterator_category
                iterator_category;
            typedef typename boost::mpl::bool_<boost::is_same<
                    iterator_category, std::input_iterator_tag
                >::value> forced_seq;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<future<local_iterator_type> > segments;
            segments.reserve(std::distance(sit, send));

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(), beg, end, f, proj
                    ));
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(), beg, end, f, proj
                    ));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        segments.push_back(dispatch_async(traits::get_id(sit),
                            algo, policy, forced_seq(), beg, end, f, proj
                        ));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(), beg, end, f, proj
                    ));
                }
            }

            return result::get(
                dataflow(
                    [=](std::vector<hpx::future<local_iterator_type> > && r)
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
        template <typename ExPolicy, typename SegIter, typename F,
            typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy, SegIter>::type
        for_each_(ExPolicy && policy, SegIter first, SegIter last, F && f,
            Proj && proj, std::true_type)
        {
            typedef typename parallel::is_sequential_execution_policy<
                    ExPolicy
                >::type is_seq;

            if (first == last)
            {
                typedef util::detail::algorithm_result<ExPolicy, SegIter> result;
                return result::get(std::move(last));
            }

            typedef hpx::traits::segmented_iterator_traits<SegIter>
                iterator_traits;

            return segmented_for_each(
                for_each<typename iterator_traits::local_iterator>(),
                std::forward<ExPolicy>(policy), first, last,
                std::forward<F>(f), std::forward<Proj>(proj), is_seq());
        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename InIter, typename F,
            typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy, InIter>::type
        for_each_(ExPolicy && policy, InIter first, InIter last, F && f,
            Proj && proj, std::false_type);

        /// \endcond
    }
}}}

#endif
