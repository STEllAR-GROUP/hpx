//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHM_MINMAX_JAN_24_2016_0800PM)
#define HPX_PARALLEL_SEGMENTED_ALGORITHM_MINMAX_JAN_24_2016_0800PM

#include <hpx/config.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/minmax.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <list>
#include <vector>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // segmented_minmax
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F, typename Proj>
        static typename util::detail::algorithm_result<ExPolicy, SegIter>::type
        segmented_minormax(Algo && algo, ExPolicy const& policy,
            SegIter first, SegIter last, F && f, Proj && proj, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<SegIter> positions;

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    local_iterator_type out = dispatch(
                        traits::get_id(sit), algo, policy, std::true_type(),
                        beg, end, f, proj);

                    positions.push_back(traits::compose(send, out));
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);

                if (beg != end)
                {
                    local_iterator_type out = dispatch(
                        traits::get_id(sit), algo, policy, std::true_type(),
                        beg, end, f, proj);

                    positions.push_back(traits::compose(sit, out));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);

                    if (beg != end)
                    {
                        local_iterator_type out = dispatch(
                            traits::get_id(sit), algo, policy, std::true_type(),
                            beg, end, f, proj);

                        positions.push_back(traits::compose(sit, out));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    local_iterator_type out = dispatch(
                        traits::get_id(sit), algo, policy, std::true_type(),
                        beg, end, f, proj);

                    positions.push_back(traits::compose(sit, out));
                }
            }

            return Algo::sequential_minmax_element_ind(
                positions.begin(), positions.size(), f, proj);
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F, typename Proj>
        static typename util::detail::algorithm_result<ExPolicy, SegIter>::type
        segmented_minormax(Algo && algo, ExPolicy const& policy,
            SegIter first, SegIter last, F && f, Proj && proj, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;

            typedef hpx::traits::is_input_iterator<SegIter> forced_seq;
            typedef util::detail::algorithm_result<ExPolicy, SegIter> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<future<SegIter> > segments;
            segments.reserve(std::distance(sit, send));

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(
                        hpx::make_future<SegIter>(
                            dispatch_async(traits::get_id(sit), algo,
                                policy, forced_seq(), beg, end, f, proj),
                            [send](local_iterator_type const& out)
                                -> SegIter
                            {
                                return traits::compose(send, out);
                            }));
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    segments.push_back(
                        hpx::make_future<SegIter>(
                            dispatch_async(traits::get_id(sit), algo,
                                policy, forced_seq(), beg, end, f, proj),
                            [sit](local_iterator_type const& out)
                                -> SegIter
                            {
                                return traits::compose(sit, out);
                            }));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        segments.push_back(
                            hpx::make_future<SegIter>(
                                dispatch_async(traits::get_id(sit), algo,
                                    policy, forced_seq(), beg, end, f, proj),
                                [sit](local_iterator_type const& out)
                                    -> SegIter
                                {
                                    return traits::compose(sit, out);
                                }));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(
                        hpx::make_future<SegIter>(
                            dispatch_async(traits::get_id(sit), algo,
                                policy, forced_seq(), beg, end, f, proj),
                            [sit](local_iterator_type const& out)
                                -> SegIter
                            {
                                return traits::compose(sit, out);
                            }));
                }
            }

            return result::get(
                dataflow(
                    [=](std::vector<hpx::future<SegIter> > && r)
                        ->  SegIter
                    {
                        // handle any remote exceptions, will throw on error
                        std::list<boost::exception_ptr> errors;
                        parallel::util::detail::handle_remote_exceptions<
                            ExPolicy
                        >::call(r, errors);

                        std::vector<SegIter> res =
                            hpx::util::unwrapped(std::move(r));
                        return Algo::sequential_minmax_element_ind(
                            res.begin(), res.size(), f, proj);
                    },
                    std::move(segments)));
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename SegIter, typename F, typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy, SegIter>::type
        min_element_(ExPolicy && policy, SegIter first, SegIter last, F && f,
            Proj && proj, std::true_type)
        {
            typedef parallel::is_sequential_execution_policy<ExPolicy> is_seq;

            SegIter result = first;
            if (first == last || ++first == last)
            {
                return util::detail::algorithm_result<
                        ExPolicy, SegIter
                    >::get(std::move(result));
            }

             typedef hpx::traits::segmented_iterator_traits<SegIter>
                iterator_traits;

           return segmented_minormax(
                min_element<typename iterator_traits::local_iterator>(),
                std::forward<ExPolicy>(policy), first, last,
                std::forward<F>(f), std::forward<Proj>(proj), is_seq());
        }

        template <typename ExPolicy, typename SegIter, typename F, typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy, SegIter>::type
        max_element_(ExPolicy && policy, SegIter first, SegIter last, F && f,
            Proj && proj, std::true_type)
        {
            typedef parallel::is_sequential_execution_policy<ExPolicy> is_seq;

            SegIter result = first;
            if (first == last || ++first == last)
            {
                return util::detail::algorithm_result<
                        ExPolicy, SegIter
                    >::get(std::move(result));
            }

             typedef hpx::traits::segmented_iterator_traits<SegIter>
                iterator_traits;

           return segmented_minormax(
                max_element<typename iterator_traits::local_iterator>(),
                std::forward<ExPolicy>(policy), first, last,
                std::forward<F>(f), std::forward<Proj>(proj), is_seq());
        }

        // forward declare the non-segmented version of those algorithm
        template <typename ExPolicy, typename FwdIter, typename F, typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        min_element_(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
            Proj && proj, std::false_type);

        template <typename ExPolicy, typename FwdIter, typename F, typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        max_element_(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
            Proj && proj, std::false_type);

        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    // segmented_minmax
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F, typename Proj>
        static typename util::detail::algorithm_result<
            ExPolicy, std::pair<SegIter, SegIter>
        >::type
        segmented_minmax(Algo && algo, ExPolicy const& policy,
            SegIter first, SegIter last, F && f, Proj && proj, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef std::pair<SegIter, SegIter> result_type;

            typedef std::pair<
                    local_iterator_type, local_iterator_type
                > local_iterator_pair_type;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<result_type> positions;

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    local_iterator_pair_type out = dispatch(
                        traits::get_id(sit), algo, policy, std::true_type(),
                        beg, end, f, proj);

                    positions.push_back(std::make_pair(
                        traits::compose(send, out.first),
                        traits::compose(send, out.second)));
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);

                if (beg != end)
                {
                    local_iterator_pair_type out = dispatch(
                        traits::get_id(sit), algo, policy, std::true_type(),
                        beg, end, f, proj);

                    positions.push_back(std::make_pair(
                        traits::compose(sit, out.first),
                        traits::compose(sit, out.second)));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);

                    if (beg != end)
                    {
                        local_iterator_pair_type out = dispatch(
                            traits::get_id(sit), algo, policy, std::true_type(),
                            beg, end, f, proj);

                        positions.push_back(std::make_pair(
                            traits::compose(sit, out.first),
                            traits::compose(sit, out.second)));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    local_iterator_pair_type out = dispatch(
                        traits::get_id(sit), algo, policy, std::true_type(),
                        beg, end, f, proj);

                    positions.push_back(std::make_pair(
                        traits::compose(sit, out.first),
                        traits::compose(sit, out.second)));
                }
            }

            return sequential_minmax_element_ind(
                positions.begin(), positions.size(), f, proj);
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename SegIter,
            typename F, typename Proj>
        static typename util::detail::algorithm_result<
            ExPolicy, std::pair<SegIter, SegIter>
        >::type
        segmented_minmax(Algo && algo, ExPolicy const& policy,
            SegIter first, SegIter last, F && f, Proj && proj, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<SegIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;

            typedef hpx::traits::is_input_iterator<SegIter> forced_seq;
            typedef std::pair<SegIter, SegIter> result_type;
            typedef util::detail::algorithm_result<ExPolicy, result_type> result;

            typedef std::pair<
                    local_iterator_type, local_iterator_type
                > local_iterator_pair_type;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<future<result_type> > segments;
            segments.reserve(std::distance(sit, send));

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(
                        hpx::make_future<result_type>(
                            dispatch_async(traits::get_id(sit), algo,
                                policy, forced_seq(), beg, end, f, proj),
                            [send](local_iterator_pair_type out)
                                -> result_type
                            {
                                return std::make_pair(
                                    traits::compose(send, out.first),
                                    traits::compose(send, out.second));
                            }));
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    segments.push_back(
                        hpx::make_future<result_type>(
                            dispatch_async(traits::get_id(sit), algo,
                                policy, forced_seq(), beg, end, f, proj),
                            [sit](local_iterator_pair_type const& out)
                                -> result_type
                            {
                                return std::make_pair(
                                    traits::compose(sit, out.first),
                                    traits::compose(sit, out.second));
                            }));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        segments.push_back(
                            hpx::make_future<result_type>(
                                dispatch_async(traits::get_id(sit), algo,
                                    policy, forced_seq(), beg, end, f, proj),
                                [sit](local_iterator_pair_type const& out)
                                    -> result_type
                                {
                                    return std::make_pair(
                                        traits::compose(sit, out.first),
                                        traits::compose(sit, out.second));
                                }));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(
                        hpx::make_future<result_type>(
                            dispatch_async(traits::get_id(sit), algo,
                                policy, forced_seq(), beg, end, f, proj),
                            [sit](local_iterator_pair_type const& out)
                                -> result_type
                            {
                                return std::make_pair(
                                    traits::compose(sit, out.first),
                                    traits::compose(sit, out.second));
                            }));
                }
            }

            return result::get(
                dataflow(
                    [=](std::vector<hpx::future<result_type> > && r)
                        ->  result_type
                    {
                        // handle any remote exceptions, will throw on error
                        std::list<boost::exception_ptr> errors;
                        parallel::util::detail::handle_remote_exceptions<
                            ExPolicy
                        >::call(r, errors);

                        std::vector<result_type> res =
                            hpx::util::unwrapped(std::move(r));
                        return sequential_minmax_element_ind(
                            res.begin(), res.size(), f, proj);
                    },
                    std::move(segments)));
        }

        ///////////////////////////////////////////////////////////////////////
        // segmented implementation
        template <typename ExPolicy, typename SegIter, typename F, typename Proj>
        inline typename util::detail::algorithm_result<
            ExPolicy, std::pair<SegIter, SegIter>
        >::type
        minmax_element_(ExPolicy && policy, SegIter first, SegIter last, F && f,
            Proj && proj, std::true_type)
        {
            typedef parallel::is_sequential_execution_policy<ExPolicy> is_seq;
            typedef std::pair<SegIter, SegIter> result_type;

            result_type result(first, first);
            if (first == last || ++first == last)
            {
                return util::detail::algorithm_result<
                        ExPolicy, result_type
                    >::get(std::move(result));
            }

             typedef hpx::traits::segmented_iterator_traits<SegIter>
                iterator_traits;

           return segmented_minmax(
                minmax_element<typename iterator_traits::local_iterator>(),
                std::forward<ExPolicy>(policy), first, last,
                std::forward<F>(f), std::forward<Proj>(proj), is_seq());
        }

        // forward declare the non-segmented version of this algorithm
        template <typename ExPolicy, typename FwdIter, typename F, typename Proj>
        inline typename util::detail::algorithm_result<
            ExPolicy, std::pair<FwdIter, FwdIter>
        >::type
        minmax_element_(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
            Proj && proj, std::false_type);

        /// \endcond
    }
}}}

#endif
