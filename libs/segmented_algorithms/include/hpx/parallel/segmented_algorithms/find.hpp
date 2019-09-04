//  Copyright (c) 2017 Ajai V George
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHM_FIND_JUN_22_2017_1157AM)
#define HPX_PARALLEL_SEGMENTED_ALGORITHM_FIND_JUN_22_2017_1157AM

#include <hpx/config.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/find.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <exception>
#include <iterator>
#include <list>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1
{
    ///////////////////////////////////////////////////////////////////////////
    // segmented_find
    namespace detail
    {
        template <typename Algo, typename ExPolicy, typename FwdIter,
            typename U>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        segmented_find(Algo && algo, ExPolicy && policy, FwdIter first,
            FwdIter last, U && f_or_val, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, FwdIter> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);
            FwdIter output = last;
            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    local_iterator_type out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f_or_val
                    );
                    output=traits::compose(send,out);
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                local_iterator_type out = traits::local(last);
                bool found = false;
                if (beg != end)
                {
                    out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f_or_val
                    );
                    if(out != end)
                        found = true;
                }
                if(!found)
                {
                    // handle all of the full partitions
                    for (++sit; sit != send; ++sit)
                    {
                        beg = traits::begin(sit);
                        end = traits::end(sit);
                        out = traits::begin(send);
                        if (beg != end)
                        {
                            out = dispatch(traits::get_id(sit),
                                algo, policy, std::true_type(), beg, end, f_or_val
                            );
                            if(out != end)
                            {
                                found = true;
                                break;
                            }
                        }
                    }
                }
                if(!found)
                {
                    // handle the beginning of the last partition
                    beg = traits::begin(sit);
                    end = traits::local(last);
                    if (beg != end)
                    {
                        out = dispatch(traits::get_id(sit),
                            algo, policy, std::true_type(), beg, end, f_or_val
                        );
                        if(out != end)
                            found = true;
                    }
                }
                if(found)
                    output=traits::compose(sit,out);
            }
            return result::get(std::move(output));
        }

        template <typename Algo, typename ExPolicy, typename FwdIter,
            typename U>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        segmented_find(Algo && algo, ExPolicy && policy, FwdIter first,
            FwdIter last, U && f_or_val, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, FwdIter> result;

            typedef std::integral_constant<bool,
                    !hpx::traits::is_forward_iterator<FwdIter>::value
                > forced_seq;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<future<FwdIter> > segments;
            segments.reserve(std::distance(sit, send));

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(
                        hpx::make_future<FwdIter>(
                            dispatch_async(traits::get_id(sit), algo,
                                policy, forced_seq(), beg, end, f_or_val),
                            [send,end,last](local_iterator_type const& out)
                                -> FwdIter
                            {
                                if(out != end)
                                    return traits::compose(send, out);
                                else
                                    return last;
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
                        hpx::make_future<FwdIter>(
                            dispatch_async(traits::get_id(sit), algo,
                                policy, forced_seq(), beg, end, f_or_val),
                            [sit,end,last](local_iterator_type const& out)
                                -> FwdIter
                            {
                                if(out != end)
                                    return traits::compose(sit, out);
                                else
                                    return last;
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
                            hpx::make_future<FwdIter>(
                                dispatch_async(traits::get_id(sit), algo,
                                    policy, forced_seq(), beg, end, f_or_val),
                                [sit,end,last](local_iterator_type const& out)
                                    -> FwdIter
                                {
                                    if(out != end)
                                        return traits::compose(sit, out);
                                    else
                                        return last;
                                }));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(
                        hpx::make_future<FwdIter>(
                            dispatch_async(traits::get_id(sit), algo,
                                policy, forced_seq(), beg, end, f_or_val),
                            [sit,end,last](local_iterator_type const& out)
                                -> FwdIter
                            {
                                if(out != end)
                                    return traits::compose(sit, out);
                                else
                                    return last;
                            }));
                }
            }
            return result::get(
                dataflow(
                    [=](std::vector<hpx::future<FwdIter> > && r)
                        ->  FwdIter
                    {
                        // handle any remote exceptions, will throw on error
                        std::list<std::exception_ptr> errors;
                        parallel::util::detail::handle_remote_exceptions<
                            ExPolicy
                        >::call(r, errors);

                        std::vector<FwdIter> res =
                            hpx::util::unwrap(std::move(r));
                        auto it = res.begin();
                        while(it!=res.end())
                        {
                            if(*it != last)
                                return *it;
                            it++;
                        }
                        return res.back();
                    },
                    std::move(segments)));
        }

        template <typename ExPolicy, typename FwdIter, typename T>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        find_(ExPolicy && policy, FwdIter first, FwdIter last, T const& val,
            std::true_type)
        {
            typedef parallel::execution::is_sequenced_execution_policy<
                    ExPolicy
                > is_seq;

            if (first == last)
            {
                return util::detail::algorithm_result<
                        ExPolicy, FwdIter
                    >::get(std::forward<FwdIter>(first));
            }
            typedef hpx::traits::segmented_iterator_traits<FwdIter>
                iterator_traits;
            return segmented_find(
                find<typename iterator_traits::local_iterator>(),
                std::forward<ExPolicy>(policy), first, last,
                std::move(val),is_seq());
        }

        template <typename ExPolicy, typename FwdIter, typename T>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        find_(ExPolicy && policy, FwdIter first, FwdIter last, T const& val,
            std::false_type);

        template <typename ExPolicy, typename FwdIter, typename F>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        find_if_(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
            std::true_type)
        {
            typedef parallel::execution::is_sequenced_execution_policy<
                    ExPolicy
                > is_seq;

            if (first == last)
            {
                return util::detail::algorithm_result<
                        ExPolicy, FwdIter
                    >::get(std::forward<FwdIter>(first));
            }
            typedef typename std::iterator_traits<FwdIter>::value_type type;
            typedef hpx::traits::segmented_iterator_traits<FwdIter>
                iterator_traits;
            return segmented_find(
                find_if<typename iterator_traits::local_iterator>(),
                std::forward<ExPolicy>(policy), first, last,
                std::forward<F>(f),is_seq());
        }

        template <typename ExPolicy, typename FwdIter, typename F>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        find_if_(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
            std::false_type);

        template <typename ExPolicy, typename FwdIter, typename F>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        find_if_not_(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
            std::true_type)
        {
            typedef parallel::execution::is_sequenced_execution_policy<
                    ExPolicy
                > is_seq;

            if (first == last)
            {
                return util::detail::algorithm_result<
                        ExPolicy, FwdIter
                    >::get(std::forward<FwdIter>(first));
            }
            typedef typename std::iterator_traits<FwdIter>::value_type type;
            typedef hpx::traits::segmented_iterator_traits<FwdIter>
                iterator_traits;
            return segmented_find(
                find_if_not<typename iterator_traits::local_iterator>(),
                std::forward<ExPolicy>(policy), first, last,
                std::forward<F>(f),is_seq());
        }

        template <typename ExPolicy, typename FwdIter, typename F>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        find_if_not_(ExPolicy && policy, FwdIter first, FwdIter last, F && f,
            std::false_type);
    }
}}}
#endif
