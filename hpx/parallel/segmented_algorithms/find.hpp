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

#include <boost/exception_ptr.hpp>

#include <algorithm>
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
        template <typename Algo, typename ExPolicy, typename InIter,
            typename T, typename F = util::projection_identity>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, InIter>::type
        >::type
        segmented_find(Algo && algo, ExPolicy && policy, InIter first,
            InIter last, T const& val, bool flag, std::true_type,
            F && f = F())
        {
            typedef hpx::traits::segmented_iterator_traits<InIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, InIter> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                local_iterator_type out;
                if (beg != end)
                {
                    if(!flag)
                        out = dispatch(traits::get_id(sit),
                            algo, policy, std::true_type(), beg, end, val
                        );
                    else
                        out = dispatch(traits::get_id(sit),
                            algo, policy, std::true_type(), beg, end, f
                        );
                    last = traits::compose(send, out);
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                local_iterator_type out;

                if (beg != end)
                {
                    if(!flag)
                        out = dispatch(traits::get_id(sit),
                            algo, policy, std::true_type(), beg, end, val
                        );
                    else
                        out = dispatch(traits::get_id(sit),
                            algo, policy, std::true_type(), beg, end, f
                        );
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);

                    if (beg != end)
                    {
                        if(!flag)
                            out = dispatch(traits::get_id(sit),
                                algo, policy, std::true_type(), beg, end, val
                            );
                        else
                            out = dispatch(traits::get_id(sit),
                                algo, policy, std::true_type(), beg, end, f
                            );
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    if(!flag)
                        out = dispatch(traits::get_id(sit),
                            algo, policy, std::true_type(), beg, end, val
                        );
                    else
                        out = dispatch(traits::get_id(sit),
                            algo, policy, std::true_type(), beg, end, f
                        );
                }

                last = traits::compose(send, out);
            }
            return result::get(std::move(last));
        }

        // template <typename Algo, typename ExPolicy, typename InIter,
        //     typename T, typename F>
        // inline typename std::enable_if<
        //     execution::is_execution_policy<ExPolicy>::value,
        //     typename util::detail::algorithm_result<ExPolicy, InIter>::type
        // >::type
        // segmented_find(Algo && algo, ExPolicy && policy, InIter first,
        //     InIter last, T const& val, bool flag1, bool flag2, std::false_type,
        //     F && f = F())
        // {
        //     typedef hpx::traits::segmented_iterator_traits<InIter> traits;
        //     typedef typename traits::segment_iterator segment_iterator;
        //     typedef typename traits::local_iterator local_iterator_type;
        //     typedef util::detail::algorithm_result<ExPolicy, InIter> result;
        //
        //     segment_iterator sit = traits::segment(first);
        //     segment_iterator send = traits::segment(last);
        //
        //     std::vector<future<local_iterator_type> > segments;
        //     segments.reserve(std::distance(sit, send));
        //
        //     if (sit == send)
        //     {
        //     }
        //     else
        //     {
        //
        //     }
        //     return result::get(std::move(output));
        // }

        template <typename ExPolicy, typename InIter, typename T>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, InIter>::type
        >::type
        find_(ExPolicy && policy, InIter first, InIter last, T const& val,
            std::true_type)
        {
            typedef parallel::execution::is_sequenced_execution_policy<
                    ExPolicy
                > is_seq;

            if (first == last)
            {
                return util::detail::algorithm_result<
                        ExPolicy, InIter
                    >::get(std::forward<InIter>(first));
            }
            typedef hpx::traits::segmented_iterator_traits<InIter>
                iterator_traits;
            return segmented_find(
                find<typename iterator_traits::local_iterator>(),
                std::forward<ExPolicy>(policy), first, last,
                std::move(val),false,is_seq());
        }

        template <typename ExPolicy, typename InIter, typename T>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, InIter>::type
        >::type
        find_(ExPolicy && policy, InIter first, InIter last, T const& val,
            std::false_type);

        template <typename ExPolicy, typename InIter, typename F>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, InIter>::type
        >::type
        find_if_(ExPolicy && policy, InIter first, InIter last, F && f,
            std::true_type)
        {
            typedef parallel::execution::is_sequenced_execution_policy<
                    ExPolicy
                > is_seq;

            if (first == last)
            {
                return util::detail::algorithm_result<
                        ExPolicy, InIter
                    >::get(std::forward<InIter>(first));
            }
            typedef typename std::iterator_traits<InIter>::value_type type;
            typedef hpx::traits::segmented_iterator_traits<InIter>
                iterator_traits;
            return segmented_find(
                find_if<typename iterator_traits::local_iterator>(),
                std::forward<ExPolicy>(policy), first, last,
                std::move(type(0)),true,is_seq(),f);
        }

        template <typename ExPolicy, typename InIter, typename F>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, InIter>::type
        >::type
        find_if_(ExPolicy && policy, InIter first, InIter last, F && f,
            std::false_type);

        template <typename ExPolicy, typename InIter, typename F>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, InIter>::type
        >::type
        find_if_not_(ExPolicy && policy, InIter first, InIter last, F && f,
            std::true_type)
        {
            typedef parallel::execution::is_sequenced_execution_policy<
                    ExPolicy
                > is_seq;

            if (first == last)
            {
                return util::detail::algorithm_result<
                        ExPolicy, InIter
                    >::get(std::forward<InIter>(first));
            }
            typedef typename std::iterator_traits<InIter>::value_type type;
            typedef hpx::traits::segmented_iterator_traits<InIter>
                iterator_traits;
            return segmented_find(
                find_if_not<typename iterator_traits::local_iterator>(),
                std::forward<ExPolicy>(policy), first, last,
                std::move(type(0)),true,is_seq(),f);
        }

        template <typename ExPolicy, typename InIter, typename F>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, InIter>::type
        >::type
        find_if_not_(ExPolicy && policy, InIter first, InIter last, F && f,
            std::false_type);
    }
}}}
#endif
