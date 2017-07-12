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
#include <hpx/parallel/segmented_algorithms/detail/find_end_return.hpp>
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
            typename U>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, InIter>::type
        >::type
        segmented_find(Algo && algo, ExPolicy && policy, InIter first,
            InIter last, U && f_or_val, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<InIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, InIter> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);
            InIter output = first;
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

                if (beg != end)
                {
                    out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f_or_val
                    );
                    if(out != end)
                        output=traits::compose(sit,out);
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
                            algo, policy, std::true_type(), beg, end, f_or_val
                        );
                        if(out != end)
                            output=traits::compose(sit,out);
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f_or_val
                    );
                    if(out != end)
                        output=traits::compose(sit,out);
                }
            }
            return result::get(std::move(output));
        }

        template <typename Algo, typename ExPolicy, typename InIter,
            typename U>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, InIter>::type
        >::type
        segmented_find(Algo && algo, ExPolicy && policy, InIter first,
            InIter last, U && f_or_val, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<InIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, InIter> result;

            typedef std::integral_constant<bool,
                    !hpx::traits::is_forward_iterator<InIter>::value
                > forced_seq;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<future<InIter> > segments;
            segments.reserve(std::distance(sit, send));

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(
                        hpx::make_future<InIter>(
                            dispatch_async(traits::get_id(sit), algo,
                                policy, forced_seq(), beg, end, f_or_val),
                            [send,end,last](local_iterator_type const& out)
                                -> InIter
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
                        hpx::make_future<InIter>(
                            dispatch_async(traits::get_id(sit), algo,
                                policy, forced_seq(), beg, end, f_or_val),
                            [sit,end,last](local_iterator_type const& out)
                                -> InIter
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
                            hpx::make_future<InIter>(
                                dispatch_async(traits::get_id(sit), algo,
                                    policy, forced_seq(), beg, end, f_or_val),
                                [sit,end,last](local_iterator_type const& out)
                                    -> InIter
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
                        hpx::make_future<InIter>(
                            dispatch_async(traits::get_id(sit), algo,
                                policy, forced_seq(), beg, end, f_or_val),
                            [sit,end,last](local_iterator_type const& out)
                                -> InIter
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
                    [=](std::vector<hpx::future<InIter> > && r)
                        ->  InIter
                    {
                        // handle any remote exceptions, will throw on error
                        std::list<std::exception_ptr> errors;
                        parallel::util::detail::handle_remote_exceptions<
                            ExPolicy
                        >::call(r, errors);

                        std::vector<InIter> res =
                            hpx::util::unwrapped(std::move(r));
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
                std::move(val),is_seq());
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
                std::forward<F>(f),is_seq());
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
                std::forward<F>(f),is_seq());
        }

        template <typename ExPolicy, typename InIter, typename F>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, InIter>::type
        >::type
        find_if_not_(ExPolicy && policy, InIter first, InIter last, F && f,
            std::false_type);

        /// \cond NOINTERNAL
        template<typename Iter>
        struct seg_find_end : public detail::algorithm<seg_find_end<Iter>, find_end_return<Iter> >
        {
            seg_find_end()
              : seg_find_end<Iter>::algorithm("segmented_find_end")
            {}

            template <typename ExPolicy, typename FwdIter1, typename SeqVec,
                typename Pred>
            static find_end_return<FwdIter1>
            sequential(ExPolicy, FwdIter1 first1, FwdIter1 last1,
                SeqVec sequence, Pred && op,
                FwdIter1 start, unsigned int g_cursor = 0)
            {
                unsigned int cursor = g_cursor;
                while(last1 != first1)
                {
                    if(op(*first1, sequence[cursor]))
                    {
                        if(cursor == 0)
                            start = first1;
                        cursor++;
                        if(cursor == sequence.size())
                        {
                            find_end_return<FwdIter1> ret = {start,
                                (unsigned int) sequence.size()};
                            return ret;
                        }
                    }
                    else if(op(*first1, sequence[0]))
                    {
                        start = first1;
                        cursor = 1;
                    }
                    else
                    {
                        cursor = 0;
                    }
                    first1++;
                };
                find_end_return<FwdIter1> ret = {start, cursor};
                return ret;
            }

        //     template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        //         typename Pred>
        //     static typename util::detail::algorithm_result<
        //         ExPolicy, find_end_return<FwdIter1>
        //     >::type
        //     parallel(ExPolicy && policy, FwdIter1 first1, FwdIter1 last1,
        //         SeqVec sequence, Pred && op,
        //         FwdIter1 start, unsigned int g_cursor = 0)
        //     {
        //         typedef util::detail::algorithm_result<ExPolicy, FwdIter1> result;
        //         typedef typename std::iterator_traits<FwdIter1>::reference reference;
        //         typedef typename std::iterator_traits<FwdIter1>::difference_type
        //             difference_type;
        //
        //         difference_type diff = sequence.size();
        //         if (diff <= 0)
        //             return result::get(std::move(last1));
        //
        //         difference_type count = std::distance(first1, last1);
        //         if (diff > count)
        //             return result::get(std::move(last1));
        //
        //         util::cancellation_token<
        //             difference_type, std::greater<difference_type>
        //         > tok(-1);
        //
        //         unsigned int cursor = g_cursor;
        //
        //         return util::partitioner<ExPolicy, FwdIter1, void>::
        //             call_with_index(
        //                 std::forward<ExPolicy>(policy), first1, count-(diff-1), 1,
        //                 [=](FwdIter1 it, std::size_t part_size,
        //                     std::size_t base_idx) mutable
        //                 {
        //                     FwdIter1 curr = it;
        //
        //                     util::loop_idx_n(
        //                         base_idx, it, part_size, tok,
        //                         [=, &tok, &curr](reference t, std::size_t i)
        //                         {
        //                             ++curr;
        //                             if (op(t, *first2))
        //                             {
        //                                 difference_type local_count = 1;
        //                                 FwdIter2 needle = first2;
        //                                 FwdIter1 mid = curr;
        //
        //                                 for (difference_type len = 0;
        //                                      local_count != diff && len != count;
        //                                      (void) ++local_count, ++len, ++mid)
        //                                 {
        //                                     if (*mid != *++needle)
        //                                         break;
        //                                 }
        //
        //                                 if (local_count == diff)
        //                                     tok.cancel(i);
        //                             }
        //                         });
        //                 },
        //                 [=](std::vector<hpx::future<void> > &&) mutable -> FwdIter1
        //                 {
        //                     difference_type find_end_res = tok.get_data();
        //                     if (find_end_res != count)
        //                         std::advance(first1, find_end_res);
        //                     else
        //                         first1 = last1;
        //
        //                     return std::move(first1);
        //                 });
        //     }
        };

        template <typename Algo, typename ExPolicy, typename FwdIter1,
            typename FwdIter2, typename Pred>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, FwdIter1>::type
        >::type
        segmented_find_end(Algo && algo, ExPolicy && policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, FwdIter2 last2, Pred && op, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter1> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, FwdIter1> result;
            typedef typename std::iterator_traits<FwdIter2>::value_type seq_value_type;

            segment_iterator sit = traits::segment(first1);
            segment_iterator send = traits::segment(last1);
            FwdIter1 output = first1;

            std::vector<seq_value_type> sequence(first2, last2);

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first1);
                local_iterator_type end = traits::local(last1);
                if (beg != end)
                {
                    find_end_return<local_iterator_type> out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, sequence, op, beg
                    );
                    output=traits::compose(send, out.seq_first);
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first1);
                local_iterator_type end = traits::end(sit);
                find_end_return<local_iterator_type> out;
                out.seq_first = traits::local(last1);

                if (beg != end)
                {
                    out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, sequence, op, beg
                    );
                    if(out.seq_first != end)
                        output=traits::compose(sit, out.seq_first);
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    out.seq_first = traits::begin(send);
                    if (beg != end)
                    {
                        out = dispatch(traits::get_id(sit),
                            algo, policy, std::true_type(), beg, end, sequence,
                            op, out.seq_first, out.partial_position
                        );
                        if(out.seq_first != end)
                            output=traits::compose(sit,out.seq_first);
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last1);
                if (beg != end)
                {
                    out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, sequence,
                        op, out.seq_first, out.partial_position
                    );
                    if(out.seq_first != end)
                        output=traits::compose(sit,out.seq_first);
                }
            }
            return result::get(std::move(output));
        }

        template <typename Algo, typename ExPolicy, typename FwdIter1,
            typename FwdIter2, typename Pred>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, FwdIter1>::type
        >::type
        segmented_find_end(Algo && algo, ExPolicy && policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, FwdIter2 last2, Pred && op, std::false_type)
        {
          return util::detail::algorithm_result<
                  ExPolicy, FwdIter1
              >::get(std::forward<FwdIter1>(first1));
        }
        //     typedef hpx::traits::segmented_iterator_traits<FwdIter1> traits;
        //     typedef typename traits::segment_iterator segment_iterator;
        //     typedef typename traits::local_iterator local_iterator_type;
        //     typedef util::detail::algorithm_result<ExPolicy, FwdIter1> result;
        //
        //     typedef std::integral_constant<bool,
        //             !hpx::traits::is_forward_iterator<FwdIter1>::value
        //         > forced_seq;
        //
        //     segment_iterator sit = traits::segment(first1);
        //     segment_iterator send = traits::segment(last1);
        //
        //     std::vector<future<FwdIter1> > segments;
        //     segments.reserve(std::distance(sit, send));
        //
        //     if (sit == send)
        //     {
        //         // all elements are on the same partition
        //         local_iterator_type beg = traits::local(first1);
        //         local_iterator_type end = traits::local(last1);
        //         if (beg != end)
        //         {
        //             segments.push_back(
        //                 hpx::make_future<FwdIter1>(
        //                     dispatch_async(traits::get_id(sit), algo,
        //                         policy, forced_seq(), beg, end, first2, last2, op),
        //                     [send,end,last1](local_iterator_type const& out)
        //                         -> FwdIter1
        //                     {
        //                         if(out != end)
        //                             return traits::compose(send, out);
        //                         else
        //                             return last1;
        //                     }));
        //         }
        //     }
        //     else {
        //         // handle the remaining part of the first partition
        //         local_iterator_type beg = traits::local(first1);
        //         local_iterator_type end = traits::end(sit);
        //         if (beg != end)
        //         {
        //             segments.push_back(
        //                 hpx::make_future<FwdIter1>(
        //                     dispatch_async(traits::get_id(sit), algo,
        //                         policy, forced_seq(), beg, end, first2, last2, op),
        //                     [sit,end,last1](local_iterator_type const& out)
        //                         -> FwdIter1
        //                     {
        //                         if(out != end)
        //                             return traits::compose(sit, out);
        //                         else
        //                             return last1;
        //                     }));
        //         }
        //
        //         // handle all of the full partitions
        //         for (++sit; sit != send; ++sit)
        //         {
        //             beg = traits::begin(sit);
        //             end = traits::end(sit);
        //             if (beg != end)
        //             {
        //                 segments.push_back(
        //                     hpx::make_future<FwdIter1>(
        //                         dispatch_async(traits::get_id(sit), algo,
        //                             policy, forced_seq(), beg, end, first2, last2, op),
        //                         [sit,end,last1](local_iterator_type const& out)
        //                             -> FwdIter1
        //                         {
        //                             if(out != end)
        //                                 return traits::compose(sit, out);
        //                             else
        //                                 return last1;
        //                         }));
        //             }
        //         }
        //
        //         // handle the beginning of the last partition
        //         beg = traits::begin(sit);
        //         end = traits::local(last1);
        //         if (beg != end)
        //         {
        //             segments.push_back(
        //                 hpx::make_future<FwdIter1>(
        //                     dispatch_async(traits::get_id(sit), algo,
        //                         policy, forced_seq(), beg, end, first2, last2, op),
        //                     [sit,end,last1](local_iterator_type const& out)
        //                         -> FwdIter1
        //                     {
        //                         if(out != end)
        //                             return traits::compose(sit, out);
        //                         else
        //                             return last1;
        //                     }));
        //         }
        //     }
        //     return result::get(
        //         dataflow(
        //             [=](std::vector<hpx::future<FwdIter1> > && r)
        //                 ->  FwdIter1
        //             {
        //                 // handle any remote exceptions, will throw on error
        //                 std::list<std::exception_ptr> errors;
        //                 parallel::util::detail::handle_remote_exceptions<
        //                     ExPolicy
        //                 >::call(r, errors);
        //
        //                 std::vector<FwdIter1> res =
        //                     hpx::util::unwrapped(std::move(r));
        //                 auto it = res.begin();
        //                 while(it!=res.end())
        //                 {
        //                     if(*it != last1)
        //                         return *it;
        //                     it++;
        //                 }
        //                 return res.back();
        //             },
        //             std::move(segments)));
        // }

        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Pred>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, FwdIter1>::type
        >::type
        find_end_(ExPolicy && policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, FwdIter2 last2, Pred && op, std::true_type)
        {
            typedef parallel::execution::is_sequenced_execution_policy<
                    ExPolicy
                > is_seq;

            if (first1 == last1)
            {
                return util::detail::algorithm_result<
                        ExPolicy, FwdIter1
                    >::get(std::forward<FwdIter1>(first1));
            }
            typedef typename std::iterator_traits<FwdIter1>::value_type type;
            typedef hpx::traits::segmented_iterator_traits<FwdIter1>
                iterator_traits;
            return segmented_find_end(
                seg_find_end<typename iterator_traits::local_iterator>(),
                std::forward<ExPolicy>(policy), first1, last1, first2, last2,
                std::forward<Pred>(op),is_seq());
        }

        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Pred>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, FwdIter1>::type
        >::type
        find_end_(ExPolicy && policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, FwdIter2 last2, Pred && op, std::false_type);
    }
}}}
#endif
