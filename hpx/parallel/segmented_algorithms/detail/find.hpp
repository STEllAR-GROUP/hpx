//  Copyright (c) 2017 Ajai V George
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_FIND)
#define HPX_PARALLEL_SEGMENTED_FIND

#include <hpx/util/invoke.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail
{
    /// \cond NOINTERNAL
    template<typename Iter>
    struct seg_find_end : public detail::algorithm<seg_find_end<Iter>, find_return<Iter> >
    {
        seg_find_end()
          : seg_find_end<Iter>::algorithm("segmented_find_end")
        {}

        template <typename ExPolicy, typename FwdIter1, typename SeqVec,
            typename Pred>
        static find_return<FwdIter1>
        sequential(ExPolicy, FwdIter1 first1, FwdIter1 last1,
            SeqVec sequence, Pred && op,
            FwdIter1 start, unsigned int g_cursor = 0)
        {
            unsigned int cursor = g_cursor;
            bool found = false;
            unsigned int found_cursor = 0;
            FwdIter1 found_start = start;
            while(last1 != first1)
            {
                if(hpx::util::invoke(op, *first1, sequence[cursor]))
                {
                    if(cursor == 0)
                        start = first1;
                    cursor++;
                    if(cursor == sequence.size())
                    {
                        found_cursor = cursor;
                        cursor = 0;
                        found = true;
                        found_start = start;
                    }

                }
                else if(hpx::util::invoke(op, *first1, sequence[0]))
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
            if(found)
            {
                cursor = found_cursor;
                start = found_start;
            }
            find_return<FwdIter1> ret = {std::move(start), cursor,
                std::move(last1), 0};
            return ret;
        }

        template <typename ExPolicy, typename FwdIter1, typename SeqVec,
            typename Pred>
        static typename util::detail::algorithm_result<
            ExPolicy, find_return<FwdIter1>
        >::type
        parallel(ExPolicy && policy, FwdIter1 first1, FwdIter1 last1,
            SeqVec sequence, Pred && op)
        {
            typedef util::detail::algorithm_result<ExPolicy,
                find_return<FwdIter1> > result;
            typedef typename std::iterator_traits<FwdIter1>::reference reference;
            typedef typename std::iterator_traits<FwdIter1>::difference_type
                difference_type;

            find_return<FwdIter1> ret;

            difference_type diff = sequence.size();
            if (diff <= 0)
            {
                ret = {std::move(last1), 0, std::move(first1), 0};
                return result::get(ret);
            }

            difference_type count = std::distance(first1, last1);

            util::cancellation_token<
                difference_type, std::greater<difference_type>
            > tok(-1);

            return util::partitioner<ExPolicy, find_return<FwdIter1>, void>::
                call_with_index(
                    std::forward<ExPolicy>(policy), first1, count, 1,
                    [=](FwdIter1 it, std::size_t part_size,
                        std::size_t base_idx) mutable
                    {
                        FwdIter1 curr = it;
                        util::loop_idx_n(
                            base_idx, it, part_size, tok,
                            [=, &tok, &curr](reference t, std::size_t i)
                            {
                                ++curr;
                                if (hpx::util::invoke(op, t, sequence[0]))
                                {
                                    difference_type local_count = 1;

                                    FwdIter1 mid = curr;

                                    for (difference_type len = 0;
                                         local_count != diff && len != count;
                                         (void) ++local_count, ++len, ++mid)
                                    {
                                        if (!hpx::util::invoke(op, *mid,
                                                sequence[local_count]))
                                            break;
                                    }

                                    if (local_count == diff)
                                        tok.cancel(i);
                                }
                            });
                    },
                    [=](std::vector<hpx::future<void> > &&) mutable
                        -> find_return<FwdIter1>
                    {
                        FwdIter1 seq_start = first1, seq_last;
                        std::size_t partial_position, last_position = 0;
                        difference_type find_end_res = tok.get_data();
                        if (find_end_res != count)
                        {
                            std::advance(seq_start, find_end_res);
                            partial_position = sequence.size();
                        }
                        else
                        {
                            FwdIter1 curr = first1;
                            std::size_t index = 0;
                            while(!hpx::util::invoke(op, *curr, sequence[index]))
                            {
                                index++;
                            };
                            partial_position = index;
                            seq_start = curr;
                            for(; index != sequence.size(); index++)
                            {
                                if(hpx::util::invoke(op, *curr, sequence[index]))
                                {
                                    curr++;
                                }
                                else if(hpx::util::invoke(op, *(--curr), sequence[index]))
                                {
                                    ;
                                }
                                else
                                    break;
                            }
                            if(index != sequence.size())
                            {
                                partial_position = 0;
                                seq_start = last1;
                            }
                        }
                        FwdIter1 curr = last1 - 1;
                        std::size_t index = 0;
                        for(; std::distance(first1, curr) != 0; curr--)
                        {
                            if(hpx::util::invoke(op, *curr, sequence[index]))
                                break;
                            if(std::distance(curr, last1)>diff)
                            {
                                curr = last1;
                                break;
                            }
                        }
                        seq_last = curr;
                        if(curr != last1 && curr != seq_start)
                        {
                            for(; curr != last1; curr++, index++)
                            {
                                if(!hpx::util::invoke(op, *curr, sequence[index]))
                                {
                                    seq_last = last1;
                                    last_position = 0;
                                }
                            }
                            if(seq_last != last1)
                                last_position = index-1;
                        }
                        find_return<FwdIter1> ret = {std::move(seq_start),
                            partial_position, std::move(seq_last), last_position};
                        printf("Combine partition difference_type = %ld partial_position = %ld last_position = %ld\n", find_end_res,
                            partial_position, last_position);
                        return ret;
                    });
        }
    };

    template<typename Iter>
    struct seg_find_first_of : public detail::algorithm<seg_find_first_of<Iter>, find_return<Iter> >
    {
        seg_find_first_of()
          : seg_find_first_of<Iter>::algorithm("segmented_find_first_of")
        {}

        template <typename ExPolicy, typename FwdIter1, typename SeqVec,
            typename Pred>
        static find_return<FwdIter1>
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
                        find_return<FwdIter1> ret = {std::move(last1),
                            0, std::move(start), (unsigned int) sequence.size()};
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
            find_return<FwdIter1> ret = {std::move(last1), 0,
                std::move(start), cursor};
            return ret;
        }

        template <typename ExPolicy, typename FwdIter1, typename SeqVec,
            typename Pred>
        static typename util::detail::algorithm_result<
            ExPolicy, find_return<FwdIter1>
        >::type
        parallel(ExPolicy && policy, FwdIter1 first1, FwdIter1 last1,
            SeqVec sequence, Pred && op)
        {
            typedef util::detail::algorithm_result<ExPolicy,
                find_return<FwdIter1> > result;
            typedef typename std::iterator_traits<FwdIter1>::reference reference;
            typedef typename std::iterator_traits<FwdIter1>::difference_type
                difference_type;

            find_return<FwdIter1> ret;

            difference_type diff = sequence.size();
            if (diff <= 0)
            {
                ret = {std::move(first1), 0, std::move(last1), 0};
                return result::get(ret);
            }

            difference_type count = std::distance(first1, last1);

            util::cancellation_token<difference_type> tok(count);


            return util::partitioner<ExPolicy, find_return<FwdIter1>, void>::
                call_with_index(
                    std::forward<ExPolicy>(policy), first1, count, 1,
                    [=](FwdIter1 it,
                        std::size_t part_size, std::size_t base_idx) mutable
                    {
                        FwdIter1 curr = it;
                        util::loop_idx_n(
                            base_idx, it, part_size, tok,
                            [=, &tok, &curr, &op]
                            (reference t, std::size_t i)
                            {
                                ++curr;
                                if (hpx::util::invoke(op, t, sequence[0]))
                                {
                                    difference_type local_count = 1;

                                    FwdIter1 mid = curr;

                                    for (difference_type len = 0;
                                         local_count != diff && len != count;
                                         (void) ++local_count, ++len, ++mid)
                                    {
                                        if (!hpx::util::invoke(op, *mid,
                                                sequence[local_count]))
                                            break;
                                    }

                                    if (local_count == diff)
                                    {
                                        tok.cancel(i);
                                    }
                                }
                            });
                    },
                    [=](std::vector<hpx::future<void> > &&) mutable
                        -> find_return<FwdIter1>
                    {
                        FwdIter1 seq_last = first1, seq_start;
                        std::size_t partial_position = 0, last_position;
                        difference_type find_first_of_res = tok.get_data();
                        if(find_first_of_res != count)
                        {
                            std::advance(seq_last, find_first_of_res);
                            last_position = sequence.size();
                        }
                        else
                        {
                            FwdIter1 curr = last1 - 1;
                            std::size_t index = 0;
                            for(; std::distance(first1, curr) != 0; curr--)
                            {
                                if(hpx::util::invoke(op, *curr, sequence[index]))
                                    break;
                                if(std::distance(curr, last1)>diff)
                                {
                                    curr = last1;
                                    break;
                                }
                            }
                            seq_last = curr;
                            if(curr != last1 && curr != seq_start)
                            {
                                for(; curr != last1; curr++, index++)
                                {
                                    if(!hpx::util::invoke(op, *curr, sequence[index]))
                                    {
                                        seq_last = last1;
                                        last_position = 0;
                                    }
                                }
                                if(seq_last != last1)
                                    last_position = index-1;
                            }
                        }
                        FwdIter1 curr = first1;
                        std::size_t index = 0;
                        while(!hpx::util::invoke(op, *curr, sequence[index]))
                        {
                            index++;
                        };
                        partial_position = index;
                        seq_start = curr;
                        for(; index != sequence.size(); index++)
                        {
                            if(hpx::util::invoke(op, *curr, sequence[index]))
                            {
                                curr++;
                            }
                            else if(hpx::util::invoke(op, *(--curr), sequence[index]))
                            {
                                ;
                            }
                            else
                                break;
                        }
                        if(index != sequence.size())
                        {
                            partial_position = 0;
                            seq_start = last1;
                        }
                        find_return<FwdIter1> ret = {std::move(seq_start),
                            partial_position, std::move(seq_last), last_position};
                        printf("Combine partition difference_type = %ld prefix = %ld partial_position = %ld last_position = %ld\n",
                            find_first_of_res, std::distance(first1, seq_start), partial_position, last_position);
                        return ret;
                    });
        }
    };
}}}}
#endif
