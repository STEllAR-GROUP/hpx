//  Copyright (c) 2017 Ajai V George
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_FIND)
#define HPX_PARALLEL_SEGMENTED_FIND

#include <hpx/util/invoke.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>

#include <cstddef>
#include <utility>
#include <vector>

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
            SeqVec sequence, Pred && op, unsigned int g_cursor = 0)
        {
            unsigned int cursor = g_cursor, found_cursor = 0;
            FwdIter1 found_start = last1, start = last1;
            while(last1 != first1)
            {
                // test if sequence matches for current cursor and iter position
                if(cursor != sequence.size() &&
                    hpx::util::invoke(op, *first1, sequence[cursor]))
                {
                    // if beginning of sequence set start
                    if(cursor == 0)
                        start = first1;
                    ++cursor;
                    // if complete sequence found, then store result and continue search
                    if(cursor == sequence.size())
                    {
                        found_cursor = cursor;
                        cursor = 0;
                        found_start = start;
                    }
                }
                // if iter does not match current sequence position then check
                    // with beginning
                else if(hpx::util::invoke(op, *first1, sequence[0]))
                {
                    start = first1;
                    cursor = 1;
                }
                //otherwise start search again
                else
                {
                    start = last1;
                    cursor = 0;
                }
                ++first1;
            }
            // return consists of positions of complete sequence if found and
                // partial sequence at the beginning
            return std::move(find_return<FwdIter1>{std::move(found_start), found_cursor,
                std::move(start), cursor});
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
                return result::get(find_return<FwdIter1>{
                    std::move(last1), 0, std::move(first1), 0});

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
                        // loop within a thread to see if complete sequence present.
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
                        FwdIter1 complete_sequence_position = first1;
                        FwdIter1 partial_sequence_position = last1;
                        std::size_t complete_sequence_cursor = 0;
                        std::size_t partial_sequence_cursor = sequence.size();
                        difference_type find_end_res = tok.get_data();
                        bool span = false;
                        if (find_end_res != count && find_end_res != -1)
                        {
                            // complete sequence is present
                            std::advance(complete_sequence_position, find_end_res);
                            complete_sequence_cursor = sequence.size();
                        }
                        else
                        {
                            // complete sequence is not present so search for
                            // partial sequence at the beginning.
                            std::size_t index = 1;
                            //loop till suffix of sequence at beginning found
                            while(index != sequence.size())
                            {
                                if(hpx::util::invoke(op, *first1, sequence[index]))
                                {
                                    FwdIter1 curr = first1;
                                    std::size_t temp_index = index;
                                    while(curr != last1 &&
                                        temp_index != sequence.size() &&
                                        hpx::util::invoke(op, *curr,
                                            sequence[temp_index]))
                                    {
                                        ++curr;
                                        ++temp_index;
                                    }
                                    if(temp_index == sequence.size())
                                    {
                                        complete_sequence_position = first1;
                                        complete_sequence_cursor = index;
                                        break;
                                    }
                                    else if(curr == last1 &&
                                        hpx::util::invoke(op, *(std::prev(curr)),
                                            sequence[temp_index-1]))
                                    {
                                        complete_sequence_position = last1;
                                        complete_sequence_cursor = index;
                                        partial_sequence_cursor = temp_index-1;
                                        span = true;
                                        break;
                                    }
                                }
                                ++index;
                            }
                        }
                        if(!span)
                        {
                            //Also search for partial sequence at the end
                            FwdIter1 curr = first1;
                            std::advance(curr, count-diff+1);
                            //loop till prefix of sequence at end found
                            for(; curr != last1; ++curr)
                            {
                                FwdIter1 temp = curr;
                                std::size_t index = 0;
                                while(temp != last1 &&
                                    hpx::util::invoke(op, *temp, sequence[index]))
                                {
                                    ++temp;
                                    ++index;
                                }
                                if(temp == last1)
                                {
                                    partial_sequence_cursor = index;
                                    partial_sequence_position = curr;
                                    break;
                                }
                            }
                        }
                        // return both results
                        return find_return<FwdIter1>{
                            std::move(complete_sequence_position),
                            complete_sequence_cursor,
                            std::move(partial_sequence_position),
                            partial_sequence_cursor};
                    });
        }
    };

    template<typename Iter>
    struct seg_find_first_of : public detail::algorithm<seg_find_first_of<Iter>,
        find_return<Iter> >
    {
        seg_find_first_of()
          : seg_find_first_of<Iter>::algorithm("segmented_find_first_of")
        {}

        template <typename ExPolicy, typename FwdIter1, typename SeqVec,
            typename Pred>
        static find_return<FwdIter1>
        sequential(ExPolicy, FwdIter1 first1, FwdIter1 last1,
            SeqVec sequence, Pred && op, unsigned int g_cursor = 0)
        {
            unsigned int cursor = g_cursor, found_cursor = 0;
            FwdIter1 found_start = last1, start = last1;
            bool found = false;
            while(last1 != first1)
            {
                // test if sequence matches for current cursor and iter position
                if(hpx::util::invoke(op, *first1, sequence[cursor]))
                {
                    // if beginning of sequence set start
                    if(cursor == 0)
                        start = first1;
                    ++cursor;
                    // if complete sequence found, then store result
                    // and continue search
                    if(cursor == sequence.size())
                    {
                        //only if sequence not previously found
                        if(!found)
                        {
                            found = true;
                            found_cursor = cursor;
                            found_start = start;
                        }
                        cursor = 0;
                    }
                }
                // if iter does not match current sequence position then check
                    // with beginning
                else if(hpx::util::invoke(op, *first1, sequence[0]))
                {
                    start = first1;
                    cursor = 1;
                }
                //otherwise start search again
                else
                {
                    start = last1;
                    cursor = 0;
                }
                ++first1;
            }
            // return consists of positions of complete sequence if found and
                // partial sequence at the beginning
            return find_return<FwdIter1>{std::move(found_start), found_cursor,
                std::move(start), cursor};
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
                return result::get(find_return<FwdIter1>{
                    std::move(last1), 0, std::move(last1), 0});

            difference_type count = std::distance(first1, last1);

            util::cancellation_token<difference_type> tok(count);


            return util::partitioner<ExPolicy, find_return<FwdIter1>, void>::
                call_with_index(
                    std::forward<ExPolicy>(policy), first1, count, 1,
                    [=](FwdIter1 it,
                        std::size_t part_size, std::size_t base_idx) mutable
                    {
                        // loop within a thread to see if complete sequence present.
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
                        FwdIter1 complete_sequence_position = first1;
                        FwdIter1 partial_sequence_position = last1;
                        std::size_t complete_sequence_cursor = 0;
                        std::size_t partial_sequence_cursor = sequence.size();
                        difference_type find_first_of_res = tok.get_data();
                        bool span = false;
                        if(find_first_of_res != count)
                        {
                            // complete sequence is not present so search for
                            // partial sequence at the beginning.
                            std::advance(complete_sequence_position,
                                find_first_of_res);
                            complete_sequence_cursor = sequence.size();
                        }
                        else
                        {
                            // complete sequence is not present so search for
                            // partial sequence at the beginning.
                            std::size_t index = 1;
                            //loop till suffix of sequence at beginning found
                            while(index != sequence.size())
                            {
                                if(hpx::util::invoke(op, *first1, sequence[index]))
                                {
                                    FwdIter1 curr = first1;
                                    std::size_t temp_index = index;
                                    while(curr != last1 &&
                                        temp_index != sequence.size() &&
                                        hpx::util::invoke(op, *curr,
                                            sequence[temp_index]))
                                    {
                                        ++curr;
                                        ++temp_index;
                                    }
                                    if(temp_index == sequence.size())
                                    {
                                        complete_sequence_position = first1;
                                        complete_sequence_cursor = index;
                                        break;
                                    }
                                    else if(curr == last1 &&
                                        hpx::util::invoke(op, *(std::prev(curr)),
                                            sequence[temp_index-1]))
                                    {
                                        complete_sequence_position = last1;
                                        complete_sequence_cursor = index;
                                        partial_sequence_cursor = temp_index;
                                        span = true;
                                        break;
                                    }
                                }
                                ++index;
                            }
                        }
                        if(!span)
                        {
                            //Also search for partial sequence at the end
                            FwdIter1 curr = first1;
                            std::advance(curr, count-diff+1);
                            //loop till prefix of sequence at end found
                            for(; curr != last1; ++curr)
                            {
                                FwdIter1 temp = curr;
                                std::size_t index = 0;
                                while(temp != last1 &&
                                    hpx::util::invoke(op, *temp, sequence[index]))
                                {
                                    ++temp;
                                    ++index;
                                }
                                if(temp == last1)
                                {
                                    partial_sequence_cursor = index;
                                    partial_sequence_position = curr;
                                    break;
                                }
                            }
                        }
                        return find_return<FwdIter1>{
                            std::move(complete_sequence_position),
                            complete_sequence_cursor,
                            std::move(partial_sequence_position),
                            partial_sequence_cursor};
                    });
        }
    };
}}}}
#endif
