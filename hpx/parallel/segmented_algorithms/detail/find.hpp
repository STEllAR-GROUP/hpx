//  Copyright (c) 2017 Ajai V George
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_FIND_RETURN)
#define HPX_PARALLEL_SEGMENTED_FIND_RETURN

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
            while(last1 != first1)
            {
                if(op(*first1, sequence[cursor]))
                {
                    if(cursor == 0)
                        start = first1;
                    cursor++;
                    if(cursor == sequence.size())
                    {
                        find_return<FwdIter1> ret = {start,
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
            find_return<FwdIter1> ret = {start, cursor};
            return ret;
        }

        template <typename ExPolicy, typename FwdIter1, typename SeqVec,
            typename Pred>
        static typename util::detail::algorithm_result<
            ExPolicy, find_return<FwdIter1>
        >::type
        parallel(ExPolicy && policy, FwdIter1 first1, FwdIter1 last1,
            SeqVec sequence, Pred && op,
            FwdIter1 start, unsigned int g_cursor = 0)
        {
            typedef util::detail::algorithm_result<ExPolicy, FwdIter1> result;
            typedef typename std::iterator_traits<FwdIter1>::reference reference;
            typedef typename std::iterator_traits<FwdIter1>::difference_type
                difference_type;

            difference_type diff = sequence.size();
            if (diff <= 0)
                return result::get(std::move(last1));

            difference_type count = std::distance(first1, last1);
            if (diff > count)
                return result::get(std::move(last1));

            util::cancellation_token<
                difference_type, std::greater<difference_type>
            > tok(-1);

            unsigned int cursor = g_cursor;

            return util::partitioner<ExPolicy, FwdIter1, void>::
                call_with_index(
                    std::forward<ExPolicy>(policy), first1, count-(diff-1), 1,
                    [=](FwdIter1 it, std::size_t part_size,
                        std::size_t base_idx) mutable
                    {
                        FwdIter1 curr = it;

                        util::loop_idx_n(
                            base_idx, it, part_size, tok,
                            [=, &tok, &curr, &sequence, &cursor](reference t, std::size_t i)
                            {
                                ++curr;
                                if (op(t, sequence[cursor]))
                                {
                                    difference_type local_count = 1;

                                    FwdIter1 mid = curr;

                                    for (difference_type len = cursor;
                                         local_count != diff && len != count;
                                         (void) ++local_count, ++len, ++mid)
                                    {
                                        if (!op(*mid, sequence[local_count]))
                                            break;
                                    }

                                    if (local_count == diff)
                                        tok.cancel(i);
                                    cursor = local_count;
                                }
                            });
                    },
                    [=](std::vector<hpx::future<void> > &&) mutable
                        -> find_return<FwdIter1>
                    {
                        difference_type find_end_res = tok.get_data();
                        if (find_end_res != count)
                            std::advance(first1, find_end_res);
                        else
                            first1 = last1;
                        find_return<FwdIter1> ret = {std::move(first1),
                            cursor};
                        return ret;
                    });
        }
    };
}}}}
#endif
