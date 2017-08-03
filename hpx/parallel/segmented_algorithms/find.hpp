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
#include <hpx/parallel/segmented_algorithms/detail/find.hpp>
#include <hpx/parallel/segmented_algorithms/detail/find_return.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cstddef>
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
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
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
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
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
                            [=](local_iterator_type const& out)
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
                            [=](local_iterator_type const& out)
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
                                [=](local_iterator_type const& out)
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
                            [=](local_iterator_type const& out)
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

        template <typename Algo, typename ExPolicy, typename FwdIter1,
            typename FwdIter2, typename Pred>
        typename util::detail::algorithm_result<ExPolicy, FwdIter1>::type
        segmented_find_end(Algo && algo, ExPolicy && policy,
            FwdIter1 first1, FwdIter1 last1, FwdIter2 first2, FwdIter2 last2,
            Pred && op, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter1> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, FwdIter1> result;
            typedef typename std::iterator_traits<FwdIter2>::value_type seq_value_type;

            segment_iterator sit = traits::segment(first1);
            segment_iterator send = traits::segment(last1);

            FwdIter1 output = last1;

            std::vector<seq_value_type> sequence(first2, last2);

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first1);
                local_iterator_type end = traits::local(last1);
                if (beg != end)
                {
                    find_return<local_iterator_type> out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, sequence, op
                    );
                    output=traits::compose(sit, out.complete_sequence_position);
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first1);
                local_iterator_type end = traits::end(sit);
                find_return<local_iterator_type> out;
                FwdIter1 partial_out;
                bool partial_found = false;
                out.complete_sequence_position = traits::local(last1);

                if (beg != end)
                {
                    out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, sequence, op
                    );
                    //if complete sequence found the store that
                    if(out.complete_sequence_position != end &&
                        out.complete_sequence_cursor == sequence.size())
                    {
                        output=traits::compose(sit, out.complete_sequence_position);
                    }
                    // if partial sequence found then store that for next segment
                    if (out.partial_sequence_cursor != 0)
                    {
                        partial_out = traits::compose(sit,
                            out.partial_sequence_position);
                        partial_found = true;
                    }
                }
                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        out = dispatch(traits::get_id(sit),
                            algo, policy, std::true_type(), beg, end, sequence,
                            op, out.partial_sequence_cursor
                        );
                        //if complete sequence found the store that
                        if(out.complete_sequence_position != end &&
                            out.complete_sequence_cursor == sequence.size())
                        {
                            output=traits::compose(sit,
                                out.complete_sequence_position);
                        }
                        //if earlier partial sequence found completely then store that
                        else if(partial_found &&
                            out.complete_sequence_cursor == sequence.size())
                        {
                            output = partial_out;
                        }
                        //else discard partial sequence
                        else
                        {
                            partial_found = false;
                        }
                        //check if new partial sequence found
                        if (out.partial_sequence_cursor != 0)
                        {
                            if(out.partial_sequence_cursor <
                                (std::size_t) std::distance(beg, end))
                            {
                                // update partial_out only if not spanning sequence
                                partial_out = traits::compose(sit,
                                    out.partial_sequence_position);
                            }
                            partial_found = true;
                        }
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last1);
                if (beg != end)
                {
                    out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, sequence,
                        op, out.partial_sequence_cursor
                    );
                    //if complete sequence found then store that
                    if(out.complete_sequence_position != end &&
                        out.complete_sequence_cursor == sequence.size())
                    {
                        output=traits::compose(sit, out.complete_sequence_position);
                    }
                    //check if previous partial sequence completely found now
                    else if(partial_found &&
                        out.complete_sequence_cursor == sequence.size())
                    {
                        output = partial_out;
                    }
                }
            }
            return result::get(std::move(output));
        }

        template <typename Algo, typename ExPolicy, typename FwdIter1,
            typename FwdIter2, typename Pred>
        typename util::detail::algorithm_result<ExPolicy, FwdIter1>::type
        segmented_find_end(Algo && algo, ExPolicy && policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, FwdIter2 last2, Pred && op, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter1> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, FwdIter1> result;
            typedef typename std::iterator_traits<FwdIter2>::value_type seq_value_type;

            typedef std::integral_constant<bool,
                    !hpx::traits::is_forward_iterator<FwdIter1>::value
                > forced_seq;

            segment_iterator sit = traits::segment(first1);
            segment_iterator send = traits::segment(last1);

            std::vector<future<find_return<FwdIter1> > > segments;
            segments.reserve(std::distance(sit, send));

            std::vector<seq_value_type> sequence(first2, last2);

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first1);
                local_iterator_type end = traits::local(last1);
                if (beg != end)
                {
                    segments.push_back(
                        hpx::make_future<find_return<FwdIter1> >(
                            dispatch_async(traits::get_id(sit), algo,
                                policy, forced_seq(), beg, end, sequence, op),
                            [=](find_return<local_iterator_type> const& out)
                                -> find_return<FwdIter1>
                            {
                                FwdIter1 it_first, it_last;
                                if(out.complete_sequence_position != end)
                                    it_first =  traits::compose(send,
                                        out.complete_sequence_position);
                                else
                                    it_first = last1;
                                if(out.partial_sequence_position != end)
                                    it_last =  traits::compose(send,
                                        out.partial_sequence_position);
                                else
                                    it_last = last1;
                                return find_return<FwdIter1>{it_first,
                                    out.complete_sequence_cursor,
                                    it_last, out.partial_sequence_cursor};
                            }));
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first1);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    segments.push_back(
                        hpx::make_future<find_return<FwdIter1> >(
                            dispatch_async(traits::get_id(sit), algo,
                                policy, forced_seq(), beg, end, sequence, op),
                            [=](find_return<local_iterator_type> const& out)
                                -> find_return<FwdIter1>
                            {
                                FwdIter1 it_first, it_last;
                                if(out.complete_sequence_position != end)
                                    it_first =  traits::compose(sit,
                                        out.complete_sequence_position);
                                else
                                    it_first = last1;
                                if(out.partial_sequence_position != end)
                                    it_last =  traits::compose(sit,
                                        out.partial_sequence_position);
                                else
                                    it_last = last1;
                                return find_return<FwdIter1>{it_first,
                                    out.complete_sequence_cursor,
                                    it_last, out.partial_sequence_cursor};
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
                            hpx::make_future<find_return<FwdIter1> >(
                                dispatch_async(traits::get_id(sit), algo,
                                    policy, forced_seq(), beg, end, sequence, op),
                                [=](find_return<local_iterator_type> const& out)
                                    -> find_return<FwdIter1>
                                {
                                    FwdIter1 it_first, it_last;
                                    if(out.complete_sequence_position != end)
                                        it_first =  traits::compose(sit,
                                            out.complete_sequence_position);
                                    else
                                        it_first = last1;
                                    if(out.partial_sequence_position != end)
                                        it_last =  traits::compose(sit,
                                            out.partial_sequence_position);
                                    else
                                        it_last = last1;
                                    return find_return<FwdIter1>{it_first,
                                        out.complete_sequence_cursor,
                                        it_last, out.partial_sequence_cursor};
                                }));
                    }
                }
                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last1);
                if (beg != end)
                {
                    segments.push_back(
                        hpx::make_future<find_return<FwdIter1> >(
                            dispatch_async(traits::get_id(sit), algo,
                                policy, forced_seq(), beg, end, sequence, op),
                            [sit,end,last1](find_return<local_iterator_type> const& out)
                                -> find_return<FwdIter1>
                            {
                                FwdIter1 it_first, it_last;
                                if(out.complete_sequence_position != end)
                                    it_first =  traits::compose(sit,
                                        out.complete_sequence_position);
                                else
                                    it_first = last1;
                                if(out.partial_sequence_position != end)
                                    it_last =  traits::compose(sit,
                                        out.partial_sequence_position);
                                else
                                    it_last = last1;
                                return find_return<FwdIter1>{it_first,
                                    out.complete_sequence_cursor,
                                    it_last, out.partial_sequence_cursor};
                            }));
                }
            }
            return result::get(
                dataflow(
                    [=](std::vector<hpx::future<find_return<FwdIter1> > > && r)
                        ->  FwdIter1
                    {
                        // handle any remote exceptions, will throw on error
                        std::list<std::exception_ptr> errors;
                        parallel::util::detail::handle_remote_exceptions<
                            ExPolicy
                        >::call(r, errors);

                        std::vector<find_return<FwdIter1>> res =
                            hpx::util::unwrap(std::move(r));
                        // iterate from the end using a reverse iterator
                        auto it = res.rbegin();
                        while(it!=res.rend())
                        {
                            //if complete sequence found then store that
                            if(it->complete_sequence_position != last1 &&
                                it->complete_sequence_cursor == sequence.size())
                            {
                                return it->complete_sequence_position;
                            }
                            //loop to match partial sequences
                            auto temp = it;
                            while (temp != res.rend() &&
                                std::next(temp)->complete_sequence_cursor !=
                                    sequence.size())
                            {
                                ++temp;
                                if(temp->partial_sequence_cursor !=
                                    std::prev(temp)->complete_sequence_cursor)
                                {
                                    //if partial sequence of current segment
                                    //does not match partial_sequence in the
                                    //segment in front of it
                                    break;
                                }
                                if(temp->partial_sequence_position != last1)
                                {
                                    //if prefix (start) of sequence matched
                                    return temp->partial_sequence_position;
                                }
                            }
                            it++;
                        }
                        return last1;
                    },
                    std::move(segments)));
        }

        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Pred>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter1>::type
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
            typedef hpx::traits::segmented_iterator_traits<FwdIter1>
                iterator_traits;
            return segmented_find_end(
                seg_find_end<typename iterator_traits::local_iterator>(),
                std::forward<ExPolicy>(policy), first1, last1, first2, last2,
                std::forward<Pred>(op), is_seq());
        }

        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Pred>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter1>::type
        find_end_(ExPolicy && policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, FwdIter2 last2, Pred && op, std::false_type);

        template <typename Algo, typename ExPolicy, typename FwdIter1,
            typename FwdIter2, typename Pred>
        typename util::detail::algorithm_result<ExPolicy, FwdIter1>::type
        segmented_find_first_of(Algo && algo, ExPolicy && policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, FwdIter2 last2, Pred && op, std::true_type)
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
                    find_return<local_iterator_type> out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, sequence, op
                    );
                    output=traits::compose(send, out.complete_sequence_position);
                }
            }
            else
            {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first1);
                local_iterator_type end = traits::end(sit);
                find_return<local_iterator_type> out;
                FwdIter1 partial_out;
                bool partial_found = false;
                out.complete_sequence_position = traits::local(last1);
                bool found = false;

                if (beg != end)
                {
                    out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, sequence, op
                    );
                    //if complete sequence found the store that
                    if(out.complete_sequence_position != end &&
                        out.complete_sequence_cursor == sequence.size())
                    {
                        found = true;
                        output=traits::compose(sit, out.complete_sequence_position);
                    }
                    //keep track of partial sequence if found
                    if (out.partial_sequence_cursor != 0)
                    {
                        partial_out = traits::compose(sit,
                            out.partial_sequence_position);
                        partial_found = true;
                        out.complete_sequence_cursor = out.partial_sequence_cursor;
                    }
                }

                // handle all of the full partitions
                for (++sit; sit != send  && !found; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    out.complete_sequence_position = traits::end(sit);
                    if (beg != end) //do this only if sequence not found
                    {
                        out = dispatch(traits::get_id(sit),
                            algo, policy, std::true_type(), beg, end, sequence,
                            op, out.complete_sequence_cursor
                        );
                        //if complete sequence found the store that
                        if(out.complete_sequence_position != end &&
                            out.complete_sequence_cursor == sequence.size())
                        {
                            output=traits::compose(sit,out.complete_sequence_position);
                            found = true;
                            break;
                        }
                        //if earlier partial sequence completed store that
                        else if(partial_found &&
                            out.complete_sequence_cursor == sequence.size())
                        {
                            output = partial_out;
                            found = true;
                        }
                        //else discard earlier partial sequence
                        else
                        {
                            partial_found = false;
                        }
                        //store new partial sequence if found
                        if (out.partial_sequence_cursor != 0)
                        {
                            if(out.partial_sequence_cursor <
                                (std::size_t) std::distance(beg, end))
                            {
                                //update partial_out only if not spanning sequence
                                partial_out = traits::compose(sit,
                                    out.partial_sequence_position);
                            }
                            partial_found = true;
                            out.complete_sequence_cursor = out.partial_sequence_cursor;
                        }
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last1);
                if (beg != end && !found)
                {
                    out = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, sequence,
                        op, out.complete_sequence_cursor
                    );
                    //if complete sequence found the store that
                    if(out.complete_sequence_position != end &&
                        out.complete_sequence_cursor == sequence.size())
                    {
                        output=traits::compose(sit,out.complete_sequence_position);
                        found = true;
                    }
                    //if earlier partial sequence completed then store  that
                    else if(partial_found &&
                        out.complete_sequence_cursor == sequence.size())
                    {
                        found = true;
                        output = partial_out;
                    }
                }
            }
            return result::get(std::move(output));
        }

        template <typename Algo, typename ExPolicy, typename FwdIter1,
            typename FwdIter2, typename Pred>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter1>::type
        segmented_find_first_of(Algo && algo, ExPolicy && policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, FwdIter2 last2, Pred && op, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter1> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, FwdIter1> result;
            typedef typename std::iterator_traits<FwdIter2>::value_type seq_value_type;

            typedef std::integral_constant<bool,
                    !hpx::traits::is_forward_iterator<FwdIter1>::value
                > forced_seq;

            segment_iterator sit = traits::segment(first1);
            segment_iterator send = traits::segment(last1);

            std::vector<future<find_return<FwdIter1> > > segments;
            segments.reserve(std::distance(sit, send));

            std::vector<seq_value_type> sequence(first2, last2);

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first1);
                local_iterator_type end = traits::local(last1);
                if (beg != end)
                {
                    segments.push_back(
                        hpx::make_future<find_return<FwdIter1> >(
                            dispatch_async(traits::get_id(sit), algo,
                                policy, forced_seq(), beg, end, sequence, op),
                            [=](find_return<local_iterator_type> const& out)
                                -> find_return<FwdIter1>
                            {
                                FwdIter1 it_first, it_last;
                                if(out.complete_sequence_position != end)
                                    it_first =  traits::compose(send,
                                        out.complete_sequence_position);
                                else
                                    it_first = last1;
                                if(out.partial_sequence_position != end)
                                    it_last =  traits::compose(send,
                                        out.partial_sequence_position);
                                else
                                    it_last = last1;
                                return find_return<FwdIter1>{it_first,
                                    out.complete_sequence_cursor,
                                    it_last, out.partial_sequence_cursor};
                            }));
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first1);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    segments.push_back(
                        hpx::make_future<find_return<FwdIter1> >(
                            dispatch_async(traits::get_id(sit), algo,
                                policy, forced_seq(), beg, end, sequence, op),
                            [=](find_return<local_iterator_type> const& out)
                                -> find_return<FwdIter1>
                            {
                                FwdIter1 it_first, it_last;
                                if(out.complete_sequence_position != end)
                                    it_first =  traits::compose(sit,
                                        out.complete_sequence_position);
                                else
                                    it_first = last1;
                                if(out.partial_sequence_position != end)
                                    it_last =  traits::compose(sit,
                                        out.partial_sequence_position);
                                else
                                    it_last = last1;
                                return find_return<FwdIter1>{it_first,
                                    out.complete_sequence_cursor,
                                    it_last, out.partial_sequence_cursor};
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
                            hpx::make_future<find_return<FwdIter1> >(
                                dispatch_async(traits::get_id(sit), algo,
                                    policy, forced_seq(), beg, end, sequence, op),
                                [=](find_return<local_iterator_type> const& out)
                                    -> find_return<FwdIter1>
                                {
                                    FwdIter1 it_first, it_last;
                                    if(out.complete_sequence_position != end)
                                        it_first =  traits::compose(sit,
                                            out.complete_sequence_position);
                                    else
                                        it_first = last1;
                                    if(out.partial_sequence_position != end)
                                        it_last =  traits::compose(sit,
                                            out.partial_sequence_position);
                                    else
                                        it_last = last1;
                                    return find_return<FwdIter1>{it_first,
                                        out.complete_sequence_cursor,
                                        it_last, out.partial_sequence_cursor};
                                }));
                    }
                }
                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last1);
                if (beg != end)
                {
                    segments.push_back(
                        hpx::make_future<find_return<FwdIter1> >(
                            dispatch_async(traits::get_id(sit), algo,
                                policy, forced_seq(), beg, end, sequence, op),
                            [=](find_return<local_iterator_type> const& out)
                                -> find_return<FwdIter1>
                            {
                                FwdIter1 it_first, it_last;
                                if(out.complete_sequence_position != end)
                                    it_first =  traits::compose(sit,
                                        out.complete_sequence_position);
                                else
                                    it_first = last1;
                                if(out.partial_sequence_position != end)
                                    it_last =  traits::compose(sit,
                                        out.partial_sequence_position);
                                else
                                    it_last = last1;
                                return find_return<FwdIter1>{it_first,
                                    out.complete_sequence_cursor,
                                    it_last, out.partial_sequence_cursor};
                            }));
                }
            }
            return result::get(
                dataflow(
                    [=](std::vector<hpx::future<find_return<FwdIter1> > > && r)
                        ->  FwdIter1
                    {
                        // handle any remote exceptions, will throw on error
                        std::list<std::exception_ptr> errors;
                        parallel::util::detail::handle_remote_exceptions<
                            ExPolicy
                        >::call(r, errors);

                        std::vector<find_return<FwdIter1>> res =
                            hpx::util::unwrap(std::move(r));

                        //iterate from the first segment
                        auto it = res.begin();
                        while(it != res.end())
                        {
                            //if complete sequence found the return that
                            if(it->complete_sequence_position != last1 &&
                                it->complete_sequence_cursor == sequence.size())
                            {
                                return it->complete_sequence_position;
                            }
                            //if partial sequence found in this segment
                            if(it->partial_sequence_position != last1)
                            {
                                auto temp = std::next(it);
                                while (temp != res.end())
                                {
                                    if(temp->complete_sequence_cursor !=
                                        std::prev(temp)->partial_sequence_cursor)
                                    {
                                        //if found partial sequence matches
                                        //previous segment
                                        break;
                                    }
                                    if(temp->complete_sequence_position != last1)
                                    {
                                        // if ending suffix of sequence found
                                        return it->partial_sequence_position;
                                    }
                                    ++temp;
                                }
                            }
                            it++;
                        }
                        return last1;
                    },
                    std::move(segments)));
        }

        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Pred>
        typename util::detail::algorithm_result<ExPolicy, FwdIter1>::type
        find_first_of_(ExPolicy && policy, FwdIter1 first, FwdIter1 last,
            FwdIter2 s_first, FwdIter2 s_last, Pred && op, std::true_type)
        {
            typedef parallel::execution::is_sequenced_execution_policy<
                    ExPolicy
                > is_seq;

            if (first == last)
            {
                return util::detail::algorithm_result<
                        ExPolicy, FwdIter1
                    >::get(std::forward<FwdIter1>(first));
            }
            typedef hpx::traits::segmented_iterator_traits<FwdIter1>
                iterator_traits;
            return segmented_find_first_of(
                seg_find_first_of<typename iterator_traits::local_iterator>(),
                std::forward<ExPolicy>(policy), first, last, s_first, s_last,
                std::forward<Pred>(op), is_seq());
        }

        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Pred>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter1>::type
        find_first_of_(ExPolicy && policy, FwdIter1 first, FwdIter1 last,
            FwdIter2 s_first, FwdIter2 s_last, Pred && op, std::false_type);
    }
}}}
#endif
