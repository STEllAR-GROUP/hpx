//  Copyright (c) 2015 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/make_heap.hpp

#if !defined(HPX_PARALLEL_ALGORITHMS_MAKE_HEAP_DEC_10_2015_0331PM)
#define HPX_PARALLEL_ALGORITHMS_MAKE_HEAP_DEC_10_2015_0331PM

#include <hpx/lcos/local/dataflow.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/util/bind.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>

#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>

#include <algorithm>
#include <iterator>
#include <functional>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/type_traits/is_same.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    //////////////////////////////////////////////////////////////////////
    // make_heap
    namespace detail
    {
        // Perform bottom up heap construction given a range of elements.
        // sift_down_range will take a range from [start,start-count) and
        // apply sift_down to each element in the range
        template <typename RndIter, typename Pred>
        void sift_down(RndIter first, Pred && pred,
                typename std::iterator_traits<RndIter>::difference_type len,
                RndIter start)
        {
            typedef typename std::iterator_traits<RndIter>::difference_type difference_type;
            typedef typename std::iterator_traits<RndIter>::value_type value_type;

            difference_type child = start - first;

            if(len < 2 || (len - 2) / 2 < child)
                return;

            child = 2 * child + 1;
            RndIter child_i = first + child;

            if ((child + 1) < len && pred(*child_i, *(child_i+1))) {
                ++child_i;
                ++child;
            }

            if(pred(*child_i, *start))
                return;

            value_type top = *start;
            do {
                *start = *child_i;
                start = child_i;

                if ((len - 2) / 2 < child)
                    break;

                child = 2 * child + 1;
                child_i = first + child;

                if ((child + 1) < len && pred(*child_i, *(child_i + 1))) {
                    ++child_i;
                    ++child;
                }
            }while(!pred(*child_i, top));
            *start = top;
        }

        template <typename RndIter, typename Pred>
        void sift_down_range(RndIter first, Pred && pred,
                typename std::iterator_traits<RndIter>::difference_type len,
                RndIter start, std::size_t count)
        {
            typedef typename std::iterator_traits<RndIter>::difference_type dtype;
            for(std::size_t i = 0; i < count; i++) {
                sift_down<RndIter>(first, std::forward<Pred>(pred), len, start - i);
            }
        }

        /// \cond NOINTERNAL
        struct make_heap: public detail::algorithm<make_heap>
        {
            make_heap()
                : make_heap::algorithm("make_heap")
            {}

            template<typename ExPolicy, typename RndIter, typename Pred>
            static hpx::util::unused_type
            sequential(ExPolicy, RndIter first, RndIter last,
                    Pred && pred)
            {
                std::make_heap(first, last, std::forward<Pred>(pred));
                return hpx::util::unused;
            }

            template<typename ExPolicy, typename RndIter, typename Pred>
            static typename util::detail::algorithm_result<ExPolicy>::type
            parallel(ExPolicy policy, RndIter first, RndIter last,
                    Pred && pred)
            {
                typedef typename ExPolicy::executor_type executor_type;
                typedef typename hpx::parallel::executor_traits<executor_type>
                    executor_traits;
                typedef typename std::iterator_traits<RndIter>::difference_type
                    dtype;

                dtype n = last - first;

                if(n <= 1) {
                    return util::detail::algorithm_result<ExPolicy>::get();
                }

                std::list<boost::exception_ptr> errors;
                std::size_t chunk_size = 4;

                std::vector<hpx::future<void> > workitems;
                workitems.reserve(std::distance(first,last)/chunk_size);
                try{
                    // Beginning at node (n-2)/2, partition the items within a level
                    // and execute each chunk asynchronously. Each level must synchronize
                    // before continuing up, loop will iterate over levels not items
                    for(dtype start = (n-2)/2; start > 0;
                        start = (dtype)pow(2, (dtype)log2(start)) - 2) {
                        dtype end_exclusive = (dtype)pow(2, (dtype)log2(start))-2;

                        // The amount of work for this level
                        std::size_t items = (start-end_exclusive);

                        // TO-DO: determine the best solution for when chunk_size becomes
                        // too large as the heap shrinks
                        if(chunk_size > items)
                            chunk_size = items / 2;

                        // Sift down the nodes children and reposition 
                        std::size_t cnt = 0;
                        while(cnt + chunk_size < items) {
                            // Perform sift_down_range on each chunk
                            auto op = 
                                hpx::util::bind(
                                    &sift_down_range<RndIter, Pred&>, first,
                                    std::forward<Pred&>(pred), n, first + start - cnt,
                                    chunk_size);

                            workitems.push_back(
                                executor_traits::async_execute(policy.executor(), op));
                            
                            cnt += chunk_size;
                        }

                        if(cnt < items) {
                            // Perform sift_down_range on remaining items
                            auto op =
                                hpx::util::bind(
                                    &sift_down_range<RndIter, Pred&>, first,
                                    std::forward<Pred&>(pred), n, first + start - cnt,
                                    items-cnt);
     
                            workitems.push_back(
                                executor_traits::async_execute(policy.executor(), op));
                            
                        }
                        // Synchronize level
                        hpx::wait_all(workitems);
                    }
                    // Sift down the first element synchronously
                    sift_down_range(first, std::forward<Pred>(pred), n, first, 1);
                } catch(...) {
                    util::detail::handle_local_exceptions<ExPolicy>::call(
                            boost::current_exception(), errors);
                }
                
                util::detail::handle_local_exceptions<ExPolicy>::call(
                        workitems, errors); 
                return util::detail::algorithm_result<ExPolicy>::get();
            }

            template<typename RndIter, typename Pred>
            static typename util::detail::algorithm_result<parallel_task_execution_policy>::type
            parallel(parallel_task_execution_policy policy, RndIter first, RndIter last,
                    Pred && pred)
            {
                typedef typename parallel_task_execution_policy::executor_type executor_type;
                typedef typename hpx::parallel::executor_traits<executor_type>
                    executor_traits;
                typedef typename std::iterator_traits<RndIter>::difference_type
                    dtype;

                // Size of list
                dtype n = last - first;

                if(n <= 1) {
                    return util::detail::algorithm_result<parallel_task_execution_policy>::get();
                }

                std::list<boost::exception_ptr> errors;
                std::size_t chunk_size = 4;

                std::vector<hpx::future<void> > workitems;
                workitems.reserve(std::distance(first,last)/chunk_size);
                try{
                    // Beginning at node (n-2)/2, partition the items within a level
                    // and execute each chunk asynchronously. Each level must synchronize
                    // before continuing up, loop will iterate over levels not items
                    for(dtype start = (n-2)/2; start > 0;
                        start = (dtype)pow(2, (dtype)log2(start)) - 2) {
                        dtype end_exclusive = (dtype)pow(2, (dtype)log2(start))-2;

                        // The amount of work for this level
                        std::size_t items = (start-end_exclusive);

                        // TO-DO: determine the best solution for when chunk_size becomes
                        // too large as the heap shrinks
                        if(chunk_size > items)
                            chunk_size = items / 2;

                        // Sift down the nodes children and reposition 
                        std::size_t cnt = 0;
                        while(cnt + chunk_size < items) {
                            // Perform sift_down_range on each chunk
                            auto op = 
                                hpx::util::bind(
                                    &sift_down_range<RndIter, Pred&>, first,
                                    std::forward<Pred&>(pred), n, first + start - cnt,
                                    chunk_size);

                            workitems.push_back(
                                executor_traits::async_execute(policy.executor(), op));
                            
                            cnt += chunk_size;
                        }

                        if(cnt < items) {
                            // Perform sift_down_range on remaining items
                            auto op =
                                hpx::util::bind(
                                    &sift_down_range<RndIter, Pred&>, first,
                                    std::forward<Pred&>(pred), n, first + start - cnt,
                                    items-cnt);
     
                            workitems.push_back(
                                executor_traits::async_execute(policy.executor(), op));
                            
                        }
                        // Synchronize level
                        hpx::wait_all(workitems);
                    }
                    // Sift down the first element synchronously
                    sift_down_range(first, std::forward<Pred>(pred), n, first, 1);
                } catch(std::bad_alloc const&) {
                    return hpx::make_exceptional_future<void>(
                        boost::current_exception());
                } catch(...) {
                    util::detail::handle_local_exceptions<parallel_task_execution_policy>::call(
                            boost::current_exception(), errors);
                }
              
                // Perform local exception handling within a dataflow, because otherwise the
                // exception would be thrown outside of the future which is not the desired
                // behavior
                return hpx::lcos::local::dataflow(
                    [errors](std::vector<hpx::future<void> > && r) 
                        mutable
                    {
                        util::detail::handle_local_exceptions<parallel_task_execution_policy>
                            ::call(r, errors);
                    },
                    workitems);
            }
        };
    }

    template <typename ExPolicy, typename RndIter, typename Pred>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, void>::type
    >::type
    make_heap(ExPolicy && policy, RndIter first, RndIter last, Pred && pred)
    {
        typedef typename std::iterator_traits<RndIter>::iterator_category
            iterator_category;
        static_assert(
                (boost::is_base_of<
                 std::random_access_iterator_tag, iterator_category
                    >::value),
                "Requires random access iterator.");

        typedef typename is_sequential_execution_policy<ExPolicy>::type is_seq;
        typedef typename std::iterator_traits<RndIter>::value_type value_type;

        return detail::make_heap().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last,
                std::forward<Pred>(pred));
    }

    template <typename ExPolicy, typename RndIter>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, void>::type
    >::type
    make_heap(ExPolicy && policy, RndIter first, RndIter last)
    {
        typedef typename std::iterator_traits<RndIter>::iterator_category
            iterator_category;
        static_assert(
                (boost::is_base_of<
                 std::random_access_iterator_tag, iterator_category
                    >::value),
                "Requires random access iterator.");

        typedef typename is_sequential_execution_policy<ExPolicy>::type is_seq;
        typedef typename std::iterator_traits<RndIter>::value_type value_type;

        return detail::make_heap().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last,
                std::less<value_type>());
    }
}}}
#endif
