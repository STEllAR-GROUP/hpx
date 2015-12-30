//  Copyright (c) 2015 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/is_heap.hpp

#if !defined(HPX_PARALLEL_ALGORITHMS_IS_HEAP_DEC_15_2015_1012AM)
#define HPX_PARALLEL_ALGORITHMS_IS_HEAP_DEC_15_2015_1012AM

#include <hpx/lcos/local/dataflow.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/unwrapped.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/detail/chunk_size.hpp>

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
    ///////////////////////////////////////////////////////////////
    // is_heap_until
    namespace detail {

        template <typename RndIter, typename Pred>
        void comp_heap(RndIter first, Pred && pred,
                typename std::iterator_traits<RndIter>::difference_type len,
                RndIter start,
                typename std::iterator_traits<RndIter>::difference_type n,
                util::cancellation_token<std::size_t> &tok)
        {
            typedef typename std::iterator_traits<RndIter>::difference_type
                dtype;

            dtype p = std::distance(first, start);
            dtype c = 2 * p + 1;
            RndIter pp = start;
            while(c < len && n > 0) {
                if(tok.was_cancelled(c)) {
                    break;
                }
                RndIter cp = first + c;
                if(pred(*pp, *cp)) {
                    tok.cancel(c);
                    break;
                }
                ++c;
                ++cp;
                if( c <= 0 )
                    break;
                if(pred(*pp, *cp)) {
                    tok.cancel(c);
                    break;
                }
                ++p;
                ++pp;
                --n;
                c = 2 * p + 1;
            }
        }

        /// \cond NOINTERNAL
        template <typename RndIter>
        struct is_heap_until:
            public detail::algorithm<is_heap_until<RndIter>, RndIter>
        {
            is_heap_until()
                : is_heap_until::algorithm("is_heap_until")
            {}

            template <typename ExPolicy, typename Pred>
            static RndIter
            sequential(ExPolicy, RndIter first, RndIter last,
                    Pred && pred)
            {
                return
                    std::is_heap_until(first, last, std::forward<Pred>(pred));
            }

            template <typename ExPolicy, typename Pred>
            static typename util::detail::algorithm_result<
                ExPolicy, RndIter
            >::type
            parallel(ExPolicy policy, RndIter first, RndIter last,
                Pred && pred)
            {
                typedef typename ExPolicy::executor_type executor_type;
                typedef typename hpx::parallel::executor_traits<executor_type>
                    executor_traits;
                typedef typename std::iterator_traits<RndIter>::difference_type
                    dtype;
                typedef typename util::detail::algorithm_result<ExPolicy,
                    RndIter> result;
                typedef typename hpx::util::tuple<RndIter, std::size_t>
                    tuple_type;

                std::size_t len = last - first;

                if(len <= 1) {
                    return result::get(std::move(last));
                }

                // Manually specify a chunk_size, which will be overridden in
                // get_topdown_heap_bulk_iteration_shape
                std::size_t chunk_size = 0;

                std::vector<hpx::future<void> > workitems;
                std::vector<tuple_type> shape;
                util::cancellation_token<std::size_t> tok(len);
                std::list<boost::exception_ptr> errors;

                using namespace hpx::util::placeholders;
                auto op = hpx::util::bind(
                        &comp_heap<RndIter, Pred&>, first,
                        std::forward<Pred&>(pred), len, _1,
                        _2, tok);

                try {
                    // Get workittems that are to be run in parallel
                    shape = util::detail::get_topdown_heap_bulk_iteration_shape(
                        policy, workitems, op,
                        first, std::distance(first,last), chunk_size);

                    using hpx::util::get;
                    for(auto &iteration: shape) {
                        // Chunk up range of each iteration and execute asynchronously
                        RndIter begin = get<0>(iteration);
                        std::size_t length = get<1>(iteration);
                        while(length != 0) {
                            std::size_t chunk = (std::min)(chunk_size, length);
                            auto f1 = hpx::util::bind(
                                &comp_heap<RndIter, Pred&>, first,
                                std::forward<Pred&>(pred), len, begin,
                                chunk, tok);
                            workitems.push_back(executor_traits::async_execute(
                                policy.executor(), f1));
                            length -= chunk;
                            std::advance(begin, chunk);
                        }
                        hpx::wait_all(workitems);
                    }
                } catch(...) {
                    util::detail::handle_local_exceptions<ExPolicy>::call(
                        boost::current_exception(), errors);
                }

                util::detail::handle_local_exceptions<ExPolicy>::call(
                    workitems, errors);

                std::size_t pos = tok.get_data();
                if(pos != len)
                    std::advance(first, pos);
                else
                    first = last;

                return result::get(std::move(first));
            }

            template <typename Pred>
            static typename util::detail::algorithm_result<
                parallel_task_execution_policy, RndIter
            >::type
            parallel(parallel_task_execution_policy policy,
                RndIter first, RndIter last,
                Pred && pred)
            {
                typedef typename parallel_task_execution_policy::executor_type
                    executor_type;
                typedef typename hpx::parallel::executor_traits<executor_type>
                    executor_traits;
                typedef typename std::iterator_traits<RndIter>::difference_type
                    dtype;
                typedef typename util::detail::algorithm_result<
                    parallel_task_execution_policy, RndIter> result;
                typedef typename hpx::util::tuple<RndIter, std::size_t>
                    tuple_type;

                std::size_t len = last - first;

                if(len <= 1) {
                    return result::get(std::move(last));
                }

                // Manually specify a chunk_size, which will be overridden in
                // get_topdown_heap_bulk_iteration_shape
                std::size_t chunk_size = 0;

                std::vector<hpx::future<void> > workitems;
                std::vector<tuple_type> shape;
                util::cancellation_token<std::size_t> tok(len);
                std::list<boost::exception_ptr> errors;

                using namespace hpx::util::placeholders;
                auto op = hpx::util::bind(
                        &comp_heap<RndIter, Pred&>, first,
                        std::forward<Pred&>(pred), len, _1,
                        _2, tok);

                try {
                    // Get workittems that are to be run in parallel
                    shape = util::detail::get_topdown_heap_bulk_iteration_shape(
                        policy, workitems, op,
                        first, std::distance(first,last), chunk_size);

                    using hpx::util::get;
                    for(auto &iteration: shape) {
                        // Chunk up range of each iteration and execute asynchronously
                        RndIter begin = get<0>(iteration);
                        std::size_t length = get<1>(iteration);
                        while(length != 0) {
                            std::size_t chunk = (std::min)(chunk_size, length);
                            auto f1 = hpx::util::bind(
                                &comp_heap<RndIter, Pred&>, first,
                                std::forward<Pred&>(pred), len, begin,
                                chunk, tok);
                            workitems.push_back(executor_traits::async_execute(
                                policy.executor(), f1));
                            length -= chunk;
                            std::advance(begin, chunk);
                        }
                        hpx::wait_all(workitems);
                    }
                } catch(std::bad_alloc const&) {
                    return hpx::make_exceptional_future<RndIter>(
                            boost::current_exception());
                } catch(...) {
                    util::detail::handle_local_exceptions<
                        parallel_task_execution_policy>::call(
                            boost::current_exception(), errors);
                }

                return hpx::lcos::local::dataflow(
                    [=](std::vector<hpx::future<void> > && r)
                        mutable -> RndIter
                    {
                        util::detail::handle_local_exceptions<
                            parallel_task_execution_policy>::call(
                                r, errors);

                        std::size_t pos = tok.get_data();
                        if(pos != len)
                            std::advance(first, pos);
                        else
                            first = last;

                        return std::move(first);
                    },
                    workitems);
            }
        };
        /// \endcond
    }

    template <typename ExPolicy, typename RndIter, typename Pred>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, RndIter>::type
    >::type
    is_heap_until(ExPolicy && policy, RndIter first, RndIter last, Pred && pred)
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

        return detail::is_heap_until<RndIter>().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last,
                std::forward<Pred>(pred));
    }

    template <typename ExPolicy, typename RndIter>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, RndIter>::type
    >::type
    is_heap_until(ExPolicy && policy, RndIter first, RndIter last)
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

        return detail::is_heap_until<RndIter>().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last,
                std::less<value_type>());
    }

    namespace detail
    {
        /// \cond NOINTERNAL
        struct is_heap: public detail::algorithm<is_heap, bool>
        {
            is_heap()
                : is_heap::algorithm("is_heap")
            {}

            template <typename ExPolicy, typename RndIter, typename Pred>
            static bool
            sequential(ExPolicy, RndIter first, RndIter last, Pred && pred)
            {
                return std::is_heap(first, last, std::forward<Pred>(pred));
            }

            template <typename ExPolicy, typename RndIter, typename Pred>
            static typename util::detail::algorithm_result<ExPolicy, bool>::type
            parallel(ExPolicy policy, RndIter first, RndIter last,
                Pred && pred)
            {
                typedef typename util::detail::algorithm_result<ExPolicy, bool>
                    result;

                return result::get(
                    is_heap_until<RndIter>().call(
                    policy, boost::mpl::false_(),
                    first, last, std::forward<Pred>(pred)) == last);
            }

            template <typename RndIter, typename Pred>
            static typename util::detail::algorithm_result<
                parallel_task_execution_policy, bool>::type
            parallel(parallel_task_execution_policy policy,
                RndIter first, RndIter last,
                Pred && pred)
            {
                typedef typename util::detail::algorithm_result<
                    parallel_task_execution_policy, bool>
                    result;

                return is_heap_until<RndIter>().call(
                    policy, boost::mpl::false_(),
                    first, last, std::forward<Pred>(pred)).then(
                    [=](hpx::future<RndIter> f) -> bool
                    {
                        return f.get() == last;
                    });
            }
        };
    }

    template <typename ExPolicy, typename RndIter, typename Pred>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, bool>::type
    >::type
    is_heap(ExPolicy && policy, RndIter first, RndIter last, Pred && pred)
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

        return detail::is_heap().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last,
                std::forward<Pred>(pred));
    }

    template <typename ExPolicy, typename RndIter>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, bool>::type
    >::type
    is_heap(ExPolicy && policy, RndIter first, RndIter last)
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

        return detail::is_heap().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last,
                std::less<value_type>());
    }
}}}

#endif
