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
            typedef typename std::iterator_traits<RndIter>::difference_type
                difference_type;
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
                typedef typename hpx::util::tuple<RndIter, std::size_t>
                    tuple_type;

                dtype n = last - first;

                if(n <= 1) {
                    return util::detail::algorithm_result<ExPolicy>::get();
                }


                std::size_t chunk_size = 0;

                std::vector<hpx::future<void> > workitems;
                std::vector<tuple_type> shape;
                std::list<boost::exception_ptr> errors;

                using namespace hpx::util::placeholders;
                auto op = hpx::util::bind(
                    &sift_down_range<RndIter, Pred&>, first,
                    std::forward<Pred&>(pred), (std::size_t)n,
                    _1, _2);

                try{
                    // Get workitems that are to be run in parallel
                    shape = util::detail::get_bottomup_heap_bulk_iteration_shape(
                        policy, workitems, op,
                        first, (std::size_t)n, chunk_size);

                    using hpx::util::get;
                    for(auto &iteration: shape) {
                        // Chunk up range of each iteration and execute asynchronously
                        RndIter begin = get<0>(iteration);
                        std::size_t length = get<1>(iteration);
                        while(length != 0) {
                            std::size_t chunk = (std::min)(chunk_size, length);
                            auto f1 = hpx::util::bind(
                                &sift_down_range<RndIter, Pred&>, first,
                                std::forward<Pred&>(pred), (std::size_t)n, begin,
                                chunk);
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
                return util::detail::algorithm_result<ExPolicy>::get();
            }

            template<typename RndIter, typename Pred>
            static typename util::detail::algorithm_result<
                parallel_task_execution_policy>::type
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
                typedef typename hpx::util::tuple<RndIter, std::size_t>
                    tuple_type;

                dtype n = last - first;

                if(n <= 1) {
                    return util::detail::algorithm_result<
                        parallel_task_execution_policy>::get();
                }


                std::size_t chunk_size = 0;

                std::vector<hpx::future<void> > workitems;
                std::vector<tuple_type> shape;
                std::list<boost::exception_ptr> errors;

                using namespace hpx::util::placeholders;
                auto op = hpx::util::bind(
                    &sift_down_range<RndIter, Pred&>, first,
                    std::forward<Pred&>(pred), (std::size_t)n,
                    _1, _2);

                try{
                    // Get workitems that are to be run in parallel
                    shape = util::detail::get_bottomup_heap_bulk_iteration_shape(
                        policy, workitems, op,
                        first, (std::size_t)n, chunk_size);

                    using hpx::util::get;
                    for(auto &iteration: shape) {
                        // Chunk up range of each iteration and execute asynchronously
                        RndIter begin = get<0>(iteration);
                        std::size_t length = get<1>(iteration);
                        while(length != 0) {
                            std::size_t chunk = (std::min)(chunk_size, length);
                            auto f1 = hpx::util::bind(
                                &sift_down_range<RndIter, Pred&>, first,
                                std::forward<Pred&>(pred), (std::size_t)n, begin,
                                chunk);
                            workitems.push_back(executor_traits::async_execute(
                                policy.executor(), f1));
                            length -= chunk;
                            std::advance(begin, chunk);
                        }
                        hpx::wait_all(workitems);
                    }

                } catch(std::bad_alloc const&) {
                    return hpx::make_exceptional_future<void>(
                        boost::current_exception());
                } catch(...) {
                    util::detail::handle_local_exceptions<
                        parallel_task_execution_policy>::call(
                            boost::current_exception(), errors);
                }

                // Perform local exception handling within a dataflow,
                // because otherwise the exception would be thrown outside
                // of the future which is not the desired behavior
                return hpx::lcos::local::dataflow(
                    [errors](std::vector<hpx::future<void> > && r)
                        mutable
                    {
                        util::detail::handle_local_exceptions<
                        parallel_task_execution_policy>
                            ::call(r, errors);
                    },
                    workitems);
            }
        };
    }

    /// Constructs a \a max \a heap in the range [first, last).
    ///
    /// \note Complexity: at most (3*N) comparisons where
    ///       \a N = distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution of
    ///                     the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam RndIter     The type of the source iterators used for algorithm.
    ///                     This iterator must meet the requirements for a
    ///                     random access iterator.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of that the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     that the algorithm will be applied to.
    /// \param pred         Refers to the binary predicate which returns true
    ///                     if the first argument should be treated as less than
    ///                     the second. The signature of the function should be
    ///                     equivalent to
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that objects of
    ///                     types \a RndIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// The predicate operations in the parallel \a make_heap algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// executes in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a make_heap algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a make_heap algorithm returns a \a hpx::future<void>
    ///           if the execution policy is of type \a task_execution_policy
    ///           and returns \a void otherwise.
    ///
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

    /// Constructs a \a max \a heap in the range [first, last). Uses the
    /// operator \a < for comparisons.
    ///
    /// \note Complexity: at most (3*N) comparisons where
    ///       \a N = distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution of
    ///                     the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam RndIter     The type of the source iterators used for algorithm.
    ///                     This iterator must meet the requirements for a
    ///                     random access iterator.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of that the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     that the algorithm will be applied to.
    ///
    /// The predicate operations in the parallel \a make_heap algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// executes in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a make_heap algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a make_heap algorithm returns a \a hpx::future<void>
    ///           if the execution policy is of type \a task_execution_policy
    ///           and returns \a void otherwise.
    ///
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
