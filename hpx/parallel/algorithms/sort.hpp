//  Copyright (c) 2015 John Biddiscombe
//  Copyright (c) 2015 Hartmut Kaiser
//  Copyright (c) 2015 Francisco Jose Tapia
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_ALGORITHM_SORT_OCT_2015)
#define HPX_PARALLEL_ALGORITHM_SORT_OCT_2015

#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/dataflow.hpp>

#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/util/compare_projected.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/traits/projected.hpp>

#include <boost/exception_ptr.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>
#include <list>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // sort
    namespace detail
    {
        /// \cond NOINTERNAL
        static const std::size_t sort_limit_per_task = 65536ul;

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename R>
        struct handle_sort_exception
        {
            static hpx::future<R> call(hpx::future<R> f)
            {
                HPX_ASSERT(f.has_exception());

                // Intel complains if this is not explicitly moved
                return std::move(f);
            }

            static hpx::future<R> call(boost::exception_ptr const& e)
            {
                try {
                    boost::rethrow_exception(e);
                }
                catch (std::bad_alloc const&) {
                    // rethrow bad_alloc
                    return hpx::make_exceptional_future<R>(
                        boost::current_exception());
                }
                catch (...) {
                    // package up everything else as an exception_list
                    return hpx::make_exceptional_future<R>(
                        exception_list(e));
                }
            }
        };

        template <typename R>
        struct handle_sort_exception<parallel_vector_execution_policy, R>
        {
            HPX_ATTRIBUTE_NORETURN
            static hpx::future<R> call(hpx::future<void> &&)
            {
                hpx::terminate();
            }

            HPX_ATTRIBUTE_NORETURN
            static hpx::future<R> call(boost::exception_ptr const&)
            {
                hpx::terminate();
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // std::is_sorted is not available on all supported platforms yet
        template <typename Iter, typename Compare>
        inline bool is_sorted_sequential(Iter first, Iter last, Compare comp)
        {
            bool sorted = true;
            if (first != last)
            {
                for (Iter it1 = first, it2 = first + 1;
                     it2 != last && (sorted = !comp(*it2, *it1));
                     it1 = it2++)
                {
                    /**/
                }
            }
            return sorted;
        }

        //------------------------------------------------------------------------
        //  function : sort_thread
        /// \brief this function is the work assigned to each thread in the
        ///        parallel process
        /// \exception
        /// \return
        /// \remarks
        //------------------------------------------------------------------------
        template <typename ExPolicy, typename RandomIt, typename Compare>
        hpx::future<RandomIt> sort_thread(ExPolicy& policy,
            RandomIt first, RandomIt last, Compare comp)
        {
            typedef typename hpx::util::decay<ExPolicy>::type::executor_type
                executor_type;
            typedef typename hpx::parallel::executor_traits<executor_type>
                executor_traits;

            //------------------------- begin ----------------------
            std::ptrdiff_t N = last - first;
            if (std::size_t(N) <= sort_limit_per_task)
            {
                return executor_traits::async_execute(
                    policy.executor(),
                    [first, last, comp]() -> RandomIt
                    {
                        std::sort(first, last, comp);
                        return last;
                    });
            }

            //----------------------------------------------------------------
            //                     split
            //----------------------------------------------------------------

            //------------------- check if sorted ----------------------------
            if (detail::is_sorted_sequential(first, last, comp))
                return hpx::make_ready_future(last);

            //---------------------- pivot select ----------------------------
            std::size_t nx = std::size_t(N) >> 1;

            RandomIt it_a = first + 1;
            RandomIt it_b = first + nx;
            RandomIt it_c = last - 1;

            if (comp(*it_b, *it_a))
                std::iter_swap(it_a, it_b);

            if (comp(*it_c, *it_b))
            {
                std::iter_swap(it_c, it_b);
                if (comp(*it_b, *it_a))
                    std::iter_swap(it_a, it_b);
            }

            std::iter_swap(first, it_b);

            typedef
                typename std::iterator_traits<RandomIt>::reference
                reference;

            reference val = *first;
            RandomIt c_first = first + 2, c_last = last - 2;

            while (c_first != last && comp(*c_first, val))
                ++c_first;
            while (comp(val, *c_last))
                --c_last;
            while (!(c_first > c_last))
            {
                std::iter_swap(c_first++, c_last--);
                while (comp(*c_first, val))
                    ++c_first;
                while (comp(val, *c_last))
                    --c_last;
            } // End while

            std::iter_swap(first, c_last);

            // spawn tasks for each sub section
            hpx::future<RandomIt> left =
                executor_traits::async_execute(
                    policy.executor(),
                    hpx::util::bind(
                        &sort_thread<ExPolicy, RandomIt, Compare>,
                        std::ref(policy), first, c_last, comp
                    ));

            hpx::future<RandomIt> right =
                executor_traits::async_execute(
                    policy.executor(),
                    hpx::util::bind(
                        &sort_thread<ExPolicy, RandomIt, Compare>,
                        std::ref(policy), c_first, last, comp
                    ));

            return hpx::dataflow(
                [last](hpx::future<RandomIt> && left,
                    hpx::future<RandomIt> && right) -> RandomIt
                {
                    if (left.has_exception() || right.has_exception())
                    {
                        std::list<boost::exception_ptr> errors;
                        if (left.has_exception())
                            errors.push_back(left.get_exception_ptr());
                        if (right.has_exception())
                            errors.push_back(right.get_exception_ptr());

                        boost::throw_exception(
                            exception_list(std::move(errors)));
                    }
                    return last;
                },
                std::move(left), std::move(right));
        }

        //------------------------------------------------------------------------
        //  function : parallel_sort_async
        //------------------------------------------------------------------------
        /// @param [in] first : iterator to the first element to sort
        /// @param [in] last : iterator to the next element after the last
        /// @param [in] comp : object for to compare
        /// @exception
        /// @return
        /// @remarks
        template <typename ExPolicy, typename RandomIt, typename Compare>
        hpx::future<RandomIt>
        parallel_sort_async(ExPolicy && policy, RandomIt first, RandomIt last,
            Compare comp)
        {
            hpx::future<RandomIt> result;
            try {
                std::ptrdiff_t N = last - first;
                HPX_ASSERT(N >= 0);

                if (std::size_t(N) < sort_limit_per_task)
                {
                    std::sort(first, last, comp);
                    return hpx::make_ready_future(last);
                }

                // check if already sorted
                if (detail::is_sorted_sequential(first, last, comp))
                    return hpx::make_ready_future(last);

                typedef typename hpx::util::decay<ExPolicy>::type::executor_type
                    executor_type;
                typedef typename hpx::parallel::executor_traits<executor_type>
                    executor_traits;

                result = executor_traits::async_execute(
                    policy.executor(),
                    hpx::util::bind(
                        &sort_thread<ExPolicy, RandomIt, Compare>,
                        std::ref(policy), first, last, comp
                    ));
            }
            catch (...) {
                return detail::handle_sort_exception<ExPolicy, RandomIt>::call(
                    boost::current_exception());
            }

            if (result.has_exception())
            {
                return detail::handle_sort_exception<ExPolicy, RandomIt>::call(
                    std::move(result));
            }

            return result;
        }

        ///////////////////////////////////////////////////////////////////////
        // sort
        template <typename RandomIt>
        struct sort : public detail::algorithm<sort<RandomIt>, RandomIt>
        {
            sort()
              : sort::algorithm("sort")
            {}

            template <typename ExPolicy, typename Compare, typename Proj>
            static RandomIt
            sequential(ExPolicy, RandomIt first, RandomIt last,
                Compare && comp, Proj && proj)
            {
                std::sort(first, last,
                    util::compare_projected<Compare, Proj>(
                            std::forward<Compare>(comp),
                            std::forward<Proj>(proj)
                        ));
                return last;
            }

            template <typename ExPolicy, typename Compare, typename Proj>
            static typename util::detail::algorithm_result<
                ExPolicy, RandomIt
            >::type
            parallel(ExPolicy && policy, RandomIt first, RandomIt last,
                Compare && comp, Proj && proj)
            {
                // call the sort routine and return the right type,
                // depending on execution policy
                return util::detail::algorithm_result<ExPolicy, RandomIt>::get(
                    parallel_sort_async(std::forward<ExPolicy>(policy),
                        first, last,
                        util::compare_projected<Compare, Proj>(
                            std::forward<Compare>(comp),
                            std::forward<Proj>(proj)
                        )));
            }
        };
        /// \endcond
    }

    //-----------------------------------------------------------------------------
    /// Sorts the elements in the range [first, last) in ascending order. The
    /// order of equal elements is not guaranteed to be preserved. The function
    /// uses the given comparison function object comp (defaults to using
    /// operator<()).
    ///
    /// \note   Complexity: O(Nlog(N)), where N = std::distance(first, last)
    ///                     comparisons.
    ///
    /// A sequence is sorted with respect to a comparator \a comp and a
    /// projection \a proj if for every iterator i pointing to the sequence and
    /// every non-negative integer n such that i + n is a valid iterator
    /// pointing to an element of the sequence, and
    /// INVOKE(comp, INVOKE(proj, *(i + n)), INVOKE(proj, *i)) == false.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam Iter        The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param comp         comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that comp
    ///                     will not apply any non-constant function through the
    ///                     dereferenced iterator.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each pair of elements as a
    ///                     projection operation before the actual predicate
    ///                     \a comp is invoked.
    ///
    /// \a comp has to induce a strict weak ordering on the values.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequential_execution_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_execution_policy or \a parallel_task_execution_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a sort algorithm returns a
    ///           \a hpx::future<RandomIt> if the execution policy is of
    ///           type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and returns \a RandomIt
    ///           otherwise.
    ///           The algorithm returns an iterator pointing to the first
    ///           element after the last element in the input sequence.
    //-----------------------------------------------------------------------------
    template <typename ExPolicy, typename RandomIt,
        typename Proj = util::projection_identity,
        typename Compare = std::less<
            typename std::remove_reference<
                typename traits::projected_result_of<Proj, RandomIt>::type
            >::type
        >,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<RandomIt>::value &&
        traits::is_projected<Proj, RandomIt>::value &&
        traits::is_indirect_callable<
            Compare,
                traits::projected<Proj, RandomIt>,
                traits::projected<Proj, RandomIt>
        >::value)>
    typename util::detail::algorithm_result<ExPolicy, RandomIt>::type
    sort(ExPolicy && policy, RandomIt first, RandomIt last,
        Compare && comp = Compare(), Proj && proj = Proj())
    {
        static_assert(
            (hpx::traits::is_random_access_iterator<RandomIt>::value),
            "Requires a random access iterator.");

        typedef is_sequential_execution_policy<ExPolicy> is_seq;

        return detail::sort<RandomIt>().call(
            std::forward<ExPolicy>(policy), is_seq(), first, last,
            std::forward<Compare>(comp), std::forward<Proj>(proj));
    }
}}}

#endif
