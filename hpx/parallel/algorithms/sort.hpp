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
#include <hpx/util/invoke.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/lcos/when_all.hpp>

#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/traits/projected.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // sort
    namespace detail
    {
        /// \cond NOINTERNAL
        static const std::size_t sort_limit_per_task = 65536ul;

        ///////////////////////////////////////////////////////////////////////
        template <typename Compare, typename Proj>
        struct compare_projected
        {
            template <typename Compare_, typename Proj_>
            compare_projected(Compare_ && comp, Proj_ && proj)
              : comp_(std::forward<Compare>(comp)),
                proj_(std::forward<Proj>(proj))
            {}

            template <typename T>
            inline bool operator()(T const& t1, T const& t2)
            {
                return hpx::util::invoke(comp_,
                    hpx::util::invoke(proj_, t1),
                    hpx::util::invoke(proj_, t2));
            }

            Compare comp_;
            Proj proj_;
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
        hpx::future<void> sort_thread(ExPolicy policy,
            RandomIt first, RandomIt last, Compare comp)
        {
            typedef typename ExPolicy::executor_type executor_type;
            typedef typename hpx::parallel::executor_traits<executor_type>
                executor_traits;

            //------------------------- begin ----------------------
            std::size_t N = last - first;
            if (N <= sort_limit_per_task)
            {
                return executor_traits::async_execute(
                    policy.executor(),
                    [first, last, comp]()
                    {
                        std::sort(first, last, comp);
                    });
            }

            //----------------------------------------------------------------
            //                     split
            //----------------------------------------------------------------

            //------------------- check if sorted ----------------------------
            if (detail::is_sorted_sequential(first, last, comp))
                return hpx::make_ready_future();

            //---------------------- pivot select ----------------------------
            std::size_t nx = N >> 1;

            RandomIt it_a = first + 1;
            RandomIt it_b = first + nx;
            RandomIt it_c = last - 1;

            if (comp(*it_b, *it_a))
                std::swap(*it_a, *it_b);
            if (comp(*it_c, *it_b))
            {
                std::swap(*it_c, *it_b);
                if (comp(*it_b, *it_a))
                    std::swap(*it_a, *it_b);
            }
            std::swap(*first, *it_b);

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
                std::swap(*(c_first++), *(c_last--));
                while (comp(*c_first, val))
                    ++c_first;
                while (comp(val, *c_last))
                    --c_last;
            } // End while
            std::swap(*first, *c_last);

            // spawn tasks for each sub section
            hpx::future<void> left =
                executor_traits::async_execute(
                    policy.executor(),
                    hpx::util::bind(
                        &sort_thread<ExPolicy, RandomIt, Compare>,
                        policy, first, c_last, comp
                    ));

            hpx::future<void> right =
                executor_traits::async_execute(
                    policy.executor(),
                    hpx::util::bind(
                        &sort_thread<ExPolicy, RandomIt, Compare>,
                        policy, c_first, last, comp
                    ));

            return hpx::when_all(std::move(left), std::move(right));
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
        hpx::future<void>
        parallel_sort_async(ExPolicy policy, RandomIt first, RandomIt last,
            Compare comp)
        {
            size_t N = last - first;
            HPX_ASSERT(N >= 0);

            if (N < sort_limit_per_task)
            {
                std::sort(first, last, comp);
                return hpx::make_ready_future();
            }

            // check if already sorted
            if (detail::is_sorted_sequential(first, last, comp))
                return hpx::make_ready_future();

            typedef typename ExPolicy::executor_type executor_type;
            typedef typename hpx::parallel::executor_traits<executor_type>
                executor_traits;

            return executor_traits::async_execute(policy.executor(),
                hpx::util::bind(
                    &sort_thread<ExPolicy, RandomIt, Compare>,
                    policy, first, last, comp
                ));
        }

        ///////////////////////////////////////////////////////////////////////
        // sort
        template <typename RandomIt>
        struct sort : public detail::algorithm<sort<RandomIt>, void>
        {
            sort()
              : sort::algorithm("sort")
            {}

            template <typename ExPolicy, typename Compare, typename Proj>
            static hpx::util::unused_type
            sequential(ExPolicy, RandomIt first, RandomIt last,
                Compare && comp, Proj && proj)
            {
                std::sort(first, last,
                    compare_projected<Compare, Proj>(
                            std::forward<Compare>(comp),
                            std::forward<Proj>(proj)
                        ));
                return hpx::util::unused;
            }

            template <typename ExPolicy, typename Compare, typename Proj>
            static typename util::detail::algorithm_result<ExPolicy>::type
            parallel(ExPolicy policy, RandomIt first, RandomIt last,
                Compare && comp, Proj && proj)
            {
                // call the sort routine and return the right type,
                // depending on execution policy
                return util::detail::algorithm_result<ExPolicy>::get(
                    parallel_sort_async(policy, first, last,
                        compare_projected<Compare, Proj>(
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
    ///           \a hpx::future<Iter> if the execution policy is of
    ///           type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and returns \a Iter
    ///           otherwise.
    ///           It returns \a last.
    //-----------------------------------------------------------------------------
    template <typename Proj = util::projection_identity,
        typename ExPolicy, typename RandomIt,
        typename Compare = std::less<
            typename std::remove_reference<
                typename traits::projected_result_of<Proj, RandomIt>::type
            >::type
        >,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        traits::detail::is_iterator<RandomIt>::value &&
        traits::is_projected<Proj, RandomIt>::value &&
        traits::is_indirect_callable<
            Compare,
                traits::projected<Proj, RandomIt>,
                traits::projected<Proj, RandomIt>
        >::value)>
    typename util::detail::algorithm_result<ExPolicy, void>::type
    sort(ExPolicy && policy, RandomIt first, RandomIt last,
        Compare && comp = Compare(), Proj && proj = Proj())
    {
        typedef typename std::iterator_traits<RandomIt>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::random_access_iterator_tag, iterator_category
            >::value),
            "Requires a random access iterator.");

        typedef is_sequential_execution_policy<ExPolicy> is_seq;

        return detail::sort<RandomIt>().call(
            std::forward<ExPolicy>(policy), is_seq(), first, last,
            std::forward<Compare>(comp), std::forward<Proj>(proj));
    }
}}}

#endif
