//  Copyright (c) 2015 John Biddiscombe
//  Copyright (c) 2015 Hartmut Kaiser
//  Copyright (c) 2015 Francisco José Tapia
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_ALGORITHM_SORT_OCT_2015)
#define HPX_PARALLEL_ALGORITHM_SORT_OCT_2015

#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/is_negative.hpp>
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

        ///---------------------------------------------------------------------------
        /// @struct iter_value
        /// @brief This is to obtain the type of data pointed to by an iterator
        /// @tparam RandomIt type of iterator
        /// @remarks
        //----------------------------------------------------------------------------
        template <typename RandomIt>
        struct iter_value
        {
            typedef typename
                std::remove_reference<decltype(*(std::declval<RandomIt>()))>::type type;
        };
/*
        //---------------------------------------------------------------------------
        /// @typename less_ptr_no_null
        ///
        /// @remarks this is the comparison object for pointers. Receive an object
        ///          for to compare the objects pointed. The pointers can't be nullptr
        //---------------------------------------------------------------------------
        template < typename RandomIt,
                   typename Comp = std::less<typename iter_value<RandomIt>::type> >
        struct less_ptr_no_null
        {
            Comp comp;
            inline less_ptr_no_null (Comp C1 = Comp()) : comp(C1) {};
            inline bool operator ()(RandomIt  T1,  RandomIt  T2) const
            {
                return comp(*T1 ,*T2);
            };
        };
*/
        ///---------------------------------------------------------------------------
        /// @struct parallel_sort_comp
        /// @brief implement the parallel sort using the intro_sort algorithm
        /// @tparam RandomIt : iterators pointing to the elements
        /// @tparam compare : objects for to compare two elements pointed by RandomIt
        ///                   iterators
        /// @remarks
        //----------------------------------------------------------------------------
        template < typename RandomIt,
                   typename Comp = std::less <typename iter_value<RandomIt>::type> >
        struct parallel_sort_comp
        {
            // These are not optimal on all machines,
            // we require a way to change them according to policy etc.
            size_t MaxPerThread = 65536;
            size_t Min_Parallel = 65536;
            Comp   comp;

            parallel_sort_comp(Comp && op) : comp(op)
            {}

            parallel_sort_comp() : comp(Comp())
            {}

            //------------------------------------------------------------------------
            //  function : parallel_sort_comp
            /// @brief constructor of the struct
            /// @param [in] first : iterator to the first element to sort
            /// @param [in] last : iterator to the next element after the last
            /// @param [in] comp : object for to compare
            /// @exception
            /// @return
            /// @remarks
            //------------------------------------------------------------------------
            hpx::future<void> parallel_sort_async(RandomIt first, RandomIt last)
            {
                size_t N = last - first;
                assert ( N >=0);

                if ( (size_t)N < Min_Parallel )
                {
                    std::sort(first, last, comp);
                    return hpx::make_ready_future();
                };

                // check if already sorted
                bool SW = true;
                for ( RandomIt it1 = first, it2 = first+1;
                    it2 != last && (SW = !comp(*it2,*it1));it1 = it2++);
                if (SW) return hpx::make_ready_future();

                return hpx::async(&parallel_sort_comp::sort_thread, this, first, last);;
            };

            void parallel_sort(RandomIt first, RandomIt last)
            {
                parallel_sort_async(first, last).get();
            }

            //------------------------------------------------------------------------
            //  function : sort_thread
            /// @brief this function is the work assigned to each thread in the parallel
            ///        process
            /// @exception
            /// @return
            /// @remarks
            //------------------------------------------------------------------------
            hpx::future<void> sort_thread(RandomIt first, RandomIt last)
            {
                using hpx::lcos::local::dataflow;

                //------------------------- begin ----------------------
                size_t N = last - first;
                if (N <= MaxPerThread) {
                    return hpx::async(
                        [this, first, last]() {
                            std::sort(first, last, comp);
                        }
                    );
                };

                //----------------------------------------------------------------
                //                     split
                //----------------------------------------------------------------
                typedef typename iter_value<RandomIt>::type value_t;

                //------------------- check if sort ------------------------------
                bool SW = true;
                for ( RandomIt it1 = first, it2 = first+1;
                        it2 != last && (SW = !comp(*it2,*it1));it1 = it2++);
                if (SW)
                {
                    return hpx::make_ready_future();
                };
                //---------------------- pivot select ----------------------------
                size_t Nx = ( size_t (N ) >>1 );

                RandomIt itA = first +1;
                RandomIt itB = first + Nx;
                RandomIt itC = last -1;

                if ( comp( *itB , *itA )) std::swap ( *itA, *itB);
                if ( comp (*itC , *itB))
                {
                    std::swap (*itC , *itB );
                    if ( comp( *itB , *itA )) std::swap ( *itA, *itB);
                };
                std::swap ( *first , *itB);
                value_t &  val = const_cast < value_t &>(* first);
                RandomIt c_first = first+2 , c_last  = last-2;

                while ( c_first != last && comp (*c_first, val)) ++c_first;
                while ( comp(val ,*c_last ) ) --c_last;
                while (!( c_first > c_last ))
                {
                    std::swap ( *(c_first++), *(c_last--));
                    while ( comp(*c_first,val) ) ++c_first;
                    while ( comp(val, *c_last) ) --c_last;
                }; // End while
                std::swap ( *first , *c_last);

                // spawn tasks for each sub section
                hpx::future<void> hk1 = hpx::async(&parallel_sort_comp::sort_thread, this, first, c_last);
                hpx::future<void> hk2 = hpx::async(&parallel_sort_comp::sort_thread, this, c_first, last);
                return dataflow(
                [] (future<void> f1, future<void> f2) -> void
                    {
                        f1.get();
                        f2.get();
                        return;
                    }, std::move(hk1), std::move(hk2)
                );
             }
         };

        ///////////////////////////////////////////////////////////////////////////
        // sort

        template <typename RandomIt>
        struct sort : public detail::algorithm<sort<RandomIt>, void>
        {
            sort()
              : sort::algorithm("sort")
            {}

            template <
                typename ExPolicy,
                typename Comp = std::less<typename iter_value<RandomIt>::type>,
                typename Proj = util::projection_identity >
            static hpx::util::unused_type
            sequential(ExPolicy, RandomIt first, RandomIt last, Comp && comp = Comp(),
                Proj && proj = Proj())
            {
                std::sort(first, last, std::forward<Comp>(comp));
                return hpx::util::unused;
            }

            template <
                typename ExPolicy,
                typename Comp = std::less<typename iter_value<RandomIt>::type>,
                typename Proj = util::projection_identity>
            static typename util::detail::algorithm_result<ExPolicy>::type
            parallel(ExPolicy policy, RandomIt first, RandomIt last, Comp && comp = Comp(),
                Proj && proj = Proj())
            {
                // contruct a parallel sort struct and set the comparison object
                parallel_sort_comp<RandomIt, Comp> sorter(std::forward<Comp>(comp));
                // call the sort routine and return the right type, depending on execution policy
                return util::detail::algorithm_result<ExPolicy>::get(
                    sorter.parallel_sort_async(first, last));
            }
        };
    /// \endcond
    }

    //-----------------------------------------------------------------------------
    /// Sorts elements from first - last in order according to the comparison
    /// operator. This algorithm is not stable.
    ///
    /// \note The algorithm is recursive divide && conquer based on intro sort
    /// with an average
    ///
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized && the manner
    ///                     in which it executes the assignments.
    /// \tparam RandomIt      The type of the iterators used for the
    ///                     input range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     input iterator.
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param comp         comparison operator used to order the elements pointed by RandomIt
    ///                     iterators
    /// @exception
    /// @return
    /// @remarks
    //-----------------------------------------------------------------------------
    template <typename ExPolicy, 
              typename RandomIt, 
              typename Comp = std::less<typename detail::iter_value<RandomIt>::type > >
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, void>::type
    >::type
    sort(ExPolicy && policy, RandomIt first, RandomIt last, Comp && comp = Comp())
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
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, comp);
    }

}; }; };

#endif
