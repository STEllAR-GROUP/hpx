//  Copyright (c) 2015 John Biddiscombe
//  Copyright (c) 2015 Francisco José Tapia
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_ALGORITHM_SORT_OCT_2015)
#define HPX_PARALLEL_ALGORITHM_SORT_OCT_2015

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{

    ///////////////////////////////////////////////////////////////////////////
    // sort
    namespace detail
    {

    ///---------------------------------------------------------------------------
    /// @struct iter_value
    /// @brief This is to obtain the type of data pointed to by an iterator
    /// @tparam RandomIt type of iterator
    /// @remarks
    //----------------------------------------------------------------------------
    template <typename RandomIt>
    struct iter_value
    {   typedef typename
            std::remove_reference<decltype(*(std::declval<RandomIt>()))>::type type;
    };

    //---------------------------------------------------------------------------
    /// @class less_ptr_no_null
    ///
    /// @remarks this is the comparison object for pointers. Receive an object
    ///          for to compare the objects pointed. The pointers can't be nullptr
    //---------------------------------------------------------------------------
    template <  class RandomIt ,
                class compare =std::less<typename iter_value<RandomIt>::type>
            >
    struct less_ptr_no_null
    {   //----------------------------- Variables -----------------------
        compare comp ;
        //----------------------------- Functions ----------------------
        inline less_ptr_no_null (compare C1 = compare()) : comp(C1) {};
        inline bool operator ()(RandomIt  T1,  RandomIt  T2) const
        {
            return comp(*T1 ,*T2);
        };
    };

    ///---------------------------------------------------------------------------
    /// @struct parallel_sort_comp
    /// @brief implement the parallel sort using the intro_sort algorithm
    /// @tparam RandomIt : iterators pointing to the elements
    /// @tparam compare : objects for to compare two elements pointed by RandomIt
    ///                   iterators
    /// @remarks
    //----------------------------------------------------------------------------
    template < class RandomIt,
               typename compare = std::less <typename iter_value<RandomIt>::type>  >
    struct parallel_sort_comp
    {   //------------------------- begin ----------------------
        size_t MaxPerThread = 65536;
        size_t Min_Parallel = 65536;
        compare comp;

        //------------------------------------------------------------------------
        //  function : parallel_sort_comp
        /// @brief constructor
        /// @param [in] first : iterator to the first element to sort
        /// @param [in] last : iterator to the next element after the last
        /// @exception
        /// @return
        /// @remarks
        //------------------------------------------------------------------------
        parallel_sort_comp (RandomIt first, RandomIt last )
                            : parallel_sort_comp ( first,last, compare())
                            {} ;
        //
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
        parallel_sort_comp ( RandomIt first, RandomIt last, compare comp1)
          : comp(comp1)
        {
            size_t N = last - first;
            assert ( N >=0);

            if ( (size_t)N < Min_Parallel )
            {
                std::sort(first, last, comp);
                return ;
            } ;

            // check if already sorted
            bool SW = true ;
            for ( RandomIt it1 = first, it2 = first+1 ;
                it2 != last and (SW = not comp(*it2,*it1));it1 = it2++);
            if (SW) return ;

            hpx::future<void> dummy = hpx::async(&parallel_sort_comp::sort_thread, this, first, last);
            dummy.get();

        };

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
            size_t N = last - first ;
            if (N <= MaxPerThread) {
                return hpx::async(
                    [this, first, last](){
                    std::sort(first, last, comp);
                    }
                );
            };

            //----------------------------------------------------------------
            //                     split
            //----------------------------------------------------------------
            typedef typename iter_value<RandomIt>::type value_t ;

            //------------------- check if sort ------------------------------
            bool SW = true ;
            for ( RandomIt it1 = first, it2 = first+1 ;
                    it2 != last and (SW = not comp(*it2,*it1));it1 = it2++);
            if (SW)
            {
                return hpx::make_ready_future();
            };
            //---------------------- pivot select ----------------------------
            size_t Nx = ( size_t (N ) >>1 ) ;

            RandomIt itA = first +1 ;
            RandomIt itB = first + Nx ;
            RandomIt itC = last -1 ;

            if ( comp( *itB , *itA )) std::swap ( *itA, *itB);
            if ( comp (*itC , *itB))
            {
                std::swap (*itC , *itB );
                if ( comp( *itB , *itA )) std::swap ( *itA, *itB);
            };
            std::swap ( *first , *itB);
            value_t &  val = const_cast < value_t &>(* first);
            RandomIt c_first = first+2 , c_last  = last-2;

            while ( c_first != last and comp (*c_first, val)) ++c_first ;
            while ( comp(val ,*c_last ) ) --c_last ;
            while (not( c_first > c_last ))
            {
                std::swap ( *(c_first++), *(c_last--));
                while ( comp(*c_first,val) ) ++c_first;
                while ( comp(val, *c_last) ) --c_last ;
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


    /// \cond NOINTERNAL
    struct sort : public detail::algorithm<sort>
    {
        sort() : sort::algorithm("sort")
        {}

        template <typename ExPolicy, typename RandomIt, typename Op>
        static hpx::util::unused_type
        sequential(ExPolicy, RandomIt first, RandomIt last, Op comp)
        {
            std::sort(first, last, comp);
            return hpx::util::unused;
        }

        template <typename ExPolicy, typename RandomIt, typename Op>
        static typename util::detail::algorithm_result<ExPolicy>::type
        parallel(ExPolicy policy, RandomIt first, RandomIt last, Op comp)
        {
            typedef typename util::detail::algorithm_result<ExPolicy>::type
                    result_type;
            typedef typename std::iterator_traits<RandomIt>::value_type type;

            return hpx::util::void_guard<result_type>(),
                    parallel_sort_comp<RandomIt, Op>(first, last, comp);
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
    template <typename ExPolicy, typename RandomIt, typename Op>
    inline void sort_unstable (ExPolicy policy, RandomIt first, RandomIt last, Op comp)
    {
        detail::parallel_sort_comp<RandomIt, Op>(first, last, comp);
    };


    /// Sorts elements from first - last in order according to the comparison
    /// operator. This algorithm is not stable.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized && the manner
    ///                     in which it executes the assignments.
    /// \tparam RandomIt      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    /// \tparam Op          The type of the binary function object used for
    ///                     the reduction operation.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param init         The initial value for the generalized sum.
    /// \param op           Specifies the function (or function object) which
    ///                     will be invoked for each of the values of the input
    ///                     sequence. This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it.
    ///                     The types \a Type1 && \a Ret must be
    ///                     such that an object of a type as given by the input
    ///                     sequence can be implicitly converted to any
    ///                     of those types.
    ///
    /// The reduce operations in the parallel \a inclusive_scan algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a inclusive_scan algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, && indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy_n algorithm returns a \a hpx::future<OutIter> if
    ///           the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a OutIter otherwise.
    ///           The \a inclusive_scan algorithm returns the output iterator
    ///           to the element in the destination range, one past the last
    ///           element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aK),
    ///           GENERALIZED_NONCOMMUTATIVE_SUM(op, aM, ..., aN))
    ///           where 1 < K+1 = M <= N.
    ///
    /// The difference between \a exclusive_scan && \a inclusive_scan is that
    /// \a inclusive_scan includes the ith input element in the ith sum. If
    /// \a op is not mathematically associative, the behavior of
    /// \a inclusive_scan may be non-deterministic.
    ///

    //----------------------------------------------------------------------------
    //----------------------------------------------------------------------------

    //
    //-----------------------------------------------------------------------------
    //  function : introsort
    /// @brief this function implement a non stable sort, based internally in the
    ///        intro_sort algorithm. Run with 1 thread
    /// @tparam RandomIt : iterators for to access to the elements
    /// @tparam compare : object for to compare two elements pointed by the RandomIt
    /// @param [in] firts : iterator to the first element of the range to sort
    /// @param [in] last : iterator after the last element to the range to sort
    /// @param [in] comp : object for to compare two elements pointed by RandomIt
    ///                    iterators
    /// @exception
    /// @return
    /// @remarks
    //-----------------------------------------------------------------------------
    /*
template < class RandomIt,
           typename compare = std::less <typename iter_value<RandomIt>::type> >
inline void introsort ( RandomIt first, RandomIt last,compare comp = compare())
{   //---------------------------- begin -------------------------------------
    bs_algo::intro_sort(first, last, comp);
};
     */
    //
    //-----------------------------------------------------------------------------
    //  function : paralle_introsort
    /// @brief this function implement a non stable parallel sort. The number of
    ///        threads to use is defined by the NThread parameter
    /// @tparam RandomIt : iterators for to access to the elements
    /// @param [in] firts : iterator to the first element of the range to sort
    /// @param [in] last : iterator after the last element to the range to sort
    /// @param [in] NT : This object is a integer from the ranges [1, UINT32_MAX].
    ///                  by default is the number of HW threads of the machine
    /// @exception
    /// @return
    /// @remarks
    //-----------------------------------------------------------------------------
    template    < class RandomIt >
    inline void parallel_introsort ( RandomIt first, RandomIt last )
    {   //---------------------------- begin -------------------------------------
        parallel_sort_comp (first, last);
    };
    //
    //-----------------------------------------------------------------------------
    //  function : paralle_introsort
    /// @brief this function implement a non stable parallel sort. The number of
    ///        threads to use is defined by the NThread parameter
    /// @tparam RandomIt : iterators for to access to the elements
    /// @tparam compare : object for to compare two elements pointed by the RandomIt
    /// @param [in] firts : iterator to the first element of the range to sort
    /// @param [in] last : iterator after the last element to the range to sort
    /// @param [in] comp : object for to compare two elements pointed by RandomIt
    ///                    iterators
    /// @param [in] NT : This object is a integer from the ranges [1, UINT32_MAX].
    ///                  by default is the number of HW threads of the machine
    /// @exception
    /// @return
    /// @remarks
    //-----------------------------------------------------------------------------
/*
    template < class RandomIt,
    typename compare = std::less <typename detail::iter_value<RandomIt>::type> >
    inline void parallel_introsort ( RandomIt first, RandomIt last, compare comp )
    {   //---------------------------- begin -------------------------------------
        parallel_sort_comp ( first, last, comp);
    };
*/
    //
    //-----------------------------------------------------------------------------
    //  function : stable_sort
    /// @brief this function implement a stable sort, based internally in the new
    ///        smart_merge_sort algorithm. Run with 1 thread
    /// @tparam RandomIt : iterators for to access to the elements
    /// @tparam compare : object for to compare two elements pointed by the RandomIt
    /// @param [in] firts : iterator to the first element of the range to sort
    /// @param [in] last : iterator after the last element to the range to sort
    /// @param [in] comp : object for to compare two elements pointed by RandomIt
    ///                    iterators
    /// @exception
    /// @return
    /// @remarks

    //-----------------------------------------------------------------------------
    //  function : paralle_stable_sort
    /// @brief this function implement a stable parallel sort. The number of
    ///        threads to use is defined by the NThread parameter
    /// @tparam RandomIt : iterators for to access to the elements
    /// @param [in] firts : iterator to the first element of the range to sort
    /// @param [in] last : iterator after the last element to the range to sort
    /// @param [in] NT : This object is a integer from the ranges [1, UINT32_MAX].
    ///                  by default is the number of HW threads of the machine
    /// @exception
    /// @return
    /// @remarks
    //-----------------------------------------------------------------------------
    /*
template    < class RandomIt >
inline void parallel_stable_sort ( RandomIt first, RandomIt last )
{   //---------------------------- begin -------------------------------------
    typedef typename iter_value<RandomIt>::type   value_t ;
    if ( sizeof ( value_t) > 64 )
        bs_algo::indirect_sample_sort ( first, last);
    else
       bs_algo::parallel_stable_sort ( first, last);
};
     */
    //
    //-----------------------------------------------------------------------------
    //  function : paralle_stable_sort
    /// @brief this function implement a stable parallel sort. The number of
    ///        threads to use is defined by the NThread parameter
    /// @tparam RandomIt : iterators for to access to the elements
    /// @tparam compare : object for to compare two elements pointed by the RandomIt
    /// @param [in] firts : iterator to the first element of the range to sort
    /// @param [in] last : iterator after the last element to the range to sort
    /// @param [in] comp : object for to compare two elements pointed by RandomIt
    ///                    iterators
    /// @param [in] NT : This object is a integer from the ranges [1, UINT32_MAX].
    ///                  by default is the number of HW threads of the machine
    /// @exception
    /// @return
    /// @remarks
    //-----------------------------------------------------------------------------
    /*
template    < class RandomIt,
              typename compare = std::less < typename iter_value<RandomIt>::type>
            >
inline void parallel_stable_sort ( RandomIt first, RandomIt last, compare comp )
{   //---------------------------- begin -------------------------------------
    typedef typename iter_value<RandomIt>::type   value_t ;
    if ( sizeof ( value_t) > 64)
        bs_algo::indirect_sample_sort ( first, last, comp);
    else
        bs_algo::parallel_stable_sort ( first, last, comp);
};
     */
    //
    //-----------------------------------------------------------------------------
    //  function : sample_sort
    /// @brief this function implement a stable parallel sort with the algorithm of
    ///        sample sort. The number of threads to use is defined by the NThread
    ///        parameter
    /// @tparam RandomIt : iterators for to access to the elements
    /// @param [in] firts : iterator to the first element of the range to sort
    /// @param [in] last : iterator after the last element to the range to sort
    /// @param [in] NT : This object is a integer from the ranges [1, UINT32_MAX].
    ///                  by default is the number of HW threads of the machine
    /// @exception
    /// @return
    /// @remarks
    //-----------------------------------------------------------------------------
    /*
template    < class RandomIt >
inline void sample_sort ( RandomIt first, RandomIt last )
{   //---------------------------- begin -------------------------------------
    typedef typename iter_value<RandomIt>::type   value_t ;
    if ( sizeof ( value_t) > 64 )
        bs_algo::indirect_sample_sort ( first, last);
    else
       bs_algo::sample_sort ( first, last);
};
//
//-----------------------------------------------------------------------------
//  function : sample_sort
/// @brief this function implement a stable parallel sort with the algorithm of
///        sample sort. The number of threads to use is defined by the NThread
///        parameter
/// @tparam RandomIt : iterators for to access to the elements
/// @tparam compare : object for to compare two elements pointed by the RandomIt
/// @param [in] firts : iterator to the first element of the range to sort
/// @param [in] last : iterator after the last element to the range to sort
/// @param [in] comp : object for to compare two elements pointed by RandomIt
///                    iterators
/// @param [in] NT : This object is a integer from the ranges [1, UINT32_MAX].
///                  by default is the number of HW threads of the machine
/// @exception
/// @return
/// @remarks
//-----------------------------------------------------------------------------
template    < class RandomIt,
              typename compare = std::less < typename iter_value<RandomIt>::type>
            >
inline void sample_sort ( RandomIt first, RandomIt last, compare comp )
{   //---------------------------- begin -------------------------------------
    typedef typename iter_value<RandomIt>::type   value_t ;
    if ( sizeof ( value_t) > 64 )
        bs_algo::indirect_sample_sort ( first, last,comp);
    else
        bs_algo::sample_sort ( first, last,comp);
};
     */
    //
    //****************************************************************************
};//    End namespace algorithmssort
};//    End namespace parallel
};//    End namespace hpx
//****************************************************************************
//
#endif
