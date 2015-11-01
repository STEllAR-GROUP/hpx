//----------------------------------------------------------------------------
/// @file parallel_sort.hpp
/// @brief Parallel Sort algorithm
///
/// @author Copyright (c) 2015 Francisco José Tapia (fjtapia@gmail.com )\n
///         Distributed under the Boost Software License, Version 1.0.\n
///         ( See accompanyingfile LICENSE_1_0.txt or copy at
///           http://www.boost.org/LICENSE_1_0.txt  )
/// @version 0.1
///
/// @remarks
//-----------------------------------------------------------------------------
#ifndef __HPX_PARALLEL_ALGORITHMS_PARALLEL_SORT_HPP
#define __HPX_PARALLEL_ALGORITHMS_PARALLEL_SORT_HPP

#include <hpx/runtime/threads/thread.hpp>
//
//#include <hpx/parallel/algorithms/util/atomic.hpp>
//#include <hpx/parallel/algorithms/util/util_iterator.hpp>
//
//#include <hpx/parallel/algorithms/algorithm/intro_sort.hpp>
//#include <hpx/parallel/algorithms/algorithm/indirect.hpp>
//
#include <hpx/include/util.hpp>
#include <hpx/include/lcos.hpp>

#include <vector>
#include <functional>

//namespace bspu = hpx::parallel::algorithms::util;
//using namespace hpx::parallel::algorithms::util;

namespace hpx
{
namespace parallel
{
namespace algorithms
{

///---------------------------------------------------------------------------
/// @struct iter_value
/// @brief This is for to obtain the type of data pointed by an iterator
/// @tparam iter_t type of iterator
/// @remarks
//----------------------------------------------------------------------------
template <typename iter_t>
struct iter_value
{   typedef typename
    std::remove_reference<decltype(*(std::declval<iter_t>()))>::type type ;
};

//---------------------------------------------------------------------------
/// @class less_ptr_no_null
///
/// @remarks this is the comparison object for pointers. Receive a object
///          for to compare the objects pointed. The pointers can't be nullptr
//---------------------------------------------------------------------------
template    <   class iter_t ,
                class comp_t =std::less<typename iter_value<iter_t>::type>
            >
struct less_ptr_no_null
{   //----------------------------- Variables -----------------------
    comp_t comp ;
    //----------------------------- Functions ----------------------
    inline less_ptr_no_null ( comp_t C1 = comp_t()):comp(C1){};
    inline bool operator ()( iter_t  T1,  iter_t  T2 ) const
    {   //-------------------- begin ------------------------------
        return  comp(*T1 ,*T2);
    };
};

//
///---------------------------------------------------------------------------
/// @struct parallel_sort_comp
/// @brief implement the parallel sort using the intro_sort algorithm
/// @tparam iter_t : iterators pointing to the elements
/// @tparam compare : objects for to compare two elements pointed by iter_t
///                   iterators
/// @remarks
//----------------------------------------------------------------------------
template < class iter_t,
           typename compare = std::less <typename iter_value<iter_t>::type>  >
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
    /// @param [in] NT : variable indicating the number of threads used in the
    ///                  sorting
    /// @exception
    /// @return
    /// @remarks
    //------------------------------------------------------------------------
    parallel_sort_comp (iter_t first, iter_t last )
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
    parallel_sort_comp ( iter_t first, iter_t last, compare comp1)
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
        for ( iter_t it1 = first, it2 = first+1 ;
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
    hpx::future<void> sort_thread( iter_t first, iter_t last)
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
        typedef typename iter_value<iter_t>::type  value_t ;

        //------------------- check if sort ------------------------------
        bool SW = true ;
        for ( iter_t it1 = first, it2 = first+1 ;
                it2 != last and (SW = not comp(*it2,*it1));it1 = it2++);
        if (SW)
        {
            return hpx::make_ready_future();
        };
        //---------------------- pivot select ----------------------------
        size_t Nx = ( size_t (N ) >>1 ) ;

        iter_t itA = first +1 ;
        iter_t itB = first + Nx ;
        iter_t itC = last -1 ;

        if ( comp( *itB , *itA )) std::swap ( *itA, *itB);
        if ( comp (*itC , *itB))
        {
            std::swap (*itC , *itB );
            if ( comp( *itB , *itA )) std::swap ( *itA, *itB);
        };
        std::swap ( *first , *itB);
        value_t &  val = const_cast < value_t &>(* first);
        iter_t c_first = first+2 , c_last  = last-2;

        while ( c_first != last and comp (*c_first, val)) ++c_first ;
        while ( comp(val ,*c_last ) ) --c_last ;
        while (not( c_first > c_last ))
        {
            std::swap ( *(c_first++), *(c_last--));
            while ( comp(*c_first,val) ) ++c_first;
            while ( comp(val, *c_last) ) --c_last ;
        }; // End while
        std::swap ( *first , *c_last);

        // spawn threads for each sub section
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
//
//-----------------------------------------------------------------------------
//  function : parallel_sort
/// @brief function envelope with the comparison object defined by defect
/// @tparam iter_t : iterator used for to access to the elements
/// @param [in] first : iterator to the first element
/// @param [in] last : iterator to the next element to the last valid iterator
/// @param [in] NT : NThread object for to define the number of threads used
///                  in the process. By default is the number of HW threads
/// @exception
/// @return
/// @remarks
//-----------------------------------------------------------------------------
template    < class iter_t >
void parallel_sort ( iter_t first, iter_t last )
{   parallel_sort_comp <iter_t> ( first, last);
};
//
//-----------------------------------------------------------------------------
//  function : parallel_sort
/// @brief function envelope with the comparison object defined by defect
/// @tparam iter_t : iterator used for to access to the elements
/// @tparam compare : object for to compare the lements pointed by iter_t
/// @param [in] first : iterator to the first element
/// @param [in] last : iterator to the next element to the last valid iterator
/// @param [in] comp : object for to compare
/// @param [in] NT : NThread object for to define the number of threads used
///                  in the process. By default is the number of HW threads
/// @exception
/// @return
/// @remarks
//-----------------------------------------------------------------------------
template    < class iter_t,
              typename compare = std::less < typename iter_value<iter_t>::type>
            >
void parallel_sort ( iter_t first, iter_t last, compare comp1)
{   parallel_sort_comp<iter_t,compare> ( first, last,comp1);
};


//############################################################################
//                                                                          ##
//                I N D I R E C T     F U N C T I O N S                     ##
//                                                                          ##
//############################################################################
//
//-----------------------------------------------------------------------------
//  function : indirect_parallel_sort
/// @brief indirect parallel sort
/// @tparam iter_t : iterator used for to access to the elements
/// @param [in] first : iterator to the first element
/// @param [in] last : iterator to the next element to the last valid iterator
/// @param [in] NT : NThread object for to define the number of threads used
///                  in the process. By default is the number of HW threads
/// @exception
/// @return
/// @remarks
//-----------------------------------------------------------------------------
template < class iter_t >
void indirect_parallel_sort ( iter_t first, iter_t last )
{   //------------------------------- begin--------------------------
    typedef std::less <typename iter_value<iter_t>::type> compare ;
    typedef less_ptr_no_null <iter_t, compare>      compare_ptr ;

    std::vector<iter_t> VP ;
    create_index ( first , last , VP);
    parallel_sort  ( VP.begin() , VP.end(), compare_ptr());
    sort_index ( first , VP) ;
};
//
//-----------------------------------------------------------------------------
//  function : indirect_parallel_sort
/// @brief indirect parallel sort
/// @tparam iter_t : iterator used for to access to the elements
/// @tparam compare : object for to compare the lements pointed by iter_t
/// @param [in] first : iterator to the first element
/// @param [in] last : iterator to the next element to the last valid iterator
/// @param [in] comp : object for to compare
/// @param [in] NT : NThread object for to define the number of threads used
///                  in the process. By default is the number of HW threads
/// @exception
/// @return
/// @remarks
//-----------------------------------------------------------------------------
template < class iter_t,
          typename compare = std::less < typename iter_value<iter_t>::type >
        >
void indirect_parallel_sort ( iter_t first, iter_t last,
                              compare comp1)
{   //----------------------------- begin ----------------------------------
    typedef less_ptr_no_null <iter_t, compare>      compare_ptr ;

    std::vector<iter_t> VP ;
    create_index ( first , last , VP);
    parallel_sort  ( VP.begin() , VP.end(), compare_ptr(comp1) );
    sort_index ( first , VP) ;
};
//
//****************************************************************************
};//    End namespace algorithms
};//    End namespace parallel
};//    End namespace hpx
//****************************************************************************
//
#endif
