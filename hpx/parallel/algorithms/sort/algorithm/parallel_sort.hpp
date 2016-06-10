//----------------------------------------------------------------------------
/// @file parallel_sort.hpp
/// @brief Parallel Sort algorithm
///
/// @author Copyright (c) 2015 Francisco José Tapia (fjtapia@gmail.com )\n
///                            John A. Biddiscombe  (biddisco@cscs.ch) \n
///         Distributed under the Boost Software License, Version 1.0.\n
///         ( See accompanyingfile LICENSE_1_0.txt or copy at
///           http://www.boost.org/LICENSE_1_0.txt  )
/// @version 0.1
///
/// @remarks
//-----------------------------------------------------------------------------
#ifndef __HPX_PARALLEL_SORT_ALGORITHM_PARALLEL_SORT_HPP
#define __HPX_PARALLEL_SORT_ALGORITHM_PARALLEL_SORT_HPP

#include <hpx/runtime/threads/thread.hpp>
#include <hpx/parallel/algorithms/sort/algorithm/intro_sort.hpp>
#include <hpx/parallel/algorithms/sort/algorithm/indirect.hpp>

#include <hpx/include/util.hpp>
#include <hpx/include/lcos.hpp>
#include <vector>
#include <iterator>

namespace hpx
{
namespace parallel
{
HPX_INLINE_NAMESPACE(v2) { namespace boostsort
{
namespace algorithm
{
//
///---------------------------------------------------------------------------
/// @struct parallel_sort_comp
/// @brief implement the parallel sort using the intro_sort algorithm
/// @remarks
//----------------------------------------------------------------------------
template < class iter_t,
           typename compare = std::less <typename std::iterator_traits<iter_t>::value_type>  >
struct parallel_sort_comp
{   //------------------------- begin ----------------------
    static const size_t Min_Parallel = 65536 ;
    compare comp;

    //------------------------------------------------------------------------
    //  function : parallel_sort_comp
    /// @brief constructor of the struct
    /// @param [in] first : iterator to the first element to sort
    /// @param [in] last : iterator to the next element after the last
    /// @param [in] comp : object for to compare
    /// @param [in] NT : variable indicating the number of threads used in the
    ///                  sorting
    //------------------------------------------------------------------------
    parallel_sort_comp ( iter_t first, iter_t last, compare comp1= compare ())
                        :comp(comp1)
    {   //------------------------- begin ----------------------
        auto N = last - first;
        assert ( N >=0);

        if ( (size_t)N < Min_Parallel  )
        {   intro_sort ( first, last,comp) ;
            return ;
        } ;
        //------------------- check if sort ----------------------------------
        bool SW = true ;
        for ( iter_t it1 = first, it2 = first+1 ;
            it2 != last and (SW = not comp(*it2,*it1));it1 = it2++);
        if (SW) return ;

        //-------------------------------------------------------------------
        hpx::future<void> Rt = hpx::async( &parallel_sort_comp::sort_thread,
                              this ,first , last );
        Rt.get() ;

    };
    //
    //------------------------------------------------------------------------
    //  function : sort_thread
    /// @brief this function is the work asigned to each thread in the parallel
    ///        process
    /// @param [in] first : iterator to the first element to sort
    /// @param [in] last : iterator to the next element after the last
    /// @param [in] level : level of depth
    //------------------------------------------------------------------------
    hpx::future <void> sort_thread(iter_t first, iter_t last  )
    {   //------------------------- begin ----------------------
        size_t N  = last - first ;
        if ( N <= Min_Parallel)
        {   return
            hpx::async([this,first,last]() { intro_sort(first, last, comp);});
        };
        //------------------- check if sort ------------------------------
        bool SW = true ;
        for ( iter_t it1 = first, it2 = first+1 ;
              it2 != last and (SW = not comp(*it2,*it1));it1 = it2++);
        if (SW)  return hpx::make_ready_future();

        //----------------------------------------------------------------
        //                     split
        //----------------------------------------------------------------
        typedef typename std::iterator_traits<iter_t>::value_type  value_t ;


        //---------------------- pivot select ----------------------------
        iter_t itA = first +1 ;
        iter_t itB = first + ( size_t (N ) >>1 ) ;
        iter_t itC = last -1 ;

        if ( comp( *itB , *itA )) std::iter_swap(itA, itB);
        if ( comp (*itC , *itB))
        {
            std::iter_swap (itC , itB);
            if ( comp( *itB , *itA )) std::iter_swap(itA, itB);
        }
        std::iter_swap(first, itB);
        auto&  val = *first;
        iter_t c_first = first+2 , c_last  = last-2;

        while ( c_first != last && comp(*c_first, val)) ++c_first ;
        while ( comp(val ,*c_last ) ) --c_last ;
        while (not( c_first > c_last ))
        {
            std::iter_swap (c_first++, c_last--);
            while ( comp(*c_first,val) ) ++c_first;
            while ( comp(val, *c_last) ) --c_last ;
        }; // End while
        std::iter_swap(first, c_last);

        //----------------------------------------------------------------
        hpx::future <void> Rt1 = hpx::async ( &parallel_sort_comp::sort_thread,
                                this, first, c_last);

        hpx::future<void> Rt2 = hpx::async ( &parallel_sort_comp::sort_thread,
                                this,c_first, last);
        return hpx::dataflow
               ( [] (hpx::future<void> f1, hpx::future<void> f2) -> void
                 {	f1.get();
                    f2.get();
                    return;
                }, std::move(Rt1), std::move(Rt2)      );
    }
};

//
//-----------------------------------------------------------------------------
//  function : parallel_sort
/// @brief function envelope with the comparison object defined by defect
///
/// @param [in] first : iterator to the first element
/// @param [in] last : iterator to the next element to the last valid iterator
/// @param [in] NT : NThread object for to define the number of threads used
///                  in the process. By default is the number of HW threads

//-----------------------------------------------------------------------------
template    < class iter_t >
void parallel_sort ( iter_t first, iter_t last  )
{
    parallel_sort_comp <iter_t> ( first, last);
}
//
//-----------------------------------------------------------------------------
//  function : parallel_sort
/// @brief function envelope with the comparison object defined by defect
///
/// @param [in] first : iterator to the first element
/// @param [in] last : iterator to the next element to the last valid iterator
/// @param [in] comp : object for to compare
/// @param [in] NT : NThread object for to define the number of threads used
///                  in the process. By default is the number of HW threads
//-----------------------------------------------------------------------------
template    < class iter_t,
              typename compare = std::less < typename std::iterator_traits<iter_t>::value_type>
            >
void parallel_sort ( iter_t first, iter_t last, compare comp1  )
{
    parallel_sort_comp<iter_t,compare> ( first, last,comp1);
}


//############################################################################
//                                                                          ##
//                I N D I R E C T     F U N C T I O N S                     ##
//                                                                          ##
//############################################################################
//
//-----------------------------------------------------------------------------
//  function : indirect_parallel_sort
/// @brief indirect parallel sort
///
/// @param [in] first : iterator to the first element
/// @param [in] last : iterator to the next element to the last valid iterator
/// @param [in] NT : NThread object for to define the number of threads used
///                  in the process. By default is the number of HW threads
//-----------------------------------------------------------------------------
template < class iter_t >
void indirect_parallel_sort ( iter_t first, iter_t last )
{
    //------------------------------- begin--------------------------
    typedef std::less <typename std::iterator_traits<iter_t>::value_type> compare ;
    typedef hpx::parallel::v2::boostsort::algorithm::less_ptr_no_null <iter_t, compare>      compare_ptr ;

    std::vector<iter_t> VP ;
    create_index ( first , last , VP);
    parallel_sort  ( VP.begin() , VP.end(), compare_ptr() );
    sort_index ( first , VP) ;
}
//
//-----------------------------------------------------------------------------
//  function : indirect_parallel_sort
/// @brief indirect parallel sort
///
/// @param [in] first : iterator to the first element
/// @param [in] last : iterator to the next element to the last valid iterator
/// @param [in] comp : object for to compare
/// @param [in] NT : NThread object for to define the number of threads used
///                  in the process. By default is the number of HW threads
//-----------------------------------------------------------------------------
template < class iter_t,
          typename compare = std::less < typename std::iterator_traits<iter_t>::value_type >
        >
void indirect_parallel_sort ( iter_t first, iter_t last,
                              compare comp1)
{
    //----------------------------- begin ----------------------------------
    typedef less_ptr_no_null <iter_t, compare>      compare_ptr ;

    std::vector<iter_t> VP ;
    create_index ( first , last , VP);
    parallel_sort  ( VP.begin() , VP.end(), compare_ptr(comp1) );
    sort_index ( first , VP) ;
}
//
//****************************************************************************
}//    End namespace algorithm
}}//    End HPX_INLINE_NAMESPACE(v2) 
}//    End namespace parallel
}//    End namespace hpx
//****************************************************************************
//
#endif
