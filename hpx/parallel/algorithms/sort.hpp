//----------------------------------------------------------------------------
/// @file sort.hpp
/// @brief This file contains the sort functions of the sort library
///
/// @author Copyright (c) 2015 Francisco José Tapia (fjtapia@gmail.com )\n
///         Distributed under the Boost Software License, Version 1.0.\n
///         ( See accompanyingfile LICENSE_1_0.txt or copy at
///           http://www.boost.org/LICENSE_1_0.txt  )
/// @version 0.1
///
/// @remarks
//-----------------------------------------------------------------------------
#ifndef __HPX_PARALLEL_ALGORITHMS_GENERAL_SORT_HPP
#define __HPX_PARALLEL_ALGORITHMS_GENERAL_SORT_HPP

#include <hpx/parallel/algorithms/parallel_sort.hpp>
//#include <hpx/parallel/algorithms/algorithm/parallel_stable_sort.hpp>

namespace hpx
{
namespace parallel
{
namespace algorithms
{

//****************************************************************************
//             USING AND DEFINITIONS
//****************************************************************************
//namespace bs_algo = hpx::parallel::algorithms::algorithm;
namespace hpa_algo = hpx::parallel::algorithms;
//namespace bspu = hpx::parallel::algorithms::util;
//using bspu::iter_value ;
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

//
//-----------------------------------------------------------------------------
//  function : introsort
/// @brief this function implement a non stable sort, based internally in the
///        intro_sort algorithm. Run with 1 thread
/// @tparam iter_t : iterators for to access to the elements
/// @tparam compare : object for to compare two elements pointed by the iter_t
/// @param [in] firts : iterator to the first element of the range to sort
/// @param [in] last : iterator after the last element to the range to sort
/// @param [in] comp : object for to compare two elements pointed by iter_t
///                    iterators
/// @exception
/// @return
/// @remarks
//-----------------------------------------------------------------------------
/*
template < class iter_t,
           typename compare = std::less <typename iter_value<iter_t>::type> >
inline void introsort ( iter_t first, iter_t last,compare comp = compare())
{   //---------------------------- begin -------------------------------------
    bs_algo::intro_sort(first, last, comp);
};
*/
//
//-----------------------------------------------------------------------------
//  function : paralle_introsort
/// @brief this function implement a non stable parallel sort. The number of
///        threads to use is defined by the NThread parameter
/// @tparam iter_t : iterators for to access to the elements
/// @param [in] firts : iterator to the first element of the range to sort
/// @param [in] last : iterator after the last element to the range to sort
/// @param [in] NT : This object is a integer from the ranges [1, UINT32_MAX].
///                  by default is the number of HW threads of the machine
/// @exception
/// @return
/// @remarks
//-----------------------------------------------------------------------------
template    < class iter_t >
inline void parallel_introsort ( iter_t first, iter_t last )
{   //---------------------------- begin -------------------------------------
    hpa_algo::parallel_sort ( first, last);
};
//
//-----------------------------------------------------------------------------
//  function : paralle_introsort
/// @brief this function implement a non stable parallel sort. The number of
///        threads to use is defined by the NThread parameter
/// @tparam iter_t : iterators for to access to the elements
/// @tparam compare : object for to compare two elements pointed by the iter_t
/// @param [in] firts : iterator to the first element of the range to sort
/// @param [in] last : iterator after the last element to the range to sort
/// @param [in] comp : object for to compare two elements pointed by iter_t
///                    iterators
/// @param [in] NT : This object is a integer from the ranges [1, UINT32_MAX].
///                  by default is the number of HW threads of the machine
/// @exception
/// @return
/// @remarks
//-----------------------------------------------------------------------------
template < class iter_t,
           typename compare = std::less <typename iter_value<iter_t>::type> >
inline void parallel_introsort ( iter_t first, iter_t last, compare comp )
{   //---------------------------- begin -------------------------------------
    hpa_algo::parallel_sort ( first, last, comp);
};
//
//-----------------------------------------------------------------------------
//  function : stable_sort
/// @brief this function implement a stable sort, based internally in the new
///        smart_merge_sort algorithm. Run with 1 thread
/// @tparam iter_t : iterators for to access to the elements
/// @tparam compare : object for to compare two elements pointed by the iter_t
/// @param [in] firts : iterator to the first element of the range to sort
/// @param [in] last : iterator after the last element to the range to sort
/// @param [in] comp : object for to compare two elements pointed by iter_t
///                    iterators
/// @exception
/// @return
/// @remarks
/*
//-----------------------------------------------------------------------------
template < class iter_t,
           typename compare = std::less<typename iter_value<iter_t>::type>
         >
inline void smart_merge_sort(iter_t first, iter_t last, compare comp = compare() )
{   //--------------------------------- begin --------------------------------
    typedef typename iter_value<iter_t>::type   value_t ;
    if ( sizeof ( value_t) > 128)
        bs_algo::indirect_smart_merge_sort(first, last, comp);
    else
        bs_algo::smart_merge_sort(first, last,comp);
};
*/
//
//-----------------------------------------------------------------------------
//  function : paralle_stable_sort
/// @brief this function implement a stable parallel sort. The number of
///        threads to use is defined by the NThread parameter
/// @tparam iter_t : iterators for to access to the elements
/// @param [in] firts : iterator to the first element of the range to sort
/// @param [in] last : iterator after the last element to the range to sort
/// @param [in] NT : This object is a integer from the ranges [1, UINT32_MAX].
///                  by default is the number of HW threads of the machine
/// @exception
/// @return
/// @remarks
//-----------------------------------------------------------------------------
/*
template    < class iter_t >
inline void parallel_stable_sort ( iter_t first, iter_t last )
{   //---------------------------- begin -------------------------------------
    typedef typename iter_value<iter_t>::type   value_t ;
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
/// @tparam iter_t : iterators for to access to the elements
/// @tparam compare : object for to compare two elements pointed by the iter_t
/// @param [in] firts : iterator to the first element of the range to sort
/// @param [in] last : iterator after the last element to the range to sort
/// @param [in] comp : object for to compare two elements pointed by iter_t
///                    iterators
/// @param [in] NT : This object is a integer from the ranges [1, UINT32_MAX].
///                  by default is the number of HW threads of the machine
/// @exception
/// @return
/// @remarks
//-----------------------------------------------------------------------------
/*
template    < class iter_t,
              typename compare = std::less < typename iter_value<iter_t>::type>
            >
inline void parallel_stable_sort ( iter_t first, iter_t last, compare comp )
{   //---------------------------- begin -------------------------------------
    typedef typename iter_value<iter_t>::type   value_t ;
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
/// @tparam iter_t : iterators for to access to the elements
/// @param [in] firts : iterator to the first element of the range to sort
/// @param [in] last : iterator after the last element to the range to sort
/// @param [in] NT : This object is a integer from the ranges [1, UINT32_MAX].
///                  by default is the number of HW threads of the machine
/// @exception
/// @return
/// @remarks
//-----------------------------------------------------------------------------
/*
template    < class iter_t >
inline void sample_sort ( iter_t first, iter_t last )
{   //---------------------------- begin -------------------------------------
    typedef typename iter_value<iter_t>::type   value_t ;
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
/// @tparam iter_t : iterators for to access to the elements
/// @tparam compare : object for to compare two elements pointed by the iter_t
/// @param [in] firts : iterator to the first element of the range to sort
/// @param [in] last : iterator after the last element to the range to sort
/// @param [in] comp : object for to compare two elements pointed by iter_t
///                    iterators
/// @param [in] NT : This object is a integer from the ranges [1, UINT32_MAX].
///                  by default is the number of HW threads of the machine
/// @exception
/// @return
/// @remarks
//-----------------------------------------------------------------------------
template    < class iter_t,
              typename compare = std::less < typename iter_value<iter_t>::type>
            >
inline void sample_sort ( iter_t first, iter_t last, compare comp )
{   //---------------------------- begin -------------------------------------
    typedef typename iter_value<iter_t>::type   value_t ;
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
