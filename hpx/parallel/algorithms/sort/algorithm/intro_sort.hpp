//----------------------------------------------------------------------------
/// @file intro_sort.hpp
/// @brief Intro Sort algorithm
///
/// @author Copyright (c) 2010 Francisco José Tapia (fjtapia@gmail.com )\n
///         Distributed under the Boost Software License, Version 1.0.\n
///         ( See accompanyingfile LICENSE_1_0.txt or copy at
///           http://www.boost.org/LICENSE_1_0.txt  )
/// @version 0.1
///
/// @remarks
//-----------------------------------------------------------------------------
#ifndef __HPX_PARALLEL_SORT_ALGORITHM_INTRO_SORT_HPP
#define __HPX_PARALLEL_SORT_ALGORITHM_INTRO_SORT_HPP

#include <algorithm>
#include <vector>
#include <type_traits>
#include <iterator>
#include <hpx/parallel/algorithms/sort/tools/nbits.hpp>
#include <hpx/parallel/algorithms/sort/algorithm/insertion_sort.hpp>
#include <hpx/parallel/algorithms/sort/algorithm/heap_sort.hpp>
#include <hpx/parallel/algorithms/sort/algorithm/indirect.hpp>


namespace hpx		{
namespace parallel 	{
HPX_INLINE_NAMESPACE(v2) { namespace boostsort		{
namespace algorithm	{
namespace deep   	{

using std::iterator_traits ;
using hpx::parallel::v2::boostsort::tools::NBits64;

template< typename iter_t, typename compare>
inline iter_t mid3(iter_t it_l, iter_t it_m, iter_t it_r,compare comp)
{	return comp(* it_l, *it_m)
		?( comp(*it_m, *it_r) ? it_m : ( comp(*it_l, *it_r) ? it_r:it_l))
        :( comp(*it_r, *it_m) ? it_m : ( comp(*it_r, *it_l) ? it_r:it_l));
};

//----------------------------------------------------------------------------
// calculate the pivoting using a mid of 3 and move to the firat position
//----------------------------------------------------------------------------
template <class iter_t , class compare >
inline void pivot3 ( iter_t first, iter_t last, compare comp)
{   auto N2 = ( last - first ) >>1 ;
	iter_t it_val = mid3 ( first +1, first + N2, last-1,comp);
    std::swap ( *first , * it_val);
};

template <class iter_t , class compare >
inline iter_t mid9 ( iter_t it1, iter_t it2 , iter_t it3 ,
		             iter_t it4 , iter_t it5, iter_t it6,
					 iter_t it7, iter_t it8, iter_t it9, compare comp)
{	//-------------------------- begin ---------------------------------------
	return mid3 (mid3(it1, it2, it3, comp ),
		         mid3(it4, it5, it6,comp  ),
				 mid3(it7, it8, it9,comp), comp);
};

template <class iter_t , class compare >
inline void pivot9 ( iter_t first, iter_t last, compare comp)
{	//----------------------------- begin ------------------------------------
	size_t cupo = (last - first) >>3 ;
	iter_t itaux = mid9 (first+1, first+cupo, first+2*cupo,
  	                	first+3*cupo, first + 4*cupo, first + 5*cupo,
						first + 6*cupo, first + 7*cupo,last-1,comp);
	std::swap ( *first , * itaux);
};
//
//-----------------------------------------------------------------------------
//  function : intro_sort_internal
/// @brief : internal function for to divide and sort the ranges
/// @param [in] first : iterator to the first element
/// @param [in] last : iterator to the element after the last in the range
/// @param [in] Level : Level of depth from the initial range
/// @param [in] comp : object for to compare elements
//-----------------------------------------------------------------------------
template< class iter_t,
          typename compare
          = std::less <typename iterator_traits<iter_t>::value_type >  >
void intro_sort_internal( iter_t  first , iter_t  last,
		                         uint32_t Level ,compare comp = compare())
{   //------------------------------ begin -----------------------------------
    typedef typename iterator_traits<iter_t>::value_type       value_t ;

    const uint32_t NMin = 32 ;
    auto N = last - first;
    if ( N  < NMin )   return insertion_sort( first , last,comp);
    if ( Level == 0)   return heap_sort     ( first , last,comp);

    //--------------------- division ----------------------------------
    pivot3 ( first, last, comp);

    const value_t & val = const_cast < value_t &>(* first);
    iter_t c_first = first+1 , c_last  = last-1;

    while ( comp(*c_first, val) ) ++c_first ;
    while ( comp ( val,*c_last ) ) --c_last ;
    while (not( c_first > c_last ))
    {   std::swap ( *(c_first++), *(c_last--));
        while ( comp (*c_first, val) ) ++c_first;
        while ( comp ( val, *c_last) ) --c_last ;
    }; // End while
    std::swap ( *first , * c_last);
    intro_sort_internal (first , c_last, Level -1, comp);
    intro_sort_internal (c_first, last, Level -1 , comp);
};
//
//****************************************************************************
};//    End namespace deep
//****************************************************************************
//
//
//-----------------------------------------------------------------------------
//  function : intro_sort
/// @brief : function for to sort range [first, last )
/// @param [in] first : iterator to the first element
/// @param [in] last : iterator to the element after the last in the range
/// @param [in] comp : object for to compare elements
//-----------------------------------------------------------------------------
template < class iter_t,
           typename compare
           = std::less <typename iterator_traits<iter_t>::value_type>
         >
void intro_sort ( iter_t first, iter_t last,compare comp = compare())
{   //------------------------- begin ----------------------
    auto N = last - first;
    assert ( N > 0);
    //------------------- check if sort --------------------------------------
    //if (std::is_sorted ( first, last ,comp)) return ;
    //------------------- check if sort --------------------------------------
    bool SW = true ;
    for ( iter_t it1 = first, it2 = first+1 ;
        it2 != last and (SW = not comp(*it2,*it1));it1 = it2++);
    if (SW) return ;

    uint32_t Level = ((NBits64(N)-4) *3)/2;
    deep::intro_sort_internal ( first , last, Level,comp);
};


//############################################################################
//                                                                          ##
//                I N D I R E C T     F U N C T I O N S                     ##
//                                                                          ##
//############################################################################
//
//-----------------------------------------------------------------------------
//  function : indirect_intro_sort
/// @brief : function for to implement an indirect sort range [first, last )
/// @param [in] first : iterator to the first element
/// @param [in] last : iterator to the element after the last in the range
/// @param [in] comp : object for to compare elements

//-----------------------------------------------------------------------------
template < class iter_t,
           typename compare
           = std::less<typename deep::iterator_traits<iter_t>::value_type>   >
void indirect_intro_sort ( iter_t first, iter_t last ,
                                    compare comp = compare() )
{   //------------------------------- begin--------------------------
    typedef less_ptr_no_null <iter_t, compare>      compare_ptr ;

    //------------------- check if sort --------------------------------------
    bool SW = true ;
    for ( iter_t it1 = first, it2 = first+1 ;
        it2 != last and (SW = not comp(*it2,*it1));it1 = it2++);
    if (SW) return ;

    std::vector<iter_t> VP ;
    create_index ( first , last , VP);
    intro_sort  ( VP.begin() , VP.end(), compare_ptr(comp) );
    sort_index ( first , VP) ;
};

//
//****************************************************************************
};//    End namespace algorithm
};//    End namespace parallel
};};//    End HPX_INLINE_NAMESPACE(v2) 
};//    End namespace boost
//****************************************************************************
//
#endif
