//----------------------------------------------------------------------------
/// @file range.hpp
/// @brief
///
/// @author Copyright (c) 2010 2015 Francisco José Tapia (fjtapia@gmail.com )\n
///         Distributed under the Boost Software License, Version 1.0.\n
///         ( See accompanyingfile LICENSE_1_0.txt or copy at
///           http://www.boost.org/LICENSE_1_0.txt  )
/// @version 0.1
///
/// @remarks
//-----------------------------------------------------------------------------
#ifndef __SORT_UTIL_RANGE_HPP
#define __SORT_UTIL_RANGE_HPP

#include <functional>
#include <memory>
#include <type_traits>
#include <iterator>
#include <vector>
#include <cassert>
#include <hpx/parallel/algorithms/sort/util/low_level.hpp>

#ifndef __DEBUG_SORT
#define __DEBUG_SORT 0
#endif

namespace hpx
{
namespace parallel
{
HPX_INLINE_NAMESPACE(v2) { namespace boostsort
{
namespace util
{

using std::iterator_traits;


///---------------------------------------------------------------------------
/// @struct range
/// @brief this represent a range between two iterators
/// @tparam iter_t type of paramenters of the range
/// @remarks
//----------------------------------------------------------------------------
template <class iter_t>
struct range
{   //------------------------ variables -------------------
    iter_t first , last;

    //------------------------- functions ------------------
    range ( void){};
    range ( iter_t frs, iter_t lst):first ( frs),last (lst){};

    bool 	empty 		( void ) const { return (first == last );};
    bool 	not_empty	( void ) const { return (first != last );};
    bool   	valid 		( void ) const { return ((last-first)>= 0 ); };
    size_t 	size		( void ) const { return (last-first);}
};
//
//-----------------------------------------------------------------------------
//  function : concat
/// @brief concatebate two contiguous ranges
/// @tparam value_t : class of the object to create
/// @param [in] it1 : first range
/// @param [in] it2 : second range
/// @return  range resulting of the concatenation
//-----------------------------------------------------------------------------
template <class iter_t>
range<iter_t> concat ( const range<iter_t> &it1 , const range <iter_t> &it2 )
{	//--------------------------- begin -------------------------------------
#if __DEBUG_SORT != 0
    HPX_ASSERT ( it1.last == it2.first );
#endif
    return range<iter_t> ( it1.first , it2.last );
};
//
//-----------------------------------------------------------------------------
//  function : move
/// @brief Move objets from the range src to dest
/// @tparam iter1_t : iterator to the value_t elements
/// @tparam iter2_t : iterator to the value_t elements
/// @param [in] dest : range where move the objects
/// @param [in] src : range from where move the objects
/// @return range with the objects moved and the size adjusted
//-----------------------------------------------------------------------------
template <class iter1_t , class iter2_t >
inline range<iter2_t> init_move ( const range<iter2_t> & dest,
                             const range<iter1_t> & src)
{   //------------- static checking ------------------------------------------
    typedef typename iterator_traits<iter1_t>::value_type type1;
    typedef typename iterator_traits<iter2_t>::value_type type2;
    static_assert ( std::is_same<type1, type2>::value,
                    "Incompatible iterators\n");

    //------------------------------- begin ----------------------------------
    if ( src.size() == 0 ) return range<iter2_t>(dest.first, dest.first);
#if __DEBUG_SORT != 0
    HPX_ASSERT ( dest.size() >= src.size() );
#endif
    lwl::init_move(dest.first ,src.first, src.last  );
    return range<iter2_t>(dest.first, dest.first + src.size());
};
//-----------------------------------------------------------------------------
//  function : uninit_move
/// @brief Move objets from the range src creatinf them in  dest
/// @tparam iter1_t : iterator to the value_t elements
/// @tparam iter2_t : iterator to the value_t elements
/// @param [in] dest : range where move and create the objects
/// @param [in] src : range from where move the objects
/// @return range with the objects moved and the size adjusted
//-----------------------------------------------------------------------------
template <class iter1_t ,class iter2_t >
inline range<iter2_t> uninit_move ( const range<iter2_t> &dest,
                                    const range<iter1_t> &src  )
{   //------------- static checking ------------------------------------------
    typedef typename iterator_traits<iter1_t>::value_type type1;
    typedef typename iterator_traits<iter2_t>::value_type type2;
    static_assert ( std::is_same<type1, type2>::value,
                    "Incompatible iterators\n");

    //------------------------------- begin ----------------------------------
    if ( src.size() == 0 ) return range<iter2_t>(dest.first, dest.first);
#if __DEBUG_SORT != 0
    HPX_ASSERT ( dest.size() >= src.size() );
#endif
    lwl::uninit_move (dest.first,src.first, src.last  );
    return range<iter2_t>(dest.first, dest.first + src.size());
};
//
//-----------------------------------------------------------------------------
//  function : destroy
/// @brief destroy a range of objects
/// @param [in] R : range to destroy
//-----------------------------------------------------------------------------
template <class iter_t >
inline void destroy ( range<iter_t> r)
{   //----------------- begin ---------------------------
    lwl::destroy ( r.first, r.last);
};
//
//-----------------------------------------------------------------------------
//  function : init
/// @brief initialize a range of objects with the object val moving across them

/// @param [in] r : range of elements not initialized
/// @param [in] val : object used for the initialization
/// @return range initialized
//-----------------------------------------------------------------------------
template <class iter_t >
inline range<iter_t> init ( const range<iter_t> & r,
    typename iterator_traits< iter_t>::value_type & val)
{
    //----------------- begin ---------------------------
    lwl::init ( r.first, r.last , val);
    return r;
}
//
//-----------------------------------------------------------------------------
//  function : full_merge
/// @brief Merge two contiguous ranges src1 and src2 , and put the result in
///        the range dest, returning the range merged
/// @param [in] dest : range where locate the lements merged. the size of dest
///                    must be  greater or equal than the sum of the sizes of
///                    src1 and src2
/// @param [in] src1 : first range to merge
/// @param [in] src2 : second range to merge
/// @param [in] comp : comparison object
/// @return range with the elements merged and the size adjusted
//-----------------------------------------------------------------------------
template <class iter1_t, class iter2_t, class iter3_t, class compare >
inline range<iter3_t> full_merge ( const range<iter3_t> &dest,
                                   const range<iter1_t> &src1,
                                   const range<iter2_t> &src2, compare  comp )
{
    //------------------- metaprogramming ------------------------------------
    typedef typename iterator_traits<iter1_t>::value_type type1;
    typedef typename iterator_traits<iter2_t>::value_type type2;
    typedef typename iterator_traits<iter3_t>::value_type type3;
    static_assert ( std::is_same<type1, type2>::value,
                   "Incompatible iterators\n");
    static_assert ( std::is_same<type3, type2>::value,
                   "Incompatible iterators\n");
    //--------------------- code -------------------------------------------
#if __DEBUG_SORT != 0
    HPX_ASSERT ( dest.size() >= ( src1.size() + src2.size() ) );
#endif
    return range<iter3_t> (dest.first,lwl::full_merge ( src1.first, src1.last,
                           src2.first, src2.last , dest.first, comp) );
}

//-----------------------------------------------------------------------------
//  function : uninit_full_merge
/// @brief Merge two contiguous ranges src1 and src2 , and create and move the
///        result in the range dest, returning the range merged
/// @param [in] dest : range where locate the lements merged. the size of dest
///                    must be  greater or equal than the sum of the sizes of
///                    src1 and src2
/// @param [in] src1 : first range to merge
/// @param [in] src2 : second range to merge
/// @param [in] comp : comparison object
/// @return range with the elements merged and the size adjusted
//-----------------------------------------------------------------------------
template <class iter1_t, class iter2_t, class value_t, class compare >
inline range<value_t*> uninit_full_merge ( const range<value_t *> &dest,
                                           const range<iter1_t> &src1,
                                           const range<iter2_t> &src2,
                                           compare  comp                  )
{
    //------------------- metaprogramming ------------------------------------
    typedef typename iterator_traits<iter1_t>::value_type type1;
    typedef typename iterator_traits<iter2_t>::value_type type2;
    static_assert ( std::is_same<type1, type2>::value,
                   "Incompatible iterators\n");
    static_assert ( std::is_same<value_t, type2>::value,
                   "Incompatible iterators\n");
    //--------------------- code -------------------------------------------
#if __DEBUG_SORT != 0
    HPX_ASSERT ( dest.size() >= ( src1.size() + src2.size() ) );
#endif
    return range<value_t *> (dest.first,
                             lwl::uninit_full_merge( src1.first, src1.last,
                                                     src2.first, src2.last ,
                                                     dest.first, comp      ));
}
//
//---------------------------------------------------------------------------
//  function : half_merge
/// @brief : Merge two buffers. The first buffer is in a separate memory
/// @param [in] dest : range where finish the two buffers merged
/// @param [in] src1 : first range to merge in a separate memory
/// @param [in] src2 : second range to merge, in the final part of the
///                    range where deposit the final results
/// @param [in] comp : object for compare two elements of the type pointed
///                    by the iter1_t and iter2_t
/// @return : range with the two buffers merged
//---------------------------------------------------------------------------
template <class iter1_t, class iter2_t, class compare >
inline range<iter2_t> half_merge ( 	const range<iter2_t> &dest,
                                    const range<iter1_t> &src1,
                                    const range<iter2_t> &src2, compare  comp )
{
    //---------------------------- begin ------------------------------------
    typedef typename iterator_traits<iter1_t>::value_type       type1;
    typedef typename iterator_traits<iter2_t>::value_type       type2;
    static_assert ( std::is_same<type1, type2>::value,
                   "Incompatible iterators\n");

    //--------------------- code -------------------------------------------
#if __DEBUG_SORT != 0
    HPX_ASSERT (( src2.first - dest.first) >= 0 and
             size_t ( src2.first - dest.first) == src1.size() );
    HPX_ASSERT ( dest.size() >= (src1.size() + src2.size() ) );
#endif
    return range<iter2_t>( dest.first ,
                           lwl::half_merge ( src1.first , src1.last,
                                             src2.first, src2.last,
                                             dest.first, comp         ) );
};
//
//-----------------------------------------------------------------------------
//  function : in_place_merge_uncontiguous
/// @brief : merge two contiguous buffers
/// @tparam iter_t : iterator to the elements
/// @tparam compare : object for to compare two elements pointed by iter_t
///                   iterators
/// @param [in] first : iterator to the first element
/// @param [in] last : iterator to the element after the last in the range
/// @param [in] comp : object for to compare elements
/// @exception
/// @return true : not changes done
///         false : changes in the buffers
/// @remarks
//-----------------------------------------------------------------------------
template <class iter1_t  , class iter2_t , class iter3_t, class compare >
bool in_place_merge_uncontiguous ( const range<iter1_t> &src1,
                                   const range<iter2_t> &src2,
                                   const range<iter3_t> &aux, compare comp)
{
    //------------------- metaprogramming ------------------------------------
    typedef typename iterator_traits<iter1_t>::value_type type1;
    typedef typename iterator_traits<iter2_t>::value_type type2;
    typedef typename iterator_traits<iter3_t>::value_type type3;

    static_assert ( std::is_same<type1, type2>::value,
                   "Incompatible iterators\n");
    static_assert ( std::is_same<type3, type2>::value,
                   "Incompatible iterators\n");
    //--------------------- code -------------------------------------------
#if __DEBUG_SORT != 0
    HPX_ASSERT ( aux.size() >= src1.size() );
#endif
    return lwl::in_place_merge_uncontiguous ( src1.first, src1.last,
                                              src2.first, src2.last,
                                              aux.first, comp );
}

//
//-----------------------------------------------------------------------------
//  function : in_place_merge
/// @brief : merge two contiguous buffers
/// @tparam iter_t : iterator to the elements
/// @tparam compare : object for to compare two elements pointed by iter_t
///                   iterators
/// @param [in] first : iterator to the first element
/// @param [in] last : iterator to the element after the last in the range
/// @param [in] comp : object for to compare elements
/// @exception
/// @return true : not changes done
///         false : changes in the buffers
/// @remarks
//-----------------------------------------------------------------------------
template <class iter1_t  , class iter2_t , class compare >
inline range<iter1_t> in_place_merge (const range<iter1_t> &src1,
                                      const range<iter1_t> &src2,
                                      const range<iter2_t> &buf, compare  comp )
{   //---------------------------- begin ------------------------------------
    typedef typename iterator_traits<iter1_t>::value_type type1;
    typedef typename iterator_traits<iter2_t>::value_type type2;

    static_assert ( std::is_same<type1, type2>::value,
                   "Incompatible iterators\n");
    //---------------------------- begin --------------------------------------
#if __DEBUG_SORT != 0
    HPX_ASSERT ( src1.last == src2.first);
    HPX_ASSERT ( buf.size() >= src1.size() );
#endif
    lwl::in_place_merge ( src1.first , src1.last,
                          src2.last, buf.first, comp );
    return concat ( src1, src2);
}
//
//****************************************************************************
}//    End namespace util
}}//    End HPX_INLINE_NAMESPACE(v2) 
}//    End namespace parallel
}//    End namespace hpx
//****************************************************************************
//
#endif
