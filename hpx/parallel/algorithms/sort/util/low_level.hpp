//----------------------------------------------------------------------------
/// @file low_level.hpp
/// @brief
///
/// @author Copyright (c) 2010 2015 Francisco José Tapia (fjtapia@gmail.com )\n
///         Distributed under the Boost Software License, Version 1.0.\n
///         (See accompanyingfile LICENSE_1_0.txt or copy at
///           http://www.boost.org/LICENSE_1_0.txt  )
/// @version 0.1
///
/// @remarks
//-----------------------------------------------------------------------------
#ifndef __BOOST_SORT_PARALLEL_UTIL_LOW_LEVEL_HPP
#define __BOOST_SORT_PARALLEL_UTIL_LOW_LEVEL_HPP

#include <functional>
#include <memory>
#include <utility>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <iterator>


namespace hpx {
namespace parallel {
HPX_INLINE_NAMESPACE(v2) { namespace boostsort {
namespace util {
namespace lwl {

using std::iterator_traits;

//
//-----------------------------------------------------------------------------
//  function : construct_object
/// @brief create an object in the memory specified by Ptr
/// @tparam value_t : class of the object to create
/// @tparam Args : parameters for the constructor
/// @param [in] Ptr : pointer to the memory where to create the object
/// @param [in] args : arguments to the constructor
//-----------------------------------------------------------------------------
template <class value_t ,class ... Args>
inline void construct_object (value_t *Ptr, Args && ... args)
{
    (::new (static_cast<void*> (Ptr)) value_t (std::forward<Args> (args)...));
}
//
//-----------------------------------------------------------------------------
//  function : destroy_object
/// @brief destroy an object in the memory specified by Ptr
/// @tparam value_t : class of the object to create
/// @param [in] Ptr : pointer to the object to destroy
//-----------------------------------------------------------------------------
template <class value_t >
inline void destroy_object (value_t *Ptr){   Ptr->~value_t (); }

//
//-----------------------------------------------------------------------------
//  function : init
/// @brief initialize a range of objects with the object val moving across them

/// @param [in] r : range of elements not initialized
/// @param [in] val : object used for the initialization
/// @return range initialized
//-----------------------------------------------------------------------------
template <class iter_t >
inline void init (iter_t  first , iter_t  last ,
                   typename iterator_traits< iter_t>::value_type & val)
{
    //----------------- begin ---------------------------
    if (first == last  ) return ;
    construct_object (&(*first), std::move (val) );
    iter_t it1 = first, it2 = first+1;
    while (it2 != last)
        construct_object (&(*(it2++)), std::move (* (it1++)));
    val = std::move (* (last -1));
}
//
//-----------------------------------------------------------------------------
//  function : construct
/// @brief create an object in the memory specified by Ptr
/// @tparam value_t : class of the object to create
/// @tparam Args : parameters for the constructor
/// @param [in] Ptr : pointer to the memory where to create the object
/// @param [in] args : arguments to the constructor
//-----------------------------------------------------------------------------
template <class value_t ,class ... Args>
inline void construct (value_t *Ptr, Args && ... args)
{
    (::new (static_cast<void*> (Ptr)) value_t (std::forward<Args> (args)...));
}

//
//-----------------------------------------------------------------------------
//  function : init_move
/// @brief Move objets
/// @tparam iter_t : iterator to the elements
/// @tparam value_t : class of the object to create
/// @param [in] itdest : iterator to the final place of the objects
/// @param [in] R : range to move
//-----------------------------------------------------------------------------
template <class iter1_t , class iter2_t >
inline iter2_t init_move (iter2_t  it_dest,
                           iter1_t  first,
                           const iter1_t   last)
{
    //----------------- begin ---------------------------
    while (first != last )   *(it_dest++) = std::move (*(first++));
    return it_dest;
}

//
//-----------------------------------------------------------------------------
//  function : uninit_move
/// @brief Move objets to uninitialized memory
/// @tparam iter_t : iterator to the elements
/// @tparam value_t : class of the object to construct
/// @param [in] Ptr : pointer to the memory where to create the object
/// @param [in] R : range to move
//-----------------------------------------------------------------------------
template < class iter_t ,
           class value_t = typename iterator_traits<iter_t>::value_type >
inline value_t * uninit_move (value_t *  Ptr,
                               iter_t  first,
                               iter_t  last      )
{
    //----------------- begin ---------------------------
    typedef typename iterator_traits<iter_t>::value_type value2_t;
    static_assert (std::is_same<value_t, value2_t>::value,
                   "Incompatible iterators\n");
    //--------------------- code -------------------------------------------
    while (first != last )
        ::new (static_cast<void*> (Ptr++)) value_t (std::move (*(first++)) );
    return Ptr;
}
//
//-----------------------------------------------------------------------------
//  function : destroy
/// @brief Move objets to uninitialized memory
/// @tparam iter_t : iterator to the elements
/// @tparam value_t : class of the object to construct
/// @param [in] Ptr : pointer to the memory where to construct the object
/// @param [in] R : range to move
//-----------------------------------------------------------------------------
template <class iter_t >
inline void destroy (iter_t   first, const iter_t  last )
{
    //----------------- begin ---------------------------
    typedef typename iterator_traits<iter_t>::value_type value_t;

    while (first != last) (&(*(first++)))->~value_t ();
}
//
//-----------------------------------------------------------------------------
//  function : full_merge
/// @brief Merge two contiguous buffers pointed by first1 and first2 , and put
///        in the buffer pointed by P
/// @tparam iter1_t : iterator to the input buffers
/// @tparam iter2_t : iterator to the output buffers
/// @tparam compare : object to compate the elements pointed by iter1_t
/// @param [in] buf1 : iterator to the first element in the first buffer
/// @param [in] buf2 : iterator to the first iterator to the second buffer
/// @param [in] end_buf2 : final iterator of the second buffer
/// @param [in] buf_out : buffer where move the elements merged
/// @param [in] comp : comparison object
//-----------------------------------------------------------------------------
template <class iter1_t, class iter2_t, class compare >
inline iter2_t full_merge (iter1_t  buf1,
                            const iter1_t  end_buf1,
                            iter1_t  buf2,
                            const iter1_t  end_buf2,
                            iter2_t  buf_out, compare  comp )
{
    //------------------- metaprogramming ------------------------------------
    typedef typename iterator_traits<iter1_t>::value_type value1_t;
    typedef typename iterator_traits<iter2_t>::value_type value2_t;
    static_assert (std::is_same<value1_t, value2_t>::value,
                   "Incompatible iterators\n");
    //--------------------- code -------------------------------------------
    while ((buf1 != end_buf1) && (buf2 != end_buf2) )
    {
        *(buf_out++) = (!comp(*buf2,*buf1)) ? std::move(*(buf1++))
                                            : std::move(*(buf2++));
    }
    return  (buf1 == end_buf1) ? init_move (buf_out, buf2, end_buf2 )
                                : init_move (buf_out, buf1, end_buf1 );
}
//
//-----------------------------------------------------------------------------
//  function : uninit_full_merge
/// @brief Merge two contiguous buffers pointed by first1 and first2 , and put
///        in the buffer pointed by P
/// @tparam iter1_t : iterator to the input buffers
/// @tparam iter2_t : iterator to the output buffers
/// @tparam compare : object to compate the elements pointed by iter1_t
/// @param [in] buf1 : iterator to the first element in the first buffer
/// @param [in] buf2 : iterator to the first iterator to the second buffer
/// @param [in] end_buf2 : final iterator of the second buffer
/// @param [in] buf_out : buffer where move the elements merged
/// @param [in] comp : comparison object
//-----------------------------------------------------------------------------
template <class iter_t, class value_t, class compare >
inline value_t* uninit_full_merge (iter_t  first1,
                                    const iter_t   last1 ,
                                    iter_t  first2 ,
                                    const iter_t  last2 ,
                                    value_t*   it_out, compare comp )
{
    //------------------------- metaprogramming -------------------------------
    typedef typename iterator_traits<iter_t>::value_type type1;
    static_assert (std::is_same<value_t, type1>::value,
                   "Incompatible iterators\n");
    //--------------------- code -------------------------------------------
    while (first1 != last1 and first2 != last2 )
    {
        construct ((it_out++),(not comp(*first2,*first1))?
                             std::move(*(first1++)):std::move(*(first2++)));
    }
    return  (first1 == last1 ) ? uninit_move (it_out, first2, last2)
                                : uninit_move (it_out, first1, last1);
}
//
//---------------------------------------------------------------------------
//  function : half_merge
/// @brief : Merge two buffers. The first buffer is in a separate memory
/// @tparam iter1_t : iterator to the first buffer
/// @tparam iter2_t : iterator to the second buffer and the output buffer
/// @tparam compare : object to compate the elements pointed by the iterators
/// @param [in] buf1 : iterator to the first element of the first buffer
/// @param [in] end_buf1 : iterator to the last element of the first buffer
/// @param [in] buf2 : iterator to the first element of the second buffer
/// @param [in] end_buf2 : iterator to the last element of the second buffer
/// @param [in] buf_out : iterator to the first element to the buffer where put
///                       the result
/// @param [in] comp : object for compare two elements of the type pointed
///                    by the iter1_t and iter2_t
/// @exception
/// @return
/// @remarks
//---------------------------------------------------------------------------
template <class iter1_t, class iter2_t, class compare >
inline iter2_t half_merge (iter1_t  buf1   ,
                            const iter1_t  end_buf1 ,
                            iter2_t  buf2   ,
                            const iter2_t  end_buf2 ,
                            iter2_t  buf_out, compare  comp     )
{
    //---------------------------- begin ------------------------------------
    typedef typename iterator_traits<iter1_t>::value_type value1_t;
    typedef typename iterator_traits<iter2_t>::value_type value2_t;
    static_assert (std::is_same<value1_t, value2_t>::value,
                   "Incompatible iterators\n");
    //--------------------- code -------------------------------------------
    while ((buf1 != end_buf1) && (buf2 != end_buf2)  )
    {
        *(buf_out++) = (!comp(*buf2,*buf1)) ? std::move (*(buf1++))
                                            : std::move (*(buf2++));
    }
    return (buf2 == end_buf2) ? init_move (buf_out , buf1, end_buf1 ) : end_buf2;
}
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
bool in_place_merge_uncontiguous (iter1_t  src1,
                                   const iter1_t  end_src1 ,
                                   iter2_t  src2,
                                   const iter2_t  end_src2 ,
                                   iter3_t  aux , compare comp )
{
    //------------------- metaprogramming ------------------------------------
    typedef typename iterator_traits<iter1_t>::value_type type1;
    typedef typename iterator_traits<iter2_t>::value_type type2;
    typedef typename iterator_traits<iter3_t>::value_type type3;

    static_assert (std::is_same<type1, type2>::value,
                   "Incompatible iterators\n");
    static_assert (std::is_same<type3, type2>::value,
                   "Incompatible iterators\n");
    //--------------------- code -------------------------------------------
    if (src1 == end_src1 || src2 == end_src2 || !comp (*src2, *(end_src1 -1)) ) 
        return true;

    while (src1 != end_src1 && !comp (*src2, *src1)) 
        ++src1;

    iter3_t const  end_aux = aux + (end_src1 - src1);
    iter2_t  src2_first = src2;
    init_move (aux , src1 , end_src1);

    while ((src1 != end_src1) && (src2 != end_src2))
    {
        if (!comp(*src2, *aux))
            *(src1++) = std::move(*aux++);
        else
            *(src1++) = std::move(*src2++);
    }

    if (src2 == end_src2)
    {
        while (src1 != end_src1) 
            *(src1 ++) = std::move (*(aux ++));
        init_move (src2_first, aux , end_aux);
    }
    else
    {
        half_merge (aux, end_aux, src2, end_src2, src2_first ,comp);
    }
    return false;
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
inline bool in_place_merge (iter1_t   src1,
                             iter1_t   src2 ,
                             iter1_t   end_src2,
                             iter2_t  buf,
                             compare  comp )
{
    //---------------------------- begin ------------------------------------
    typedef typename iterator_traits<iter1_t>::value_type type1;
    typedef typename iterator_traits<iter2_t>::value_type type2;

    static_assert (std::is_same<type1, type2>::value,
                   "Incompatible iterators\n");
    //---------------------------- begin --------------------------------------
    if (src1 == src2 || src2 == end_src2 || !comp (*src2 , * (src2 -1)))
        return true;

    iter1_t end_src1 = src2;
    while (src1 != end_src1 && !comp (*src2, *src1)) ++src1;

    if (src1 == end_src1 ) return false;

    size_t Nx  = end_src1 - src1;
    init_move (buf ,src1, end_src1 );
    half_merge (buf , buf + Nx , src2, end_src2 ,src1,comp);
    return false;
}
//****************************************************************************
}//    End namespace lwl
}//    End namespace util
}//    End namespace parallel
}}//    End HPX_INLINE_NAMESPACE(v2) 
}//    End namespace boost
//****************************************************************************
//
#endif
