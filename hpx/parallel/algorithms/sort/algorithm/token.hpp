//----------------------------------------------------------------------------
/// @file token.hpp
/// @brief Insertion Sort algorithm
///
/// @author Copyright (c) 2010 2015 Francisco José Tapia (fjtapia@gmail.com )\n
///         Distributed under the Boost Software License, Version 1.0.\n
///         ( See accompanyingfile LICENSE_1_0.txt or copy at
///           http://www.boost.org/LICENSE_1_0.txt  )
/// @version 0.1
///
/// @remarks
//-----------------------------------------------------------------------------
#ifndef __SORT_ALGORITHM_TOKEN_HPP
#define __SORT_ALGORITHM_TOKEN_HPP


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
/// @struct token
/// @brief pair of iterators. This struct is used by several sort algorithms
/// @tparam iter_t : iterator to the elements to sort
/// @remarks
//----------------------------------------------------------------------------
template <class iter_t>
struct token
{   //----------- variables -------------
    iter_t first ;
    iter_t last ;
    //------------------- Functions -------------------
    token( void){} ;
    token ( iter_t it_f , iter_t it_l):first ( it_f), last ( it_l){};
};
//
///---------------------------------------------------------------------------
/// @struct token
/// @brief pair of iterators. This struct is used by several sort algorithms
/// @tparam iter_t : iterator to the elements to sort
/// @remarks
//----------------------------------------------------------------------------
template <class iter_t>
struct token_level
{   //----------- variables -------------
    iter_t first ;
    iter_t last ;
    int32_t level ;
    //------------------- Functions -------------------
    token_level( void){} ;
    token_level ( iter_t it_f, iter_t it_l, int32_t lv)
                 :first (it_f),last(it_l),level(lv) {  };
};
//
//****************************************************************************
};//    End namespace algorithm
};};//    End HPX_INLINE_NAMESPACE(v2) 
};//    End namespace parallel
};//    End namespace hpx
//****************************************************************************
//
#endif
