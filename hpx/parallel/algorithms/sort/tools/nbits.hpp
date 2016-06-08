//----------------------------------------------------------------------------
/// @file algorithm.hpp
/// @brief This file contains the description of several low level algorithms
///
/// @author Copyright (c) 2010 2015 Francisco José Tapia (fjtapia@gmail.com )\n
///         Distributed under the Boost Software License, Version 1.0.\n
///         ( See accompanying file LICENSE_1_0.txt or copy at
///           http://www.boost.org/LICENSE_1_0.txt  )
/// @version 0.1
///
/// @remarks
//-----------------------------------------------------------------------------
#ifndef __BOOST_SORT_PARALLEL_UTIL_NBITS_HPP
#define __BOOST_SORT_PARALLEL_UTIL_NBITS_HPP

#include <cstdint>

namespace hpx		{
namespace parallel	{
HPX_INLINE_NAMESPACE(v2) { namespace boostsort		{
namespace tools		{
//
//##########################################################################
//                                                                        ##
//                    G L O B A L     V A R I B L E S                     ##
//                                                                        ##
//##########################################################################
//
static constexpr const uint32_t TMSB [256] =
{   0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8
};
//
//##########################################################################
//                                                                        ##
//                I N L I N E      F U N C T I O N S                      ##
//                                                                        ##
//##########################################################################
//
//---------------------------------------------------------------------------
//  function : NBits
/// @brief Obtain the number of bits equal or greater than N
/// @param [in] N : Number to examine
/// @exception none
/// @return Number of bits
//---------------------------------------------------------------------------
static inline uint32_t NBits32 ( uint32_t N) noexcept
{   //----------------------- begin -------------------------------------
    int Pos = (N & 0xffff0000U)?16:0 ;
    if ((N>>Pos) & 0xff00U) Pos += 8 ;
    return (  TMSB [ N >> Pos ] + Pos );
}
//
//---------------------------------------------------------------------------
//  function : NBits
/// @brief Obtain the number of bits equal or greater than N
/// @param [in] N : Number to examine
/// @exception none
/// @return Number of bits
//---------------------------------------------------------------------------
static inline uint32_t NBits64 ( uint64_t N)
{   //----------------------- begin -------------------------------------
    uint32_t  Pos = ( N & 0xffffffff00000000ULL)?32:0 ;
    if ( (N>>Pos) & 0xffff0000ULL ) Pos +=16  ;
    if ( (N>>Pos) & 0xff00ULL     ) Pos += 8 ;
    return ( TMSB [ N >> Pos ] + Pos );
}

//****************************************************************************
};//    End namespace tools
};//    End namespace parallel
};};//    End HPX_INLINE_NAMESPACE(v2) 
};//    End namespace boost
//****************************************************************************
#endif
