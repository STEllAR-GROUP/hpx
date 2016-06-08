//----------------------------------------------------------------------------
/// @file atomic.hpp
/// @brief Basic layer for to simplify the use of atomic functions
/// @author Copyright(c) 2010 2015 Francisco José Tapia (fjtapia@gmail.com )\n
///         Distributed under the Boost Software License, Version 1.0.\n
///         ( See accompanyingfile LICENSE_1_0.txt or copy at
///           http://www.boost.org/LICENSE_1_0.txt  )
/// @version 0.1
///
/// @remarks
//-----------------------------------------------------------------------------
#ifndef __TOOLS_ATOMIC_HPP
#define __TOOLS_ATOMIC_HPP

#include <thread>
#include <atomic>
#include <cassert>
#include <utility>
#include <type_traits>
#include <iterator>


namespace hpx
{
namespace parallel
{
HPX_INLINE_NAMESPACE(v2) { namespace boostsort
{
namespace tools
{
//---------------------------------------------------------------------------
//         D E F I N I T I O N S
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
//   A T O M I C  I N C R E M E N T    A N D   R E A D
//---------------------------------------------------------------------------

template <typename T >
inline T atomic_read ( std::atomic<T> & C)
{   return std::atomic_load_explicit<T> (&C, std::memory_order_acquire);
};

template <typename T, typename T2>
inline T atomic_add ( std::atomic<T> &C, T2 N1)
{   static_assert ( std::is_integral <T2>::value , "Bad parameter");
    return
    	std::atomic_fetch_add_explicit<T> (&C,(T)N1,std::memory_order_acq_rel);
};

template <typename T, typename T2>
inline T atomic_sub ( std::atomic<T> &C, T2 N1)
{   static_assert ( std::is_integral <T2>::value , "Bad parameter");
    return
    	std::atomic_fetch_sub_explicit<T> (&C,(T)N1,std::memory_order_acq_rel);
};

template <typename T, typename T2>
inline void atomic_write ( std::atomic<T> &C, T2 N1)
{   static_assert ( std::is_integral <T2>::value , "Bad parameter");
    std::atomic_store_explicit<T> (&C, (T)N1, std::memory_order_release);
};

//****************************************************************************
};//    End namespace tools
};};//    End HPX_INLINE_NAMESPACE(v2) 
};//    End namespace parallel
};//    End namespace hpx
//****************************************************************************
#endif
