//----------------------------------------------------------------------------
/// @file spinlock.hpp
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
#ifndef __TOOLS_SPINLOCK_HPP
#define __TOOLS_SPINLOCK_HPP

#include <hpx/hpx.hpp>
#include <atomic>
#include <memory>
#include <functional>
#include <mutex>
//#include <thread>
//#include <ctime>

namespace hpx
{
namespace parallel
{
HPX_INLINE_NAMESPACE(v2) { namespace boostsort
{
namespace tools
{

//##########################################################################
//                                                                        ##
//                 C L A S S     S P I N L O C K                          ##
//                                                                        ##
//##########################################################################
//---------------------------------------------------------------------------
/// @class spinlock
///
/// @remarks This class meet the BasicLockable requirements ( lock, unlock )
//---------------------------------------------------------------------------
class spinlock
{
private:
//***************************************************************************
//             P R I V A T E      V A R I A B L E S
//***************************************************************************
std::atomic_flag af ;

public :
//
//---------------------------------------------------------------------------
//  function : spinlock
/// @brief  class constructor
/// @param [in]
//---------------------------------------------------------------------------
explicit inline spinlock() noexcept{ af.clear(); };
//
//---------------------------------------------------------------------------
//  function : lock
/// @brief  Lock the spinlock
//---------------------------------------------------------------------------
inline void lock() noexcept
{   if ( af.test_and_set(std::memory_order_acquire))
    {   while ( wait() and  af.test_and_set(std::memory_order_relaxed) );
    };
};
//
//---------------------------------------------------------------------------
//  function : wait
/// @brief
/// @param [in]
/// @return true :locked false: not previous locked
//---------------------------------------------------------------------------
inline bool wait () const noexcept
{   hpx::this_thread::yield();
    return true ;
} ;
//
//---------------------------------------------------------------------------
//  function : try_lock
/// @brief Try to lock the spinlock, if not, return false
/// @param [in]
/// @return true :locked false: not previous locked
//---------------------------------------------------------------------------
inline bool try_lock() noexcept
{   return not af.test_and_set(std::memory_order_acquire);
};
//
//---------------------------------------------------------------------------
//  function : unlock
/// @brief  unlock the spinlock
//---------------------------------------------------------------------------
inline void unlock() noexcept
{   //----------------------- begin -----------------
    af.clear(std::memory_order_release );
};
//
//***************************************************************************
};// E N D    C L A S S     S P I N L O C K
//***************************************************************************
//
//****************************************************************************
};//    End namespace tools
};};//    End HPX_INLINE_NAMESPACE(v2) 
};//    End namespace parallel
};//    End namespace hpx
//****************************************************************************
#endif
