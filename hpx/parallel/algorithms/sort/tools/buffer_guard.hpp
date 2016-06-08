//----------------------------------------------------------------------------
/// @file   buffer_guard.hpp
/// @brief
///
/// @author Copyright (c) 2016 Francisco José Tapia (fjtapia@gmail.com )\n
///         Distributed under the Boost Software License, Version 1.0.\n
///         ( See accompanyingfile LICENSE_1_0.txt or copy at
///           http://www.boost.org/LICENSE_1_0.txt  )
/// @version 0.1
///
/// @remarks
//-----------------------------------------------------------------------------
#ifndef __HPX_PARALLEL_SORT_TOOLS_BUFFER_GUARD_HPP
#define __HPX_PARALLEL_SORT_TOOLS_BUFFER_GUARD_HPP

#include <hpx/parallel/algorithms/sort/tools/stack_cnc.hpp>
#include <vector>
#include <cassert>

namespace hpx
{
namespace parallel
{
HPX_INLINE_NAMESPACE(v2) { namespace boostsort
{
namespace tools
{

//###########################################################################
//                                                                         ##
//    ################################################################     ##
//    #                                                              #     ##
//    #                      C L A S S                               #     ##
//    #               B U F F E R _ G U A R D                        #     ##
//    #                                                              #     ##
//    ################################################################     ##
//                                                                         ##
//###########################################################################
//
//---------------------------------------------------------------------------
/// @class  buffer_guard
/// @brief This is a stack of bufferes initialized for to be used by the
///        threads
///        The idea is like the lock_guard, you create the object and obtain
///        an iterator to a buffer, and when the object is destroyed, the
///        buffer is automatically returned to the stack
/// @remarks
//---------------------------------------------------------------------------
template <class iter_t >
struct buffer_guard
{
    //---------------------- varibales ------------------------------------
    stack_cnc<iter_t> & stk;
    bool OK = false;
    iter_t itx ;

    buffer_guard (stack_cnc<iter_t> & STI, iter_t &it_buf ) :stk(STI)
    {   //---------------------- begin----------------------------
    	while (not (OK = stk.pop_copy_back(itx))  )hpx::this_thread::yield();
    	it_buf= itx ;
    };

    void  close ( void)
    { 	stk.push_back( itx);
    	OK = false;
    };
    ~buffer_guard ()
    { 	if ( OK) stk.push_back( itx);
    };
};



//***************************************************************************
};// end namespace tools
};};// end HPX_INLINE_NAMESPACE(v2)
};// end namespace parallel
};// end namespace hpx
//***************************************************************************
#endif
