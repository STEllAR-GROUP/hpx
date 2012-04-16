////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Dan Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////
#ifndef _SMPHPLMATREX_STUBS_HPP
#define _SMPHPLMATREX_STUBS_HPP

/*This is the smphplmatrex stub file.
*/

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/include/async.hpp>

#include <examples/hplpx/smphplmatrex/server/smphplmatrex.hpp>

namespace hpx { namespace components { namespace stubs
{
    struct smphplmatrex : stub_base<server::smphplmatrex>
    {
    //constructor and destructor
    static int construct(naming::id_type gid, unsigned int h,
        unsigned int ab, unsigned int bs){
        return hpx::async<server::smphplmatrex::construct_action>(
            gid,gid,h,ab,bs).get();
    }
    static void destruct(naming::id_type gid)
    {
        hpx::apply<server::smphplmatrex::destruct_action>(gid);
    }

    //functions for manipulating the matrix
    static double LUsolve(naming::id_type gid){
        return hpx::async<server::smphplmatrex::solve_action>(gid).get();
    }
    };
}}}

#endif
