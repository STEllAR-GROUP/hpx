////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Dan Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////
#ifndef _HPLMATREX_STUBS_HPP
#define _HPLMATREX_STUBS_HPP

/*This is the hplmatrex stub file.
*/

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/include/async.hpp>

#include "../server/hplmatrex.hpp"

namespace hpx { namespace components { namespace stubs
{
    struct hplmatrex : stub_base<server::hplmatrex>
    {
    //constructor
    static int construct(naming::id_type gid, unsigned int h,
        unsigned int ab, unsigned int bs){
        return hpx::async<server::hplmatrex::construct_action>(
            gid,gid,h,ab,bs).get();
    }

    //functions for manipulating the matrix
    static double LUsolve(naming::id_type gid){
        return hpx::async<server::hplmatrex::solve_action>(gid).get();
    }
    };
}}}

#endif
