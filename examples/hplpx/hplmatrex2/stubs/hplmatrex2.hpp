////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Dan Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////
#ifndef _HPLMATREX2_STUBS_HPP
#define _HPLMATREX2_STUBS_HPP

/*This is the HPLMatreX22 stub file.
*/

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include <examples/hplpx/hplmatrex2/server/hplmatrex2.hpp>

namespace hpx { namespace components { namespace stubs
{
    struct HPLMatreX2 : stub_base<server::HPLMatreX2>
    {
    //constructor and destructor
    static int construct(naming::id_type gid, unsigned int h,
        unsigned int w, unsigned int ab, unsigned int bs){
        return lcos::eager_future<server::HPLMatreX2::construct_action,
            int>(gid,gid,h,w,ab,bs).get();
    }
    static void destruct(naming::id_type gid)
    {
        applier::apply<server::HPLMatreX2::destruct_action>(gid);
    }

    //operators for assignment and data access
    static void set(naming::id_type gid,
        unsigned int row, unsigned int col, double val){
        applier::apply<server::HPLMatreX2::set_action>(gid,row,col,val);
    }
    static double get(naming::id_type gid,
        unsigned int row, unsigned int col){
        return lcos::eager_future<server::HPLMatreX2::get_action,
            double>(gid,row,col).get();
    }

    //functions for manipulating the matrix
    static double LUsolve(naming::id_type gid){
        return lcos::eager_future<server::HPLMatreX2::solve_action,
            double>(gid).get();
    }
    };
}}}

#endif
