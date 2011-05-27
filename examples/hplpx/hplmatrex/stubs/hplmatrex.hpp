////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Dan Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////
#ifndef _HPLMATREX_STUBS_HPP
#define _HPLMATREX_STUBS_HPP

/*This is the HPLMatreX stub file.
*/

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include <examples/hplpx/hplmatrex/server/hplmatrex.hpp>

namespace hpx { namespace components { namespace stubs
{
    struct HPLMatreX : stub_base<server::HPLMatreX>
    {
    //constructor and destructor
    static int construct(naming::id_type gid, unsigned int h,
        unsigned int w, unsigned int ab, unsigned int bs){
        return lcos::eager_future<server::HPLMatreX::construct_action,
            int>(gid,gid,h,w,ab,bs).get();
    }
    static void destruct(naming::id_type gid)
    {
        applier::apply<server::HPLMatreX::destruct_action>(gid);
    }

    //functions for manipulating the matrix
    static double LUsolve(naming::id_type gid){
        return lcos::eager_future<server::HPLMatreX::solve_action,
            double>(gid).get();
    }
    };
}}}

#endif
