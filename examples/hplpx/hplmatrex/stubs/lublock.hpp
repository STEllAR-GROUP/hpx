////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Dan Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////
#ifndef _LUBLOCK_STUBS_HPP
#define _LUBLOCK_STUBS_HPP

/*This is the lublock stub file.
*/

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/include/async.hpp>

#include "../server/lublock.hpp"

namespace hpx { namespace components { namespace stubs
{
    struct lublock : stub_base<server::lublock>
    {
    //constructor
    static int construct_block(const id_type gid, const int h, const int w,
        const int posx, const int posy, const int size,
        const vector<vector<id_type> > gidList,
        const vector<vector<double> > theData){
        return hpx::async<server::lublock::constructBlock_action>(
            gid,h,w,posx,posy,size,gidList,theData).get();
    }

    //Gaussian functions
    static server::lublock::gcFuture gauss_corner(const id_type gid,
        const int iter){
        return server::lublock::gcFuture(gid,iter);
    }
    static server::lublock::gtoFuture gauss_top(const id_type gid,
        const int iter){
        return server::lublock::gtoFuture(gid,iter);
    }
    static server::lublock::glFuture gauss_left(const id_type gid,
        const int iter){
        return server::lublock::glFuture(gid,iter);
    }
    static server::lublock::gtrFuture gauss_trail(const id_type gid,
        const int iter){
        return server::lublock::gtrFuture(gid,iter);
    }

    //get functions
    static int get_rows(const id_type gid){
        return server::lublock::rowFuture(gid).get_future().get();
    }
    static int get_columns(const id_type gid){
        return server::lublock::columnFuture(gid).get_future().get();
    }
    static vector<vector<double> > get_data(const id_type gid){
        return server::lublock::dataFuture(gid).get_future().get();
    }
    };
}}}

#endif
