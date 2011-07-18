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
#include <hpx/lcos/eager_future.hpp>

#include "../server/lublock.hpp"

namespace hpx { namespace components { namespace stubs
{
    struct lublock : stub_base<server::lublock>
    {
    //constructor
    static int construct_block(const naming::id_type gid, const int h,
        const int w, const naming::id_type _gid,
        const std::vector<std::vector<double> > theData){
        return lcos::eager_future<server::lublock::constructBlock_action,
            int>(gid,h,w,_gid,theData).get();
    }

    //Gaussian functions
    static server::lublock::gcFuture gauss_corner(const naming::id_type gid){
        return server::lublock::gcFuture(gid);
    }
    static server::lublock::gtopFuture gauss_top(const naming::id_type gid,
        const naming::id_type corner){
        return server::lublock::gtopFuture(gid,corner);
    }
    static server::lublock::glFuture gauss_left(const naming::id_type gid,
        const naming::id_type corner){
        return server::lublock::glFuture(gid,corner);
    }
    static server::lublock::gtrFuture gauss_trail(const naming::id_type gid,
        const int size, const naming::id_type corner,const naming::id_type left,
        const naming::id_type top){
        return server::lublock::gtrFuture(gid,size,corner,left,top);
    }

    //other functions
    static int get_rows(const naming::id_type gid){
        return server::lublock::rowFuture(gid).get();
    }
    static int get_columns(const naming::id_type gid){
        return server::lublock::columnFuture(gid).get();
    }
    static std::vector<std::vector<double> > get_data(const naming::id_type gid){
        return server::lublock::dataFuture(gid).get();
    }
    };
}}}

#endif
