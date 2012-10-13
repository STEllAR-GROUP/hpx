////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Dan Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////
#ifndef _LUBLOCK_HPP
#define _LUBLOCK_HPP

/*This is the lublock interface header file.*/

#include <hpx/runtime.hpp>
#include <hpx/include/client.hpp>

#include "stubs/lublock.hpp"

namespace hpx { namespace components
{
    class lublock : public client_base<lublock, stubs::lublock>
    {
    typedef
        client_base<lublock, stubs::lublock> base_type;

    public:
    //constructors
    lublock(){}
    lublock(id_type gid) : base_type(gid){}

    //initialization function
    int construct_block(const int h, const int w, const int px, const int py,
        const int size, const vector<vector<id_type> > gidList,
        const vector<vector<double> > theData){
        BOOST_ASSERT(get_gid());
        return this->base_type::construct_block(
            get_gid(),h,w,px,py,size,gidList,theData);
    }

    //Gaussian functions below
    server::lublock::gcFuture gauss_corner(const int iter){
        return this->base_type::gauss_corner(get_gid(),iter);
    }
    server::lublock::gtoFuture gauss_top(const int iter){
        return this->base_type::gauss_top(get_gid(),iter);
    }
    server::lublock::glFuture gauss_left(const int iter){
        return this->base_type::gauss_left(get_gid(),iter);
    }
    server::lublock::gtrFuture gauss_trail(const int iter){
        return this->base_type::gauss_trail(get_gid(),iter);
    }

    //get functions
    int get_rows(){
        return this->base_type::get_rows(get_gid());
    }
    int get_columns(){
        return this->base_type::get_columns(get_gid());
    }
    vector<vector<double> > get_data(){
        return this->base_type::get_data(get_gid());
    }
    };
}}

#endif
