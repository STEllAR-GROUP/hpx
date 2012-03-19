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
#include <hpx/runtime/components/client_base.hpp>

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
        BOOST_ASSERT(gid_);
        return this->base_type::construct_block(
            gid_,h,w,px,py,size,gidList,theData);
    }

    //Gaussian functions below
    server::lublock::gcFuture gauss_corner(const int iter){
        BOOST_ASSERT(gid_);
        return this->base_type::gauss_corner(gid_,iter);
    }
    server::lublock::gtoFuture gauss_top(const int iter){
        BOOST_ASSERT(gid_);
        return this->base_type::gauss_top(gid_,iter);
    }
    server::lublock::glFuture gauss_left(const int iter){
        BOOST_ASSERT(gid_);
        return this->base_type::gauss_left(gid_,iter);
    }
    server::lublock::gtrFuture gauss_trail(const int iter){
        BOOST_ASSERT(gid_);
        return this->base_type::gauss_trail(gid_,iter);
    }

    //get functions
    int get_rows(){
        BOOST_ASSERT(gid_);
        return this->base_type::get_rows(gid_);
    }
    int get_columns(){
        BOOST_ASSERT(gid_);
        return this->base_type::get_columns(gid_);
    }
    vector<vector<double> > get_data(){
        BOOST_ASSERT(gid_);
        return this->base_type::get_data(gid_);
    }
    };
}}

#endif
