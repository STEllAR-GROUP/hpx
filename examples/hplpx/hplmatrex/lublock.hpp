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
    lublock(naming::id_type gid) : base_type(gid){}

    //initialization function
    int construct_block(const int h, const int w, const naming::id_type _gid,
        const std::vector<std::vector<double> > theData){
        BOOST_ASSERT(gid_);
        return this->base_type::construct_block(gid_,h,w,_gid,theData);
    }

    //Gaussian functions below
    server::lublock::gcFuture gauss_corner(){
        BOOST_ASSERT(gid_);
        return this->base_type::gauss_corner(gid_);
    }
    server::lublock::gtopFuture gauss_top(const naming::id_type corner){
        BOOST_ASSERT(gid_);
        return this->base_type::gauss_top(gid_,corner);
    }
    server::lublock::glFuture gauss_left(const naming::id_type corner){
        BOOST_ASSERT(gid_);
        return this->base_type::gauss_left(gid_,corner);
    }
    server::lublock::gtrFuture gauss_trail(const int size, const naming::id_type corner, 
        const naming::id_type left, const naming::id_type top){
        BOOST_ASSERT(gid_);
        return this->base_type::gauss_trail(gid_, size, corner, left, top);
    }

    //other functions
    int get_rows(){
        BOOST_ASSERT(gid_);
        return this->base_type::get_rows(gid_);
    }
    int get_columns(){
        BOOST_ASSERT(gid_);
        return this->base_type::get_columns(gid_);
    }
    std::vector<std::vector<double> > get_data(){
        BOOST_ASSERT(gid_);
        return this->base_type::get_data(gid_);
    }
    };
}}

#endif
