////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Dan Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////
#ifndef _HPLMATREX_HPP
#define _HPLMATREX_HPP

/*This is the HPLMatreX interface header file.
In order to keep things simple, only operations necessary
to to perform LUP decomposition are declared, which is
basically just constructors, assignment operators, 
a destructor, and access operators.
*/

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/hplmatrex.hpp"

namespace hpx { namespace components
{
    class HPLMatreX : public client_base<HPLMatreX, stubs::HPLMatreX>
    {
    typedef
        client_base<HPLMatreX, stubs::HPLMatreX> base_type;

    public:
    //constructors and destructor
    HPLMatreX(){}
    HPLMatreX(naming::id_type gid) : base_type(gid){}
    void destruct(){
        BOOST_ASSERT(gid_);
        return this->base_type::destruct(gid_);
    }

    //initialization function
    int construct(unsigned int h, unsigned int w, unsigned int ab, unsigned int bs){
        BOOST_ASSERT(gid_);
        return this->base_type::construct(gid_,h,w,ab,bs);
    }

    //functions for solving the matrix
    double LUsolve(){
        BOOST_ASSERT(gid_);
        return this->base_type::LUsolve(gid_);
    }
    };
}}

#endif
