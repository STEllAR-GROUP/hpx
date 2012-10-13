////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Dan Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////
#ifndef _SMPHPLMATREX_HPP
#define _SMPHPLMATREX_HPP

/*This is the smphplmatrex interface header file.
In order to keep things simple, only operations necessary
to to perform LUP decomposition are declared, which is
basically just constructors, assignment operators,
a destructor, and access operators.
*/

#include <hpx/runtime.hpp>
#include <hpx/include/client.hpp>

#include "stubs/smphplmatrex.hpp"

namespace hpx { namespace components
{
    class smphplmatrex : public client_base<smphplmatrex, stubs::smphplmatrex>
    {
    typedef
        client_base<smphplmatrex, stubs::smphplmatrex> base_type;

    public:
    //constructors and destructor
    smphplmatrex(){}
    smphplmatrex(naming::id_type gid) : base_type(gid){}
    void destruct(){
        return this->base_type::destruct(get_gid());
    }

    //initialization function
    int construct(unsigned int h, unsigned int ab, unsigned int bs){
        return this->base_type::construct(get_gid(),h,ab,bs);
    }

    //functions for solving the matrix
    double LUsolve(){
        return this->base_type::LUsolve(get_gid());
    }
    };
}}

#endif
