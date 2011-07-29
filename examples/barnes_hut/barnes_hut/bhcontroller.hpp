////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////
#ifndef _BHCONTROLLER_HPP
#define _BHCONTROLLER_HPP

/*This is the bhcontroller interface header file.
*/

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/bhcontroller.hpp"

namespace hpx { namespace components
{
    class bhcontroller : public client_base<bhcontroller, stubs::bhcontroller>
    {
        typedef
            client_base<bhcontroller, stubs::bhcontroller> base_type;

        public:
        //constructors and destructor
        bhcontroller(){}
        bhcontroller(naming::id_type gid) : base_type(gid){}

        //initialization function
        int construct(std::string inputFile){
            BOOST_ASSERT(gid_);
            return this->base_type::construct(gid_,inputFile);
        }

        int run_simulation(){
            BOOST_ASSERT(gid_);
            return this->base_type::run_simulation(gid_);
        }
    };
}}

#endif
