////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////
#ifndef _BHCONTROLLER_STUBS_HPP
#define _BHCONTROLLER_STUBS_HPP

/*This is the bhcontroller stub file.
*/

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include "../server/bhcontroller.hpp"

namespace hpx { namespace components { namespace stubs
{
    struct bhcontroller : stub_base<server::bhcontroller>
    {
        //constructor
        static int construct(naming::id_type gid, std::string inputFile){
            return lcos::eager_future<server::bhcontroller::constructAction,
                int>(gid,gid,inputFile).get();
        }

        static int run_simulation(naming::id_type gid){
            return lcos::eager_future<server::bhcontroller::runAction,
                int>(gid).get();
        }
    };
}}}

#endif
