//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/lcos/async_future_wait.hpp>

#include <boost/bind.hpp>
#include <boost/ref.hpp>

#include "../stubs/point.hpp"
#include "./point.hpp"

#include <iostream>
#include <fstream>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace geometry { namespace server
{
        std::vector<std::size_t> point::traverse(std::size_t level,std::size_t parent)
        {
          if ( visited_ == false ) {
            visited_ = true;
            parent_ = parent;
            level_ = level; 

            // DEBUG
            std::cout << "node id " << idx_ << " parent id " << parent_ << " level " << level_ << std::endl;

            // return the neighbors
            return neighbors_;
          } else {
            // don't return neighbors
            std::vector<std::size_t> tmp;
            return tmp;
          }
        }

}}}

