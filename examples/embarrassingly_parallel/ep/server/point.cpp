//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/lcos/async_future_wait.hpp>

#include "../stubs/point.hpp"
#include "point.hpp"

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include <iostream>
#include <fstream>

///////////////////////////////////////////////////////////////////////////////
namespace ep { namespace server
{
    void point::bfs(std::size_t scale)
    {
      for (std::size_t i=0;i<scale;i++) {
        sum_ += i+(i-1)*(i+1)/(i+2);
      }
    }
}}

