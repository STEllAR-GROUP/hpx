//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/async.hpp>
#include <hpx/lcos/future_wait.hpp>

#include "../stubs/point.hpp"
#include "../../fname.h"
#include "point.hpp"

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include <iostream>
#include <fstream>

extern "C" {void FNAME(SETUP)(int *,int *); }

///////////////////////////////////////////////////////////////////////////////
namespace gtc { namespace server
{
    void point::setup(std::size_t numberpe,std::size_t mype)
    {
      int t1 = numberpe;
      int t2 = mype;
      FNAME(SETUP)(&t1,&t2);
    }
}}

