//  Copyright (c) 2012 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/async.hpp>

#include <hpx/lcos/future_wait.hpp>

#include "../../fname.h"
#include "partition.hpp"

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include <iostream>
#include <fstream>

extern "C" {
  void FNAME(nekproxy)(void* opaque_ptr_to_class, int *,int *);
}

///////////////////////////////////////////////////////////////////////////////
inline void set_description(char const* test_name)
{
    hpx::threads::set_thread_description(hpx::threads::get_self_id(), test_name);
}

///////////////////////////////////////////////////////////////////////////////
namespace nekbone { namespace server
{
    void partition::loop_wrapper(std::size_t numberpe,std::size_t mype,
                      std::vector<hpx::naming::id_type> const& components)
    {
      item_ = mype;
      components_ = components;

      int t2 = static_cast<int>(numberpe);
      int t1 = static_cast<int>(mype);

      FNAME(nekproxy)(static_cast<void*>(this), &t1,&t2);

    }

}}

