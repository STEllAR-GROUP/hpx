// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_UTIL_HPX_20100917T1337)
#define PXGL_UTIL_HPX_20100917T1337

#include <boost/lexical_cast.hpp>

////////////////////////////////////////////////////////////////////////////////
// Run-time system (HPX) helper
namespace pxgl { namespace rts {
  template <typename T>
  inline void get_ini_option(T& x, std::string const option_name)
  {
    if (!option_name.empty())
    {
      x = boost::lexical_cast<T>(
          hpx::get_runtime().get_config().get_entry(option_name, x));
    }
  }

  inline void busy_wait(double wait_time)
  {
    hpx::util::high_resolution_timer t;
    double start_time = t.elapsed();
    double current = 0;
    do {
      current = t.elapsed();
    } while (current - start_time < wait_time * 1e-6);
  }
}};

#endif

