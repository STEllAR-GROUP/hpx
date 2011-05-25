// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_UTIL_APPLY_20100917T1337)
#define PXGL_UTIL_APPLY_20100917T1337

#include <boost/lexical_cast.hpp>

namespace pxgl { namespace px {
  //////////////////////////////////////////////////////////////////////////////
  // Apply helpers
  template <typename Action>
  inline void apply(hpx::naming::id_type target)
  {
    hpx::applier::apply<Action>(target);
  }
  
  template <typename Action, typename Arg0>
  inline void apply(hpx::naming::id_type target, Arg0 arg0)
  {
    hpx::applier::apply<Action>(target, arg0);
  }
  
  template <typename Action, typename Arg0, typename Arg1>
  inline void apply(hpx::naming::id_type target, Arg0 arg0, Arg1 arg1)
  {
    hpx::applier::apply<Action>(target, arg0, arg1);
  }

  template <typename Action, typename Arg0, typename Arg1, typename Arg2>
  inline void apply(hpx::naming::id_type target, Arg0 arg0, Arg1 arg1, Arg2 arg2)
  {
    hpx::applier::apply<Action>(target , arg0, arg1, arg2);
  }

  template <typename Action, typename Arg0, typename Arg1, typename Arg2,
            typename Arg3>
  inline void apply(hpx::naming::id_type target, Arg0 arg0, Arg1 arg1, Arg2 arg2, Arg3 arg3)
  {
    hpx::applier::apply<Action>(target , arg0, arg1, arg2, arg3);
  }
}};

#endif

