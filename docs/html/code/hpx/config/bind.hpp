//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_CONFIG_BIND_SEP_23_2011_1153AM)
#define HPX_CONFIG_BIND_SEP_23_2011_1153AM

#include <hpx/config.hpp>

#if defined(HPX_UTIL_BIND)
#  include <hpx/util/bind.hpp>
#  include <hpx/util/protect.hpp>
#else
#  if !defined(HPX_HAVE_CXX11_STD_BIND)
#    if defined(HPX_PHOENIX_BIND)
#      include <boost/phoenix/bind.hpp>
#      include <boost/phoenix/scope/lambda.hpp>
#    else
#      include <boost/bind.hpp>
#      include <hpx/util/protect.hpp>
#    endif
#  else
#    include <functional>
#    include <hpx/util/protect.hpp>
#  endif
#endif

#endif


