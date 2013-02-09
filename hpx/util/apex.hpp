//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#ifdef HPX_HAVE_APEX
#include <apex.h>
#endif

namespace hpx { namespace util
{
#ifdef HPX_HAVE_APEX
    inline void apex_init()
    {
        apex_set_node_id(hpx::get_locality_id());
    }

    struct apex_wrapper
    {
        apex_wrapper(char const* const name)
          : name_(name)
        {
            apex_start(name_);
        }
        ~apex_wrapper()
        {
            apex_stop(name_);
        }

        char const* const name_;
    };
#else
    inline void apex_init() {}

    struct apex_wrapper
    {
        apex_wrapper(char const* const name) {}
        ~apex_wrapper() {}
    };
#endif
}}

