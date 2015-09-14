//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config/defines.hpp>

#ifdef HPX_HAVE_APEX
#include <apex.hpp>
#endif

#ifdef HPX_HAVE_APEX
extern bool use_ittnotify_api;
#endif

namespace hpx { namespace util
{
#ifdef HPX_HAVE_APEX
    inline void apex_init()
    {
        if (use_ittnotify_api)
        {
            apex::init();
            apex::set_node_id(hpx::get_locality_id());
        }
    }

    inline void apex_finalize()
    {
        if (use_ittnotify_api)
            apex::finalize();
    }

    struct apex_wrapper
    {
        apex_wrapper(char const* const name)
          : name_(name)
        {
            if (use_ittnotify_api)
                apex::start(name_);
        }
        ~apex_wrapper()
        {
            if (use_ittnotify_api)
                apex::stop(name_);
        }

        char const* const name_;
    };

    struct apex_wrapper_init
    {
        apex_wrapper_init(int argc, char **argv)
        {
            apex::init(argc, argv);
        }
        ~apex_wrapper_init()
        {
            apex::finalize();
        }
    };
#else
    inline void apex_init() {}
    inline void apex_finalize() {}

    struct apex_wrapper
    {
        apex_wrapper(char const* const name) {}
        ~apex_wrapper() {}
    };

    struct apex_wrapper_init
    {
        apex_wrapper_init(int argc, char **argv) {}
        ~apex_wrapper_init() {}
    };
#endif
}}

