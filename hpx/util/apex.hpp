//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#ifdef HPX_HAVE_APEX
#include <apex.hpp>
#include "apex_api.hpp"
#endif


namespace hpx { namespace util
{
#ifdef HPX_HAVE_APEX
    inline void apex_init()
    {
        apex::init(NULL);
        apex::set_node_id(hpx::get_locality_id());
    }

    inline void apex_finalize()
    {
        apex::finalize();
    }

    struct apex_wrapper
    {
        apex_wrapper(char const* const name)
          : name_(name), stopped(false)
        {
            profiler_ = apex::start(name_);
        }
        ~apex_wrapper()
        {
            stop();
        }

        void stop() {
            if(!stopped) {
                stopped = true;
                apex::stop(profiler_);
            }
        }

        void yield() {
            if(!stopped) {
                stopped = true;
                apex::yield(profiler_);
            }
        }

        char const* const name_;
        bool stopped;
        apex::profiler * profiler_;
    };

    struct apex_wrapper_init
    {
        apex_wrapper_init(int argc, char **argv)
        {
            apex::init(argc, argv, NULL);
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

