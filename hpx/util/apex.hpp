//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once // prevent multiple inclusions of this header file.

#include <hpx/config.hpp>
#include <hpx/runtime/get_locality_id.hpp>
#include <hpx/util/thread_description.hpp>

#ifdef HPX_HAVE_APEX
#include "apex_api.hpp"
#endif

namespace hpx { namespace util
{
#ifdef HPX_HAVE_APEX
    inline void apex_init()
    {
        apex::init(nullptr);
        apex::set_node_id(hpx::get_locality_id());
    }

    inline void apex_finalize()
    {
        apex::finalize();
    }

    struct apex_wrapper
    {
        apex_wrapper(thread_description const& name)
          : name_(name), stopped(false)
        {
            if (name_.kind() == thread_description::data_type_description)
            {
                profiler_ = apex::start(name_.get_description());
            }
            else
            {
                profiler_ = apex::start(
                    apex_function_address(name_.get_address()));
            }
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

        thread_description name_;
        bool stopped;
        apex::profiler * profiler_;
    };

    struct apex_wrapper_init
    {
        apex_wrapper_init(int argc, char **argv)
        {
            apex::init(argc, argv, nullptr);
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
        apex_wrapper(thread_description const& name) {}
        ~apex_wrapper() {}
    };

    struct apex_wrapper_init
    {
        apex_wrapper_init(int argc, char **argv) {}
        ~apex_wrapper_init() {}
    };
#endif
}}

