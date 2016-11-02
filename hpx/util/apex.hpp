//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once // prevent multiple inclusions of this header file.

#include <hpx/config.hpp>
#include <hpx/runtime/get_locality_id.hpp>
#include <hpx/util/thread_description.hpp>
#include <hpx/runtime/get_num_localities.hpp>
#include <hpx/runtime/startup_function.hpp>

#ifdef HPX_HAVE_APEX
#include "apex_api.hpp"
#endif

static void hpx_util_apex_init_startup(void) {
	apex::init(nullptr, hpx::get_locality_id(), hpx::get_initial_num_localities());
}

namespace hpx { namespace util
{
#ifdef HPX_HAVE_APEX
    inline void apex_init()
    {
	    hpx_util_apex_init_startup();
		//hpx::register_pre_startup_function(&hpx_util_apex_init_startup);
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
	    	//apex::init(nullptr, hpx::get_locality_id(), hpx::get_initial_num_localities());
			hpx::register_pre_startup_function(&hpx_util_apex_init_startup);
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

