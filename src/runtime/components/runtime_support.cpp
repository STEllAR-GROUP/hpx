//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/components/runtime_support.hpp>
#include <hpx/runtime/components/component_registry.hpp>

#include <mutex>
#include <utility>

HPX_PLUGIN_EXPORT_LIST(HPX_PLUGIN_COMPONENT_PREFIX, factory)
HPX_REGISTER_REGISTRY_MODULE()

namespace hpx { namespace components { namespace server
{
    void runtime_support::add_pre_startup_function(startup_function_type f)
    {
        if (!f.empty())
        {
            std::lock_guard<lcos::local::spinlock> l(globals_mtx_);
            pre_startup_functions_.push_back(std::move(f));
        }
    }

    void runtime_support::add_startup_function(startup_function_type f)
    {
        if (!f.empty())
        {
            std::lock_guard<lcos::local::spinlock> l(globals_mtx_);
            startup_functions_.push_back(std::move(f));
        }
    }

    void runtime_support::add_pre_shutdown_function(shutdown_function_type f)
    {
        if (!f.empty())
        {
            std::lock_guard<lcos::local::spinlock> l(globals_mtx_);
            pre_shutdown_functions_.push_back(std::move(f));
        }
    }

    void runtime_support::add_shutdown_function(shutdown_function_type f)
    {
        if (!f.empty())
        {
            std::lock_guard<lcos::local::spinlock> l(globals_mtx_);
            shutdown_functions_.push_back(std::move(f));
        }
    }
}}}


