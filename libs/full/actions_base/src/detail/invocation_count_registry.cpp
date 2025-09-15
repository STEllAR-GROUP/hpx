//  Copyright (c) 2007-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/actions_base/detail/invocation_count_registry.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/type_support.hpp>

#include <regex>
#include <string>
#include <utility>

namespace hpx::actions::detail {

    invocation_count_registry& invocation_count_registry::local_instance()
    {
        hpx::util::static_<invocation_count_registry, local_tag> registry;
        return registry.get();
    }

#if defined(HPX_HAVE_NETWORKING)
    invocation_count_registry& invocation_count_registry::remote_instance()
    {
        hpx::util::static_<invocation_count_registry, remote_tag> registry;
        return registry.get();
    }
#endif

    void invocation_count_registry::register_class(
        std::string const& name, get_invocation_count_type fun)
    {
        if (name.empty())
        {
            HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                "invocation_count_registry::register_class",
                "Cannot register an action with an empty name");
        }

        if (map_.find(name) == map_.end())
        {
            map_.emplace(name, fun);
        }
    }

    invocation_count_registry::get_invocation_count_type
    invocation_count_registry::get_invocation_counter(
        std::string const& name) const
    {
        map_type::const_iterator const it = map_.find(name);
        if (it == map_.end())
        {
            HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                "invocation_count_registry::get_invocation_counter",
                "unknown action type");
        }
        return it->second;
    }
}    // namespace hpx::actions::detail
