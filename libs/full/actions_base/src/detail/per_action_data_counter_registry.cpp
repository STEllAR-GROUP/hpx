//  Copyright (c) 2016-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_COUNTERS) &&                                   \
    defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS) &&                            \
    defined(HPX_HAVE_NETWORKING)
#include <hpx/actions_base/detail/per_action_data_counter_registry.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/type_support/static.hpp>

#include <cstdint>
#include <string>
#include <utility>

namespace hpx::actions::detail {

    ///////////////////////////////////////////////////////////////////////////
    per_action_data_counter_registry&
    per_action_data_counter_registry::instance()
    {
        hpx::util::static_<per_action_data_counter_registry, tag> registry;
        return registry.get();
    }

    void per_action_data_counter_registry::register_class(std::string name)
    {
        if (name.empty())
        {
            HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                "per_action_data_counter_registry::register_class",
                "Cannot register an action with an empty name");
        }

        auto const it = map_.find(name);
        if (it == map_.end())
        {
            map_.emplace(HPX_MOVE(name));
        }
    }

    per_action_data_counter_registry::counter_function_type
    per_action_data_counter_registry::get_counter(std::string const& name,
        hpx::function<std::int64_t(std::string const&, bool)> const& f) const
    {
        map_type::const_iterator const it = map_.find(name);
        if (it == map_.end())
        {
            HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                "per_action_data_counter_registry::get_receive_counter",
                "unknown action type");
        }
        return hpx::bind_front(f, name);
    }
}    // namespace hpx::actions::detail

#endif
