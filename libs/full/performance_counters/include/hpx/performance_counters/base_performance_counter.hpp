//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/component_action.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/components_base/server/component_base.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/server/base_performance_counter.hpp>

///////////////////////////////////////////////////////////////////////////////
//[performance_counter_base_class
namespace hpx { namespace performance_counters {
    template <typename Derived>
    class base_performance_counter;
}}    // namespace hpx::performance_counters
//]

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters {
    template <typename Derived>
    class base_performance_counter
      : public hpx::performance_counters::server::base_performance_counter
      , public hpx::components::component_base<Derived>
    {
    private:
        using base_type = hpx::components::component_base<Derived>;

    public:
        using type_holder = Derived;
        using base_type_holder =
            hpx::performance_counters::server::base_performance_counter;

        base_performance_counter() = default;

        base_performance_counter(
            hpx::performance_counters::counter_info const& info)
          : base_type_holder(info)
        {
        }

        // Disambiguate finalize() which is implemented in both base classes
        void finalize()
        {
            base_type_holder::finalize();
            base_type::finalize();
        }

        hpx::naming::address get_current_address() const
        {
            return hpx::naming::address(
                hpx::naming::get_gid_from_locality_id(hpx::get_locality_id()),
                hpx::components::get_component_type<Derived>(),
                const_cast<Derived*>(static_cast<Derived const*>(this)));
        }
    };
}}    // namespace hpx::performance_counters
