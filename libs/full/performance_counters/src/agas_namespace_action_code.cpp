//  Copyright (c) 2012-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/errors.hpp>
#include <hpx/performance_counters/agas_namespace_action_code.hpp>
#include <hpx/performance_counters/counters.hpp>

#include <cstddef>
#include <string>

namespace hpx { namespace agas { namespace detail {

    // get action code from counter type
    namespace_action_code retrieve_action_code(
        std::string const& name, error_code& ec)
    {
        performance_counters::counter_path_elements p;
        performance_counters::get_counter_path_elements(name, p, ec);
        if (ec)
            return invalid_request;

        if (p.objectname_ != "agas")
        {
            HPX_THROWS_IF(ec, bad_parameter, "retrieve_action_code",
                "unknown performance counter (unrelated to AGAS)");
            return invalid_request;
        }

        // component_ns
        for (std::size_t i = 0; i != num_component_namespace_services; ++i)
        {
            if (p.countername_ == component_namespace_services[i].name_)
                return component_namespace_services[i].code_;
        }

        // locality_ns
        for (std::size_t i = 0; i != num_locality_namespace_services; ++i)
        {
            if (p.countername_ == locality_namespace_services[i].name_)
                return locality_namespace_services[i].code_;
        }

        // primary_ns
        for (std::size_t i = 0; i != num_primary_namespace_services; ++i)
        {
            if (p.countername_ == primary_namespace_services[i].name_)
                return primary_namespace_services[i].code_;
        }

        // symbol_ns
        for (std::size_t i = 0; i != num_symbol_namespace_services; ++i)
        {
            if (p.countername_ == symbol_namespace_services[i].name_)
                return symbol_namespace_services[i].code_;
        }

        HPX_THROWS_IF(ec, bad_parameter, "retrieve_action_code",
            "unknown performance counter (unrelated to AGAS)");
        return invalid_request;
    }

    // get service action code from counter type
    namespace_action_code retrieve_action_service_code(
        std::string const& name, error_code& ec)
    {
        performance_counters::counter_path_elements p;
        performance_counters::get_counter_path_elements(name, p, ec);
        if (ec)
            return invalid_request;

        if (p.objectname_ != "agas")
        {
            HPX_THROWS_IF(ec, bad_parameter, "retrieve_action_service_code",
                "unknown performance counter (unrelated to AGAS)");
            return invalid_request;
        }

        // component_ns
        for (std::size_t i = 0; i != num_component_namespace_services; ++i)
        {
            if (p.countername_ == component_namespace_services[i].name_)
                return component_namespace_services[i].service_code_;
        }

        // locality_ns
        for (std::size_t i = 0; i != num_locality_namespace_services; ++i)
        {
            if (p.countername_ == locality_namespace_services[i].name_)
                return locality_namespace_services[i].service_code_;
        }

        // primary_ns
        for (std::size_t i = 0; i != num_primary_namespace_services; ++i)
        {
            if (p.countername_ == primary_namespace_services[i].name_)
                return primary_namespace_services[i].service_code_;
        }

        // symbol_ns
        for (std::size_t i = 0; i != num_symbol_namespace_services; ++i)
        {
            if (p.countername_ == symbol_namespace_services[i].name_)
                return symbol_namespace_services[i].service_code_;
        }

        HPX_THROWS_IF(ec, bad_parameter, "retrieve_action_service_code",
            "unknown performance counter (unrelated to AGAS)");
        return invalid_request;
    }
}}}    // namespace hpx::agas::detail
