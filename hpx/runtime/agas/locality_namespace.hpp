////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2012-2013 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>

#include <hpx/futures/future.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/parcelset/locality.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace hpx { namespace agas
{
    struct locality_namespace
    {
        virtual ~locality_namespace() = default;

        virtual naming::address::address_type ptr() const = 0;
        virtual naming::address addr() const = 0;
        virtual naming::id_type gid() const = 0;

        virtual std::uint32_t allocate(
            parcelset::endpoints_type const& endpoints, std::uint64_t count,
            std::uint32_t num_threads,
            naming::gid_type const& suggested_prefix) = 0;

        virtual void free(naming::gid_type const& locality) = 0;

        virtual std::vector<std::uint32_t> localities() = 0;

        virtual parcelset::endpoints_type resolve_locality(
            naming::gid_type const& locality) = 0;

        virtual std::uint32_t get_num_localities() = 0;
        virtual hpx::future<std::uint32_t> get_num_localities_async() = 0;

        virtual std::vector<std::uint32_t> get_num_threads() = 0;
        virtual hpx::future<std::vector<std::uint32_t> > get_num_threads_async() = 0;

        virtual std::uint32_t get_num_overall_threads() = 0;
        virtual hpx::future<std::uint32_t> get_num_overall_threads_async() = 0;

        virtual naming::gid_type statistics_counter(std::string name) = 0;

        virtual void register_counter_types() {}

        virtual void register_server_instance(std::uint32_t /*locality_id*/) {}

        virtual void unregister_server_instance(error_code& /*ec*/) {}
    };
}}


