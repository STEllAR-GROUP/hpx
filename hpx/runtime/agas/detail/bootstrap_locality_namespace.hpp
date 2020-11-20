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

#include <hpx/agas/agas_fwd.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/agas/locality_namespace.hpp>
#include <hpx/runtime/agas/server/locality_namespace.hpp>
#include <hpx/runtime/parcelset/locality.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace hpx { namespace agas { namespace detail
{
    struct bootstrap_locality_namespace : locality_namespace
    {
        explicit bootstrap_locality_namespace(
            server::primary_namespace* primary);

        naming::address::address_type ptr() const override
        {
            return reinterpret_cast<naming::address::address_type>(&server_);
        }
        naming::address addr() const override;
        naming::id_type gid() const override;

        std::uint32_t allocate(parcelset::endpoints_type const& endpoints,
            std::uint64_t count, std::uint32_t num_threads,
            naming::gid_type const& suggested_prefix) override;

        void free(naming::gid_type const& locality) override;

        std::vector<std::uint32_t> localities() override;

        parcelset::endpoints_type resolve_locality(
            naming::gid_type const& locality) override;

        std::uint32_t get_num_localities() override;
        hpx::future<std::uint32_t> get_num_localities_async() override;

        std::vector<std::uint32_t> get_num_threads() override;
        hpx::future<std::vector<std::uint32_t>> get_num_threads_async()
            override;

        std::uint32_t get_num_overall_threads() override;
        hpx::future<std::uint32_t> get_num_overall_threads_async() override;

        naming::gid_type statistics_counter(std::string name) override;

        void register_counter_types() override;

        void register_server_instance(std::uint32_t locality_id) override;

        void unregister_server_instance(error_code& ec) override;

    private:
        server::locality_namespace server_;
    };
}}}

