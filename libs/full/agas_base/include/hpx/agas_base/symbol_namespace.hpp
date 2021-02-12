////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#include <hpx/agas_base/agas_fwd.hpp>
#include <hpx/agas_base/server/symbol_namespace.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/id_type.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace agas {

    struct symbol_namespace
    {
        using server_type = server::symbol_namespace;
        using iterate_names_return_type =
            std::map<std::string, naming::id_type>;

        static naming::gid_type get_service_instance(
            std::uint32_t service_locality_id);

        static naming::gid_type get_service_instance(
            naming::gid_type const& dest, error_code& ec = throws);

        static naming::gid_type get_service_instance(
            naming::id_type const& dest)
        {
            return get_service_instance(dest.get_gid());
        }

        static bool is_service_instance(naming::gid_type const& gid);

        static bool is_service_instance(naming::id_type const& id)
        {
            return is_service_instance(id.get_gid());
        }

        static naming::id_type symbol_namespace_locality(
            std::string const& key);

        symbol_namespace();
        ~symbol_namespace() = default;

        naming::address_type ptr() const;
        naming::address addr() const;
        naming::id_type gid() const;

        hpx::future<bool> bind_async(std::string key, naming::gid_type gid);
        bool bind(std::string key, naming::gid_type gid);

        hpx::future<naming::id_type> resolve_async(std::string key) const;
        naming::id_type resolve(std::string key) const;

        hpx::future<naming::id_type> unbind_async(std::string key);
        naming::id_type unbind(std::string key);

        hpx::future<bool> on_event(std::string const& name,
            bool call_for_past_events, hpx::id_type lco);

        hpx::future<iterate_names_return_type> iterate_async(
            std::string const& pattern) const;
        iterate_names_return_type iterate(std::string const& pattern) const;

        void register_server_instance(std::uint32_t locality_id);
        void unregister_server_instance(error_code& ec);

        server::symbol_namespace& get_service()
        {
            return *server_;
        }

    private:
        std::unique_ptr<server_type> server_;
    };

}}    // namespace hpx::agas

#include <hpx/config/warnings_suffix.hpp>
