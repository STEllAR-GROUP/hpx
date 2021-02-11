//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/ini/ini.hpp>
#include <hpx/modules/filesystem.hpp>
#include <hpx/modules/plugin.hpp>
#include <hpx/runtime_configuration/agas_service_mode.hpp>
#include <hpx/runtime_configuration/component_registry_base.hpp>
#include <hpx/runtime_configuration/plugin_registry_base.hpp>
#include <hpx/runtime_configuration/runtime_configuration_fwd.hpp>
#include <hpx/runtime_configuration/runtime_mode.hpp>
#include <hpx/runtime_configuration/static_factory_data.hpp>
#include <hpx/runtime_configuration_local/runtime_configuration_local.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    // The runtime_configuration class is a wrapper for the runtime
    // configuration data allowing to extract configuration information in a
    // more convenient way
    class runtime_configuration
      : public hpx::local::detail::runtime_configuration
    {
    public:
        runtime_configuration(char const* argv0_, runtime_mode mode);

        std::vector<std::shared_ptr<plugins::plugin_registry_base>>
        load_modules(
            std::vector<std::shared_ptr<components::component_registry_base>>&
                component_registries);

        void load_components_static(
            std::vector<components::static_factory_load_data_type> const&
                static_modules);

        // Returns the AGAS mode of this locality, returns either hosted (for
        // localities connecting to a remote AGAS server) or bootstrap for the
        // locality hosting the AGAS server.
        agas::service_mode get_agas_service_mode() const;

        // initial number of localities
        std::uint32_t get_num_localities() const;
        void set_num_localities(std::uint32_t);

        // should networking be enabled
        bool enable_networking() const;

        // sequence number of first usable pu
        std::uint32_t get_first_used_core() const;
        void set_first_used_core(std::uint32_t);

        // Get the size of the ipc parcelport data buffer cache
        std::size_t get_ipc_data_buffer_cache_size() const;

        // Get AGAS client-side local cache size
        std::size_t get_agas_local_cache_size(
            std::size_t dflt = HPX_AGAS_LOCAL_CACHE_SIZE) const;

        bool get_agas_caching_mode() const;

        bool get_agas_range_caching_mode() const;

        std::size_t get_agas_max_pending_refcnt_requests() const;

        // Enable global lock tracking
        bool enable_global_lock_detection() const;

        // Return the endianness to be used for out-serialization
        std::string get_endian_out() const;

        // Return maximally allowed message sizes
        std::uint64_t get_max_inbound_message_size() const;
        std::uint64_t get_max_outbound_message_size() const;

        std::map<std::string, hpx::util::plugin::dll>& modules()
        {
            return modules_;
        }

    private:
        void pre_initialize_ini() override;

        void load_component_paths(
            std::vector<std::shared_ptr<plugins::plugin_registry_base>>&
                plugin_registries,
            std::vector<std::shared_ptr<components::component_registry_base>>&
                component_registries,
            std::string const& component_base_paths,
            std::string const& component_path_suffixes,
            std::set<std::string>& component_paths,
            std::map<std::string, filesystem::path>& basenames);

        void load_component_path(
            std::vector<std::shared_ptr<plugins::plugin_registry_base>>&
                plugin_registries,
            std::vector<std::shared_ptr<components::component_registry_base>>&
                component_registries,
            std::string const& path, std::set<std::string>& component_paths,
            std::map<std::string, filesystem::path>& basenames);

    public:
        runtime_mode mode_;

    private:
        mutable std::uint32_t num_localities;
        std::map<std::string, hpx::util::plugin::dll> modules_;
    };
}}    // namespace hpx::util
