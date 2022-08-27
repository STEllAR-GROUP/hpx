//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_TCP)
#include <hpx/plugin/traits/plugin_config_data.hpp>

#include <hpx/parcelport_tcp/connection_handler.hpp>
#include <hpx/plugin_factories/parcelport_factory.hpp>

namespace hpx::traits {

    // Inject additional configuration data into the factory registry for this
    // type. This information ends up in the system wide configuration database
    // under the plugin specific section:
    //
    //      [hpx.parcel.tcp]
    //      ...
    //      priority = 1
    //
    template <>
    struct plugin_config_data<hpx::parcelset::policies::tcp::connection_handler>
    {
        static constexpr char const* priority() noexcept
        {
            return "1";
        }

        static constexpr void init(int* /* argc */, char*** /* argv */,
            util::command_line_handling& /* cfg */) noexcept
        {
        }

        static constexpr void destroy() noexcept {}

        static constexpr char const* call() noexcept
        {
            return "";
        }
    };
}    // namespace hpx::traits

HPX_REGISTER_PARCELPORT(hpx::parcelset::policies::tcp::connection_handler, tcp)

#endif
