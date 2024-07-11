//  Copyright (c) 2007-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/plugin_factories/parcelport_factory_base.hpp>

#include <algorithm>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::plugins {

    std::vector<parcelport_factory_base*>& get_parcelport_factories()
    {
        static std::vector<plugins::parcelport_factory_base*> factories;
        return factories;
    }

    void add_parcelport_factory(parcelport_factory_base* factory)
    {
        auto& factories = get_parcelport_factories();
        if (std::find(factories.begin(), factories.end(), factory) !=
            factories.end())
        {
            return;
        }
        factories.push_back(factory);
    }
}    // namespace hpx::plugins
