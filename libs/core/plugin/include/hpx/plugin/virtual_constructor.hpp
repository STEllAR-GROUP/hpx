//  Copyright Vladimir Prus 2004.
//  Copyright (c) 2005-2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/plugin/config.hpp>
#include <hpx/datastructures/any.hpp>
#include <hpx/type_support/pack.hpp>

#include <map>
#include <memory>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::util::plugin {

    ///////////////////////////////////////////////////////////////////////////
    using exported_plugins_type = std::map<std::string, hpx::any_nonser>;
    typedef exported_plugins_type*(HPX_PLUGIN_API* get_plugins_list_type)();
    typedef exported_plugins_type*(HPX_PLUGIN_API get_plugins_list_np)();
    using dll_handle = shared_ptr<get_plugins_list_np>;

    ///////////////////////////////////////////////////////////////////////////
    template <typename BasePlugin>
    struct virtual_constructor
    {
        using type = hpx::util::pack<>;
    };

    template <typename BasePlugin>
    using virtual_constructor_t =
        typename virtual_constructor<BasePlugin>::type;
}    // namespace hpx::util::plugin
