//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_USER_MAIN_CONFIG_JUL_26_2015_0222PM)
#define HPX_USER_MAIN_CONFIG_JUL_26_2015_0222PM

#include <hpx/config.hpp>
#include <vector>
#include <string>

namespace hpx_startup
{
    // Allow applications to add configuration settings if HPX_MAIN is set
    HPX_EXPORT extern std::vector<std::string> (*user_main_config_function)(
        std::vector<std::string> const&);

    inline std::vector<std::string>
        user_main_config(std::vector<std::string> const& cfg)
    {
        return user_main_config_function ? user_main_config_function(cfg) : cfg;
    }
}

#endif
