//  Copyright (c) 2005-2014 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/ini/ini.hpp>
#include <hpx/modules/filesystem.hpp>

#include <map>
#include <memory>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    bool handle_ini_file(section& ini, std::string const& loc);
    bool handle_ini_file_env(
        section& ini, char const* env_var, char const* file_suffix = nullptr);

    ///////////////////////////////////////////////////////////////////////////
    // read system and user specified ini files
    //
    // returns true if at least one alternative location has been read
    // successfully
    bool init_ini_data_base(section& ini, std::string& hpx_ini_file);
}}    // namespace hpx::util
