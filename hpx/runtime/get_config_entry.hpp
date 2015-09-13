//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_GET_CONFIG_ENTRY_SEP_01_2015_0638PM)
#define HPX_GET_CONFIG_ENTRY_SEP_01_2015_0638PM

#include <hpx/config.hpp>

#include <string>
#include <cstdlib>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    /// Retrieve the string value of a configuration entry as given by \p key.
    HPX_API_EXPORT std::string get_config_entry(std::string const& key,
        std::string const& dflt);
    /// Retrieve the integer value of a configuration entry as given by \p key.
    HPX_API_EXPORT std::string get_config_entry(std::string const& key,
        std::size_t dflt);
}

#endif
