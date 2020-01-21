//  Copyright (c) 2019 The STE||AR GROUP
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_CONFIG_FORCE_LINKING_HPP
#define HPX_CONFIG_FORCE_LINKING_HPP

namespace hpx { namespace config {
    struct force_linking_helper
    {
        const char* const hpx_version;
        const char* const boost_version;
    };

    force_linking_helper& force_linking();
}}    // namespace hpx::config

#endif
