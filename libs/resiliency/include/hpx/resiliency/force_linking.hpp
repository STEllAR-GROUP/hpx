//  Copyright (c) 2019 The STE||AR GROUP
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RESILIENCY_FORCE_LINKING_HPP
#define HPX_RESILIENCY_FORCE_LINKING_HPP

#include <string>

namespace hpx { namespace resiliency {

    struct force_linking_helper
    {
        unsigned int (*major_version)();
        unsigned int (*minor_version)();
        unsigned int (*subminor_version)();
        unsigned long (*full_version)();
        std::string (*full_version_str)();
    };

    force_linking_helper& force_linking();
}}    // namespace hpx::resiliency

#endif
