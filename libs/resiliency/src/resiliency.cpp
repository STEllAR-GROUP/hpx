//  Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
//                     LLC (NTESS).
//  Copyright (c) 2018-2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/resiliency/version.hpp>

#include <string>

namespace hpx { namespace resiliency { namespace experimental {

    unsigned int major_version()
    {
        return HPX_RESILIENCY_VERSION_MAJOR;
    }

    unsigned int minor_version()
    {
        return HPX_RESILIENCY_VERSION_MINOR;
    }

    unsigned int subminor_version()
    {
        return HPX_RESILIENCY_VERSION_SUBMINOR;
    }

    unsigned long full_version()
    {
        return HPX_RESILIENCY_VERSION_FULL;
    }

    std::string full_version_str()
    {
        return std::to_string(HPX_RESILIENCY_VERSION_MAJOR) + "." +
            std::to_string(HPX_RESILIENCY_VERSION_MINOR) + "." +
            std::to_string(HPX_RESILIENCY_VERSION_SUBMINOR);
    }
}}}    // namespace hpx::resiliency::experimental
