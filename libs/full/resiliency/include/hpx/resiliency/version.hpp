//  Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
//                     LLC (NTESS).
//  Copyright (c) 2018-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
//  The version of HPX_RESILILIENCY
//
//  HPX_RESILIENCY_VERSION_FULL & 0x0000FF is the sub-minor version
//  HPX_RESILIENCY_VERSION_FULL & 0x00FF00 is the minor version
//  HPX_RESILIENCY_VERSION_FULL & 0xFF0000 is the major version
//
//  HPX_RESILIENCY_VERSION_DATE   YYYYMMDD is the date of the release
//                               (estimated release date for master branch)
//
#define HPX_RESILIENCY_VERSION_FULL 0x010000

#define HPX_RESILIENCY_VERSION_MAJOR 1
#define HPX_RESILIENCY_VERSION_MINOR 0
#define HPX_RESILIENCY_VERSION_SUBMINOR 0

#define HPX_RESILIENCY_VERSION_DATE 20190823

namespace hpx { namespace resiliency { namespace experimental {

    // return version of this library
    HPX_EXPORT unsigned int major_version();
    HPX_EXPORT unsigned int minor_version();
    HPX_EXPORT unsigned int subminor_version();
    HPX_EXPORT unsigned long full_version();
    HPX_EXPORT std::string full_version_str();

}}}    // namespace hpx::resiliency::experimental
