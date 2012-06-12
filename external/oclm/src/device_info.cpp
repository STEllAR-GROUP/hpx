
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <oclm/device_info.hpp>

namespace oclm
{
    const device_info<CL_DEVICE_PROFILE>    device_profile = {};
    const device_info<CL_DEVICE_VERSION>    device_version = {};
    const device_info<CL_DEVICE_NAME>       device_name = {};
    const device_info<CL_DEVICE_VENDOR>     device_vendor = {};
    const device_info<CL_DEVICE_EXTENSIONS> device_extensions = {};
}
