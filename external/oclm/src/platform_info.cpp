
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <oclm/platform_info.hpp>

namespace oclm
{
    const platform_info<CL_PLATFORM_PROFILE> platform_profile =
        platform_info<CL_PLATFORM_PROFILE>();
    const platform_info<CL_PLATFORM_VERSION> platform_version =
        platform_info<CL_PLATFORM_VERSION>();
    const platform_info<CL_PLATFORM_NAME> platform_name =
        platform_info<CL_PLATFORM_NAME>();
    const platform_info<CL_PLATFORM_VENDOR> platform_vendor =
        platform_info<CL_PLATFORM_VENDOR>();
    const platform_info<CL_PLATFORM_EXTENSIONS> platform_extensions =
        platform_info<CL_PLATFORM_EXTENSIONS>();
}
