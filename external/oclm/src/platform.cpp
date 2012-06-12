
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <oclm/platform.hpp>

#include <iostream>

namespace oclm
{
    platform_manager::platform_manager()
    {
        cl_uint n = 0;
        cl_int err = ::clGetPlatformIDs(0, NULL, &n);
        OCLM_THROW_IF_EXCEPTION(err, "clGetPlatformIDs");

        std::vector<cl_platform_id> platform_ids(n);

        err = ::clGetPlatformIDs(n, &platform_ids[0], NULL);
        OCLM_THROW_IF_EXCEPTION(err, "clGetPlatformIDs");

        platforms.resize(n);
        std::copy(platform_ids.begin(), platform_ids.end(), platforms.begin());

        default_platform = platforms[0];
        default_device = default_platform.devices[0];
    }

    platform_manager & platform_manager::get()
    {
        util::static_<platform_manager> man;
        return man.get();
    }
}
