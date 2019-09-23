//  Copyright (c) 2019 The STE||AR GROUP
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/resiliency/force_linking.hpp>
#include <hpx/resiliency/version.hpp>

namespace hpx { namespace resiliency {
    force_linking_helper& force_linking()
    {
        static force_linking_helper helper{major_version, minor_version,
            subminor_version, full_version, full_version_str};
        return helper;
    }
}}    // namespace hpx::resiliency
