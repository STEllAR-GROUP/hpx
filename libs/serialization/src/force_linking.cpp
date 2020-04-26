//  Copyright (c) 2019-2020 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/serialization/force_linking.hpp>
#include <hpx/serialization/serializable_any.hpp>

namespace hpx { namespace serialization {
    force_linking_helper& force_linking()
    {
        static force_linking_helper helper{&hpx::util::hash_any::dummy};
        return helper;
    }
}}    // namespace hpx::serialization
