//  Copyright (c) 2019-2020 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/synchronization/force_linking.hpp>

namespace hpx { namespace synchronization {

    force_linking_helper& force_linking()
    {
        static force_linking_helper helper{&hpx::detail::intrusive_ptr_add_ref};
        return helper;
    }
}}    // namespace hpx::synchronization
