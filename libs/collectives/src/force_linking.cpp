//  Copyright (c) 2020 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/collectives/detail/communicator.hpp>
#include <hpx/collectives/force_linking.hpp>

namespace hpx { namespace collectives {
    // reference all symbols that have to be explicitly linked with the core
    // library
    force_linking_helper& force_linking()
    {
        static force_linking_helper helper{&hpx::lcos::detail::dummy};
        return helper;
    }
}}    // namespace hpx::collectives
