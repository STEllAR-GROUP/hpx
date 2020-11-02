//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/components_base/component_type.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/util/ios_flags_saver.hpp>

#include <iomanip>
#include <iostream>

namespace hpx { namespace naming {

    // this is defined in this module as its implementation relies on
    // components::get_component_type_name()
    std::ostream& operator<<(std::ostream& os, address const& addr)
    {
        hpx::util::ios_flags_saver ifs(os);
        os << "(" << addr.locality_ << ":"
           << components::get_component_type_name(addr.type_) << ":"
           << std::showbase << std::hex << addr.address_ << ")";
        return os;
    }
}}    // namespace hpx::naming
