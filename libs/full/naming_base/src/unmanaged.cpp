//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/naming_base/unmanaged.hpp>

namespace hpx { namespace naming {

    id_type unmanaged(id_type const& id)
    {
        return id_type(detail::strip_internal_bits_from_gid(id.get_msb()),
            id.get_lsb(), id_type::unmanaged);
    }
}}    // namespace hpx::naming
