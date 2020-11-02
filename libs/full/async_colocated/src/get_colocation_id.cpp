//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_colocated/get_colocation_id.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/agas/interface.hpp>

namespace hpx {

    naming::id_type get_colocation_id(
        launch::sync_policy, naming::id_type const& id, error_code& ec)
    {
        return agas::get_colocation_id(launch::sync, id, ec);
    }

    future<naming::id_type> get_colocation_id(naming::id_type const& id)
    {
        return agas::get_colocation_id(id);
    }
}    // namespace hpx
