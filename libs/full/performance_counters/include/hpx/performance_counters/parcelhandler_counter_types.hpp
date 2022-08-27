//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/parcelset/parcelset_fwd.hpp>

namespace hpx::performance_counters {

    HPX_EXPORT void register_parcelhandler_counter_types(
        parcelset::parcelhandler& ph);
}    // namespace hpx::performance_counters

#endif
