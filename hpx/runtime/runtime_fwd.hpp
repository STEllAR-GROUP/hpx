//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/runtime_fwd.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/runtime_local/detail/runtime_local_fwd.hpp>

namespace hpx
{
    class HPX_EXPORT runtime_distributed;

    HPX_EXPORT runtime_distributed& get_runtime_distributed();
    HPX_EXPORT runtime_distributed*& get_runtime_distributed_ptr();
}
