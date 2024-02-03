//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if defined(HPX_HAVE_STDEXEC)
#include <stdexec/execution.hpp>

namespace hpx::execution::experimental {
    using namespace stdexec;
}
#endif