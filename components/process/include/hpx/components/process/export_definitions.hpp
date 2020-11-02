//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/export_definitions.hpp>

#if defined(HPX_PROCESS_EXPORTS)
# define HPX_PROCESS_EXPORT HPX_SYMBOL_EXPORT
#else
# define HPX_PROCESS_EXPORT HPX_SYMBOL_IMPORT
#endif



