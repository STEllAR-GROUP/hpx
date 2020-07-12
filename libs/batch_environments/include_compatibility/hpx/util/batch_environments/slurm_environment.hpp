//  Copyright (c) 2019 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/batch_environments/config/defines.hpp>
#include <hpx/modules/batch_environments.hpp>

#if HPX_BATCH_ENVIRONMENTS_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/util/batch_environments/slurm_environment.hpp is deprecated, \
    please include hpx/modules/batch_environments.hpp instead")
#else
#warning                                                                       \
    "The header hpx/util/batch_environments/slurm_environment.hpp is deprecated, \
    please include hpx/modules/batch_environments.hpp instead"
#endif
#endif
