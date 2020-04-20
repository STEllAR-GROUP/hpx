//  Copyright (c) 2019 Ste||ar Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/mpi_base/config/defines.hpp>
#include <hpx/mpi_base/mpi.hpp>

#if defined(HPX_MPI_BASE_HAVE_DEPRECATION_WARNINGS)
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/plugins/parcelport/mpi/mpi.hpp is deprecated, \
    please include hpx/mpi_base/mpi.hpp instead")
#else
#warning "The header hpx/plugins/parcelport/mpi/mpi.hpp is deprecated, \
    please include hpx/mpi_base/mpi.hpp instead"
#endif
#endif
