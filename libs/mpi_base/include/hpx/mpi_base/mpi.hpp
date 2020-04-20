//  Copyright (c) 2017 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PLUGINS_PARCELPORT_MPI_MPI_HPP
#define HPX_PLUGINS_PARCELPORT_MPI_MPI_HPP

#if (defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI)) ||      \
    defined(HPX_HAVE_LIB_MPI)

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-qual"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#endif

#include <mpi.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#endif

#endif
