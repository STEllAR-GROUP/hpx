//  Copyright (c) 2019-2020 The STE||AR GROUP
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_mpi/mpi_future.hpp>

#include <mpi.h>

namespace hpx { namespace async_mpi {

    struct force_linking_helper
    {
        MPI_Errhandler* mpi_errhandler;
    };

    force_linking_helper& force_linking();
}}    // namespace hpx::async_mpi
