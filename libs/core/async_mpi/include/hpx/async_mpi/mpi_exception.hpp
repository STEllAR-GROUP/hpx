//  Copyright (c) 2020 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/errors/exception.hpp>

#include <string>

namespace hpx { namespace mpi { namespace experimental {

    // -------------------------------------------------------------------------
    // exception type for failed launch of MPI functions
    struct HPX_EXPORT mpi_exception : hpx::exception
    {
        mpi_exception(const std::string& msg, int err_code)
          : hpx::exception(hpx::bad_function_call, msg)
          , err_code_(err_code)
        {
        }

        int get_mpi_errorcode()
        {
            return err_code_;
        }

    protected:
        int err_code_;
    };

}}}    // namespace hpx::mpi::experimental
