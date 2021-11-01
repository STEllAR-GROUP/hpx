//  Copyright (c) 2020 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/errors/exception.hpp>
#include <hpx/mpi_base/mpi.hpp>

#include <cstddef>
#include <memory>
#include <string>

namespace hpx::mpi::experimental {

    namespace detail {

        // extract MPI error message
        HPX_CORE_EXPORT std::string error_message(int code);

    }    // namespace detail

    // -------------------------------------------------------------------------
    // exception type for failed launch of MPI functions
    struct mpi_exception : hpx::exception
    {
        explicit mpi_exception(int err_code, const std::string& msg = "")
          : err_code_(err_code)
        {
            hpx::exception(hpx::bad_function_call,
                msg + std::string(" MPI returned with error: ") +
                    detail::error_message(err_code));
        }

        int get_mpi_errorcode()
        {
            return err_code_;
        }

    protected:
        int err_code_;
    };

}    // namespace hpx::mpi::experimental
