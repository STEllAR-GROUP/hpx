//  Copyright (c) 2019-2020 The STE||AR GROUP
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/async_mpi/force_linking.hpp>
#include <hpx/async_mpi/mpi_future.hpp>

namespace hpx { namespace mpi {

    // reference all symbols that have to be explicitly linked with the core
    // library
    force_linking_helper& force_linking()
    {
        static force_linking_helper helper{
            &experimental::detail::hpx_mpi_errhandler};
        return helper;
    }
}}    // namespace hpx::mpi
