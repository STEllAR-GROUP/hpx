//  Copyright (c) 2020 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/async_mpi/mpi_exception.hpp>

namespace hpx::mpi::experimental::detail {

    std::string error_message(int code)
    {
        int N = 1023;
        std::unique_ptr<char[]> err_buff(new char[std::size_t(N) + 1]);
        err_buff[0] = '\0';

        MPI_Error_string(code, err_buff.get(), &N);

        return err_buff.get();
    }
}
