//  Copyright (c) 2020 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/testing.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>

#include <cstdint>

#include "iter_sent.hpp"

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::parallel::v1::detail::distance(
                    Iterator<std::int64_t>{0}, Sentinel<int64_t>{100}),
        100);

    return hpx::util::report_errors();
}
