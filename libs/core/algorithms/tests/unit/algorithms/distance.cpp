//  Copyright (c) 2020 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/iterator_support/tests/iter_sent.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>

#include <cstdint>

int main()
{
    HPX_TEST_EQ(hpx::parallel::detail::distance(
                    iterator<std::int64_t>{0}, sentinel<int64_t>{100}),
        100);

    return hpx::util::report_errors();
}
