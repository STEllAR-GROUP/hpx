//  Copyright (c) 2017 Ajai V George
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_count.hpp>
#include <hpx/include/parallel_transform.hpp>
#include <hpx/include/partitioned_vector_predef.hpp>

#include <hpx/modules/testing.hpp>

#include <vector>

#include "test_transform_binary2.hpp"

///////////////////////////////////////////////////////////////////////////////
int main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    transform_binary2_tests<int, int, double>(localities);

    return hpx::util::report_errors();
}
