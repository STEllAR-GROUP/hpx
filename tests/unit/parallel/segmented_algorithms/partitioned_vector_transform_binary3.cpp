//  Copyright (c) 2017 Ajai V George
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <vector>

#include "test_transform_binary2.hpp"

///////////////////////////////////////////////////////////////////////////////
int main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    transform_binary2_tests<double, double, int>(localities);

    return hpx::util::report_errors();
}
