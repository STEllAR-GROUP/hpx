//  Copyright (c) 2023 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

int hpx_main()
{
    return hpx::local::finalize();
}

bool called_on_finalize = false;

void finalize_callback()
{
    called_on_finalize = true;
}

struct set_finalizer
{
    set_finalizer()
    {
        hpx::on_finalize = &finalize_callback;
    }
};

set_finalizer init;

int main(int argc, char** argv)
{
    HPX_TEST_EQ(hpx::local::init(hpx_main, argc, argv), 0);

    // on_finalize should have been called at this point
    HPX_TEST(called_on_finalize);

    return hpx::util::report_errors();
}
