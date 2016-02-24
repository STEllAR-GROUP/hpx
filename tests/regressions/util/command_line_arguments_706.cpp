//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// verify #706 is fixed (`hpx::init` removes portions of non-option command
// line arguments before last `=` sign)

#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>

char const* argv[] =
{
    "command_line_argument_test",
    // We force one locality here
    "--hpx:localities=1",
    "nx=1",
    "ny=1=5"
};

int hpx_main(int argc, char** argv_init)
{
    HPX_TEST_EQ(argc, 3);
    HPX_TEST_EQ(0, std::strcmp(argv[0], argv_init[0]));
    for (int i = 1; i < argc; ++i)
    {
        HPX_TEST_EQ(0, std::strcmp(argv[i+1], argv_init[i]));
    }

    return hpx::finalize();
}

int main()
{
    HPX_TEST_EQ(hpx::init(4, const_cast<char**>(argv)), 0);
    return hpx::util::report_errors();
}
