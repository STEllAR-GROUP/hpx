//  Copyright (c) 2018 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/util/format.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <string>

int main(int argc, char* argv[])
{
    using hpx::util::format;
    {
        HPX_TEST_EQ((format("Hello")), "Hello");
        HPX_TEST_EQ((format("Hello, {}!", "world")), "Hello, world!");
        HPX_TEST_EQ((format("The number is {}", 1)), "The number is 1");
    }

    {
        HPX_TEST_EQ((format("{} {}", 1, 2)), "1 2");
        HPX_TEST_EQ((format("{} {1}", 1, 2)), "1 1");
        HPX_TEST_EQ((format("{2} {}", 1, 2)), "2 2");
        HPX_TEST_EQ((format("{2} {1}", 1, 2)), "2 1");

        HPX_TEST_EQ((format("{:}", 42)), "42");
        HPX_TEST_EQ((format("{:04}", 42)), "0042");
        HPX_TEST_EQ((format("{2:04}", 42, 43)), "0043");

        HPX_TEST_EQ((format("{:x}", 42)), "2a");
        HPX_TEST_EQ((format("{:04x}", 42)), "002a");
        HPX_TEST_EQ((format("{2:04x}", 42, 43)), "002b");

        HPX_TEST_EQ((format("{:#x}", 42)), "0x2a");
        HPX_TEST_EQ((format("{:#06x}", 42)), "0x002a");
        HPX_TEST_EQ((format("{2:#06x}", 42, 43)), "0x002b");
    }

    {
        HPX_TEST_EQ((format("{} {}", true, false)), "1 0");
    }

    return hpx::util::report_errors();
}
