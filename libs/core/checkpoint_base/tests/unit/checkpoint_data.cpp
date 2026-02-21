// Copyright (c) 2018 Adrian Serio
// Copyright (c) 2018-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>

#include <hpx/modules/checkpoint_base.hpp>
#include <hpx/modules/serialization.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <string>
#include <vector>

int main()
{
    char character = 'd';
    int integer = 10;
    float flt = 10.01f;
    bool boolean = true;
    std::string str = "I am a string of characters";
    std::vector<char> vec(str.begin(), str.end());

    std::vector<char> archive;
    hpx::util::save_checkpoint_data(
        archive, character, integer, flt, boolean, str, vec);

    std::size_t size = hpx::util::prepare_checkpoint_data(
        character, integer, flt, boolean, str, vec);
    HPX_TEST(archive.size() == size);

    char character2;
    int integer2;
    float flt2;
    bool boolean2;
    std::string str2;
    std::vector<char> vec2;

    hpx::util::restore_checkpoint_data(
        archive, character2, integer2, flt2, boolean2, str2, vec2);

    HPX_TEST_EQ(character, character2);
    HPX_TEST_EQ(integer, integer2);
    HPX_TEST_EQ(flt, flt2);
    HPX_TEST_EQ(boolean, boolean2);
    HPX_TEST_EQ(str, str2);
    HPX_TEST(vec == vec2);

    return hpx::util::report_errors();
}
