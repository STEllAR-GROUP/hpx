////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Shuangyang Yang
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/datastructures/any.hpp>
#include <hpx/modules/testing.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "small_big_object.hpp"

///////////////////////////////////////////////////////////////////////////////
int main()
{
    {
        hpx::util::streamable_any_nonser any1(big_object(30, 40));
        std::stringstream buffer;

        buffer << any1;

        HPX_TEST(buffer.str() == "3040");
    }

    // non serializable version
    {
        // test equality
        {
            hpx::any_nonser any1_nonser(7), any2_nonser(7), any3_nonser(10),
                any4_nonser(std::string("seven"));

            HPX_TEST_EQ(hpx::any_cast<int>(any1_nonser), 7);
            HPX_TEST_NEQ(hpx::any_cast<int>(any1_nonser), 10);
            HPX_TEST_NEQ(hpx::any_cast<int>(any1_nonser), 10.0f);
            HPX_TEST_EQ(hpx::any_cast<int>(any1_nonser),
                hpx::any_cast<int>(any1_nonser));
            HPX_TEST_EQ(hpx::any_cast<int>(any1_nonser),
                hpx::any_cast<int>(any2_nonser));
            HPX_TEST(any1_nonser.type() == any3_nonser.type());
            HPX_TEST(any1_nonser.type() != any4_nonser.type());

            std::string long_str =
                std::string("This is a looooooooooooooooooooooooooong string");
            std::string other_str = std::string("a different string");
            any1_nonser = long_str;
            any2_nonser = any1_nonser;
            any3_nonser = other_str;
            any4_nonser = 10.0f;

            HPX_TEST_EQ(hpx::any_cast<std::string>(any1_nonser), long_str);
            HPX_TEST_NEQ(hpx::any_cast<std::string>(any1_nonser), other_str);
            HPX_TEST(any1_nonser.type() == typeid(std::string));
            HPX_TEST(hpx::any_cast<std::string>(any1_nonser) ==
                hpx::any_cast<std::string>(any1_nonser));
            HPX_TEST(hpx::any_cast<std::string>(any1_nonser) ==
                hpx::any_cast<std::string>(any2_nonser));
            HPX_TEST(any1_nonser.type() == any3_nonser.type());
            HPX_TEST(any1_nonser.type() != any4_nonser.type());
        }

        {
            if (sizeof(small_object) <= sizeof(void*))
                std::cout << "object is small\n";
            else
                std::cout << "object is large\n";

            small_object const f(17);

            hpx::any_nonser any1_nonser(f);
            hpx::any_nonser any2_nonser(any1_nonser);
            hpx::any_nonser any3_nonser = any1_nonser;

            HPX_TEST_EQ((hpx::any_cast<small_object>(any1_nonser))(2),
                uint64_t(17 + 2));
            HPX_TEST_EQ((hpx::any_cast<small_object>(any2_nonser))(4),
                uint64_t(17 + 4));
            HPX_TEST_EQ((hpx::any_cast<small_object>(any3_nonser))(6),
                uint64_t(17 + 6));
        }

        {
            if (sizeof(big_object) <= sizeof(void*))
                std::cout << "object is small\n";
            else
                std::cout << "object is large\n";

            big_object const f(5, 12);

            hpx::any_nonser any1_nonser(f);
            hpx::any_nonser any2_nonser(any1_nonser);
            hpx::any_nonser any3_nonser = any1_nonser;

            HPX_TEST_EQ((hpx::any_cast<big_object>(any1_nonser))(3, 4),
                uint64_t(5 + 12 + 3 + 4));
            HPX_TEST_EQ((hpx::any_cast<big_object>(any2_nonser))(5, 6),
                uint64_t(5 + 12 + 5 + 6));
            HPX_TEST_EQ((hpx::any_cast<big_object>(any3_nonser))(7, 8),
                uint64_t(5 + 12 + 7 + 8));
        }

        // move semantics
        {
            hpx::any_nonser any1(5);
            HPX_TEST(any1.has_value());
            hpx::any_nonser any2(std::move(any1));
            HPX_TEST(any2.has_value());
            HPX_TEST(!any1.has_value());    // NOLINT
        }

        {
            hpx::any_nonser any1(5);
            HPX_TEST(any1.has_value());
            hpx::any_nonser any2;
            HPX_TEST(!any2.has_value());

            any2 = std::move(any1);
            HPX_TEST(any2.has_value());
            HPX_TEST(!any1.has_value());    // NOLINT
        }
    }

    return hpx::util::report_errors();
}
