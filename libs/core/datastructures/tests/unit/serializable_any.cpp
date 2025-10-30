////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Shuangyang Yang
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/modules/datastructures.hpp>
#include <hpx/modules/testing.hpp>

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "small_big_object.hpp"

struct compare_any
{
    bool operator()(hpx::any const& lhs, hpx::any const& rhs) const
    {
        return lhs.equal_to(rhs);
    }
};

////////////////////////////////////////////////////////////////////////////////
int main()
{
    {
        {
            hpx::any any1(big_object(30, 40));
            std::stringstream buffer;

            buffer << any1;

            HPX_TEST_EQ(buffer.str(), "3040");
        }

        {
            using index_type = std::uint64_t;
            using elem_type = hpx::any;
            using hash_elem_functor = hpx::util::hash_any;

            using field_index_map_type = std::unordered_multimap<elem_type,
                index_type, hash_elem_functor, compare_any>;
            using field_index_map_iterator_type =
                field_index_map_type::iterator;

            field_index_map_type field_index_map_;
            field_index_map_iterator_type it;
            elem_type elem(std::string("first string"));
            index_type id = 1;

            std::pair<elem_type, index_type> pp = std::make_pair(elem, id);
            it = field_index_map_.insert(pp);
        }

        // test equality
        {
            hpx::any any1(7), any2(7), any3(10), any4(std::string("seven"));

            HPX_TEST_EQ(hpx::any_cast<int>(any1), 7);
            HPX_TEST_NEQ(hpx::any_cast<int>(any1), 10);
            HPX_TEST_NEQ(hpx::any_cast<int>(any1), 10.0f);
            HPX_TEST_EQ(hpx::any_cast<int>(any1), hpx::any_cast<int>(any1));
            HPX_TEST_EQ(hpx::any_cast<int>(any1), hpx::any_cast<int>(any2));
            HPX_TEST(any1.type() == any3.type());
            HPX_TEST(any1.type() != any4.type());

            std::string long_str =
                std::string("This is a looooooooooooooooooooooooooong string");
            std::string other_str = std::string("a different string");
            any1 = long_str;
            any2 = any1;
            any3 = other_str;
            any4 = 10.0f;

            HPX_TEST_EQ(hpx::any_cast<std::string>(any1), long_str);
            HPX_TEST_NEQ(hpx::any_cast<std::string>(any1), other_str);
            HPX_TEST(any1.type() == typeid(std::string));
            HPX_TEST(hpx::any_cast<std::string>(any1) ==
                hpx::any_cast<std::string>(any1));
            HPX_TEST(hpx::any_cast<std::string>(any1) ==
                hpx::any_cast<std::string>(any2));
            HPX_TEST(any1.type() == any3.type());
            HPX_TEST(any1.type() != any4.type());
        }

        {
            if (sizeof(small_object) <= sizeof(void*))
                std::cout << "object is small\n";
            else
                std::cout << "object is large\n";

            small_object const f(17);

            hpx::any any1(f);
            hpx::any any2(any1);
            hpx::any any3;
            any3 = any1;

            HPX_TEST_EQ(
                (hpx::any_cast<small_object>(any1))(7), std::uint64_t(17 + 7));
            HPX_TEST_EQ(
                (hpx::any_cast<small_object>(any2))(9), std::uint64_t(17 + 9));
            HPX_TEST_EQ((hpx::any_cast<small_object>(any3))(11),
                std::uint64_t(17 + 11));
        }

        {
            if (sizeof(big_object) <= sizeof(void*))
                std::cout << "object is small\n";
            else
                std::cout << "object is large\n";

            big_object const f(5, 12);

            hpx::any any1(f);
            hpx::any any2(any1);
            hpx::any any3 = any1;

            HPX_TEST_EQ((hpx::any_cast<big_object>(any1))(0, 1),
                std::uint64_t(5 + 12 + 0 + 1));
            HPX_TEST_EQ((hpx::any_cast<big_object>(any2))(1, 0),
                std::uint64_t(5 + 12 + 1 + 0));
            HPX_TEST_EQ((hpx::any_cast<big_object>(any3))(1, 1),
                std::uint64_t(5 + 12 + 1 + 1));
        }

        // move semantics
        {
            hpx::any any1(5);
            HPX_TEST(any1.has_value());
            hpx::any any2(std::move(any1));
            HPX_TEST(any2.has_value());
            HPX_TEST(!any1.has_value());    // NOLINT
        }

        {
            hpx::any any1(5);
            HPX_TEST(any1.has_value());
            hpx::any any2;
            HPX_TEST(!any2.has_value());

            any2 = std::move(any1);
            HPX_TEST(any2.has_value());
            HPX_TEST(!any1.has_value());    // NOLINT
        }
    }

    return hpx::util::report_errors();
}
