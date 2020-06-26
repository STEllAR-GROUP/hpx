////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Shuangyang Yang
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/serialization/serializable_any.hpp>

#include <hpx/util/storage/tuple.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "small_big_object.hpp"

using hpx::util::any;
using hpx::util::any_cast;

using hpx::finalize;
using hpx::init;

struct compare_any
{
    bool operator()(hpx::util::any const& lhs, hpx::util::any const& rhs) const
    {
        return lhs.equal_to(rhs);
    }
};

////////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        {
            any any1(big_object(30, 40));
            std::stringstream buffer;

            buffer << any1;

            HPX_TEST_EQ(buffer.str(), "3040");
        }

        {
            using index_type = uint64_t;
            using elem_type = hpx::util::any;
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
            any any1(7), any2(7), any3(10), any4(std::string("seven"));

            HPX_TEST_EQ(any_cast<int>(any1), 7);
            HPX_TEST_NEQ(any_cast<int>(any1), 10);
            HPX_TEST_NEQ(any_cast<int>(any1), 10.0f);
            HPX_TEST_EQ(any_cast<int>(any1), any_cast<int>(any1));
            HPX_TEST_EQ(any_cast<int>(any1), any_cast<int>(any2));
            HPX_TEST(any1.type() == any3.type());
            HPX_TEST(any1.type() != any4.type());

            std::string long_str =
                std::string("This is a looooooooooooooooooooooooooong string");
            std::string other_str = std::string("a different string");
            any1 = long_str;
            any2 = any1;
            any3 = other_str;
            any4 = 10.0f;

            HPX_TEST_EQ(any_cast<std::string>(any1), long_str);
            HPX_TEST_NEQ(any_cast<std::string>(any1), other_str);
            HPX_TEST(any1.type() == typeid(std::string));
            HPX_TEST(
                any_cast<std::string>(any1) == any_cast<std::string>(any1));
            HPX_TEST(
                any_cast<std::string>(any1) == any_cast<std::string>(any2));
            HPX_TEST(any1.type() == any3.type());
            HPX_TEST(any1.type() != any4.type());
        }

        {
            if (sizeof(small_object) <= sizeof(void*))
                std::cout << "object is small\n";
            else
                std::cout << "object is large\n";

            small_object const f(17);

            any any1(f);
            any any2(any1);
            any any3;
            any3 = any1;

            HPX_TEST_EQ((any_cast<small_object>(any1))(7), uint64_t(17 + 7));
            HPX_TEST_EQ((any_cast<small_object>(any2))(9), uint64_t(17 + 9));
            HPX_TEST_EQ((any_cast<small_object>(any3))(11), uint64_t(17 + 11));
        }

        {
            if (sizeof(big_object) <= sizeof(void*))
                std::cout << "object is small\n";
            else
                std::cout << "object is large\n";

            big_object const f(5, 12);

            any any1(f);
            any any2(any1);
            any any3 = any1;

            HPX_TEST_EQ(
                (any_cast<big_object>(any1))(0, 1), uint64_t(5 + 12 + 0 + 1));
            HPX_TEST_EQ(
                (any_cast<big_object>(any2))(1, 0), uint64_t(5 + 12 + 1 + 0));
            HPX_TEST_EQ(
                (any_cast<big_object>(any3))(1, 1), uint64_t(5 + 12 + 1 + 1));
        }

        // move semantics
        {
            any any1(5);
            HPX_TEST(any1.has_value());
            any any2(std::move(any1));
            HPX_TEST(any2.has_value());
            HPX_TEST(!any1.has_value());    // NOLINT
        }

        {
            any any1(5);
            HPX_TEST(any1.has_value());
            any any2;
            HPX_TEST(!any2.has_value());

            any2 = std::move(any1);
            HPX_TEST(any2.has_value());
            HPX_TEST(!any1.has_value());    // NOLINT
        }
    }

    finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Initialize and run HPX
    return init(argc, argv);
}
