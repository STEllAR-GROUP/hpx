////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/util/any.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/unordered_map.hpp>
#include <hpx/util/storage/tuple.hpp>

#include <boost/any.hpp>

#include "small_big_object.hpp"

using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::util::any;
using hpx::util::any_nonser;
using hpx::util::any_cast;

using hpx::init;
using hpx::finalize;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        {
            any any1(big_object(30, 40));
            std::stringstream buffer;

            buffer << any1;

            HPX_TEST_EQ(buffer.str(), "3040");
        }

        {
            typedef uint64_t index_type;
            typedef hpx::util::any elem_type;
            typedef hpx::util::hash_any hash_elem_functor;

            typedef boost::unordered_multimap<elem_type, index_type,
                hash_elem_functor> field_index_map_type;
            typedef field_index_map_type::iterator field_index_map_iterator_type;

            field_index_map_type field_index_map_;
            field_index_map_iterator_type it;
            elem_type elem(std::string("first string"));
            index_type id = 1;

            std::pair<elem_type, index_type> pp=std::make_pair(elem,id);
            it = field_index_map_.insert(pp);
        }

        // test equality
        {
            any any1(7), any2(7), any3(10), any4(std::string("seven"));

            HPX_TEST_EQ(any1, 7);
            HPX_TEST_NEQ(any1, 10);
            HPX_TEST_NEQ(any1, 10.0f);
            HPX_TEST_EQ(any1, any1);
            HPX_TEST_EQ(any1, any2);
            HPX_TEST_NEQ(any1, any3);
            HPX_TEST_NEQ(any1, any4);

            std::string long_str =
                std::string("This is a looooooooooooooooooooooooooong string");
            std::string other_str = std::string("a different string");
            any1 = long_str;
            any2 = any1;
            any3 = other_str;
            any4 = 10.0f;

            HPX_TEST_EQ(any1, long_str);
            HPX_TEST_NEQ(any1, other_str);
            HPX_TEST_NEQ(any1, 10.0f);
            HPX_TEST_EQ(any1, any1);
            HPX_TEST_EQ(any1, any2);
            HPX_TEST_NEQ(any1, any3);
            HPX_TEST_NEQ(any1, any4);
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

            HPX_TEST_EQ((any_cast<small_object>(any1)) (7), uint64_t(17+7));
            HPX_TEST_EQ((any_cast<small_object>(any2)) (9), uint64_t(17+9));
            HPX_TEST_EQ((any_cast<small_object>(any3)) (11), uint64_t(17+11));
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

            HPX_TEST_EQ((any_cast<big_object>(any1)) (0, 1), uint64_t(5+12+0+1));
            HPX_TEST_EQ((any_cast<big_object>(any2)) (1, 0), uint64_t(5+12+1+0));
            HPX_TEST_EQ((any_cast<big_object>(any3)) (1, 1), uint64_t(5+12+1+1));
        }

        // move semantics
        {
            any any1(5);
            HPX_TEST(!any1.empty());
            any any2(std::move(any1));
            HPX_TEST(!any2.empty());
            HPX_TEST(any1.empty());
        }

        {
            any any1(5);
            HPX_TEST(!any1.empty());
            any any2;
            HPX_TEST(any2.empty());

            any2 = std::move(any1);
            HPX_TEST(!any2.empty());
            HPX_TEST(any1.empty());
        }
    }
    // non serializable version
    {
        // test equality
        {
            any_nonser any1_nonser(7), any2_nonser(7), any3_nonser(10),
                any4_nonser(std::string("seven"));

            HPX_TEST_EQ(any1_nonser, 7);
            HPX_TEST_NEQ(any1_nonser, 10);
            HPX_TEST_NEQ(any1_nonser, 10.0f);
            HPX_TEST_EQ(any1_nonser, any1_nonser);
            HPX_TEST_EQ(any1_nonser, any2_nonser);
            HPX_TEST_NEQ(any1_nonser, any3_nonser);
            HPX_TEST_NEQ(any1_nonser, any4_nonser);

            std::string long_str =
                std::string("This is a looooooooooooooooooooooooooong string");
            std::string other_str = std::string("a different string");
            any1_nonser = long_str;
            any2_nonser = any1_nonser;
            any3_nonser = other_str;
            any4_nonser = 10.0f;

            HPX_TEST_EQ(any1_nonser, long_str);
            HPX_TEST_NEQ(any1_nonser, other_str);
            HPX_TEST_NEQ(any1_nonser, 10.0f);
            HPX_TEST_EQ(any1_nonser, any1_nonser);
            HPX_TEST_EQ(any1_nonser, any2_nonser);
            HPX_TEST_NEQ(any1_nonser, any3_nonser);
            HPX_TEST_NEQ(any1_nonser, any4_nonser);
        }

        {
            if (sizeof(small_object) <= sizeof(void*))
                std::cout << "object is small\n";
            else
                std::cout << "object is large\n";

            small_object const f(17);

            any_nonser any1_nonser(f);
            any_nonser any2_nonser(any1_nonser);
            any_nonser any3_nonser = any1_nonser;

            HPX_TEST_EQ((any_cast<small_object>(any1_nonser)) (2), uint64_t(17+2));
            HPX_TEST_EQ((any_cast<small_object>(any2_nonser)) (4), uint64_t(17+4));
            HPX_TEST_EQ((any_cast<small_object>(any3_nonser)) (6), uint64_t(17+6));

        }

        {
            if (sizeof(big_object) <= sizeof(void*))
                std::cout << "object is small\n";
            else
                std::cout << "object is large\n";

            big_object const f(5, 12);

            any_nonser any1_nonser(f);
            any_nonser any2_nonser(any1_nonser);
            any_nonser any3_nonser = any1_nonser;

            HPX_TEST_EQ((any_cast<big_object>(any1_nonser)) (3,4), uint64_t(5+12+3+4));
            HPX_TEST_EQ((any_cast<big_object>(any2_nonser)) (5,6), uint64_t(5+12+5+6));
            HPX_TEST_EQ((any_cast<big_object>(any3_nonser)) (7,8), uint64_t(5+12+7+8));
        }

        // move semantics
        {
            any_nonser any1(5);
            HPX_TEST(!any1.empty());
            any_nonser any2(std::move(any1));
            HPX_TEST(!any2.empty());
            HPX_TEST(any1.empty());
        }

        {
            any_nonser any1(5);
            HPX_TEST(!any1.empty());
            any_nonser any2;
            HPX_TEST(any2.empty());

            any2 = std::move(any1);
            HPX_TEST(!any2.empty());
            HPX_TEST(any1.empty());
        }
    }

    finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    return init(cmdline, argc, argv);
}

