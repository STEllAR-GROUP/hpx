//  Copyright (C) 2019-2022 T. Zachary Laine
//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/datastructures/detail/flat_set.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// dummy implementations for testing macros
namespace hpx::detail {

    template <typename Key, typename Compare, typename KeyContainer>
    std::ostream& operator<<(std::ostream& os,
        hpx::detail::flat_set<Key, Compare, KeyContainer> const&)
    {
        return os;
    }

    template <typename KeyRef, typename KeyIter>
    std::ostream& operator<<(std::ostream& os,
        hpx::detail::flat_set_iterator<KeyRef, KeyIter> const&)
    {
        return os;
    }
}    // namespace hpx::detail

namespace std {

    template <typename T>
    std::ostream& operator<<(std::ostream& os, std::vector<T> const&)
    {
        return os;
    }
}    // namespace std

// Test instantiations.
template class hpx::detail::flat_set<std::string>;

void test_std_flat_set_iterator()
{
    using fmap_t = hpx::detail::flat_set<std::string>;

    {
        fmap_t map;
        fmap_t::iterator mutable_first = map.begin();
        fmap_t::const_iterator const_first = mutable_first;

        HPX_TEST_EQ(mutable_first, const_first);
    }

    {
        fmap_t::key_container_type c = {"key0", "key1", "key2"};

        fmap_t::iterator const first(c.begin());
        fmap_t::iterator const last(c.end());

        HPX_TEST_EQ(first + 3, last);
        HPX_TEST_EQ(first, last - 3);

        HPX_TEST_EQ(first[1], "key1");
        HPX_TEST_EQ(last[-3], "key0");

        HPX_TEST_EQ((*(first + 1)), "key1");
        HPX_TEST_EQ((*(last - 3)), "key0");

        HPX_TEST_EQ(first - last, -3);
        HPX_TEST_EQ(last - first, 3);

        {
            auto first_copy = first;
            auto last_copy = last;

            first_copy += 3;
            last_copy -= 3;

            HPX_TEST_EQ(first_copy, last);
            HPX_TEST_EQ(last_copy, first);
        }

        {
            auto first_copy = first;
            auto last_copy = last;

            HPX_TEST_EQ(first_copy++, first);
            HPX_TEST_EQ(first_copy++, first + 1);
            HPX_TEST_EQ(first_copy++, first + 2);
            HPX_TEST_EQ(first_copy, last);

            HPX_TEST_EQ(last_copy--, last);
            HPX_TEST_EQ(last_copy--, last - 1);
            HPX_TEST_EQ(last_copy--, last - 2);
            HPX_TEST_EQ(last_copy, first);
        }

        {
            auto first_copy = first;
            auto last_copy = last;

            HPX_TEST_EQ(++first_copy, first + 1);
            HPX_TEST_EQ(++first_copy, first + 2);
            HPX_TEST_EQ(++first_copy, last);
            HPX_TEST_EQ(first_copy, last);

            HPX_TEST_EQ(--last_copy, last - 1);
            HPX_TEST_EQ(--last_copy, last - 2);
            HPX_TEST_EQ(--last_copy, first);
            HPX_TEST_EQ(last_copy, first);
        }

        HPX_TEST_EQ(first, first);
        HPX_TEST_NEQ(first, last);

        HPX_TEST_LT(first, last);
        HPX_TEST_LTE(first, last);
        HPX_TEST_LTE(first, first);
    }

    {
        fmap_t::key_container_type c = {"key0", "key1", "key2"};

        fmap_t::reverse_iterator const first(c.rbegin());
        fmap_t::reverse_iterator const last(c.rend());

        HPX_TEST_EQ(first + 3, last);
        HPX_TEST_EQ(first, last - 3);

        HPX_TEST_EQ(first[1], "key1");
        HPX_TEST_EQ(last[-3], "key2");

        HPX_TEST_EQ((*(first + 1)), "key1");
        HPX_TEST_EQ((*(last - 3)), "key2");

        HPX_TEST_EQ(first - last, -3);
        HPX_TEST_EQ(last - first, 3);

        {
            auto first_copy = first;
            auto last_copy = last;

            first_copy += 3;
            last_copy -= 3;

            HPX_TEST_EQ(first_copy, last);
            HPX_TEST_EQ(last_copy, first);
        }

        {
            auto first_copy = first;
            auto last_copy = last;

            HPX_TEST_EQ(first_copy++, first);
            HPX_TEST_EQ(first_copy++, first + 1);
            HPX_TEST_EQ(first_copy++, first + 2);
            HPX_TEST_EQ(first_copy, last);

            HPX_TEST_EQ(last_copy--, last);
            HPX_TEST_EQ(last_copy--, last - 1);
            HPX_TEST_EQ(last_copy--, last - 2);
            HPX_TEST_EQ(last_copy, first);
        }

        {
            auto first_copy = first;
            auto last_copy = last;

            HPX_TEST_EQ(++first_copy, first + 1);
            HPX_TEST_EQ(++first_copy, first + 2);
            HPX_TEST_EQ(++first_copy, last);
            HPX_TEST_EQ(first_copy, last);

            HPX_TEST_EQ(--last_copy, last - 1);
            HPX_TEST_EQ(--last_copy, last - 2);
            HPX_TEST_EQ(--last_copy, first);
            HPX_TEST_EQ(last_copy, first);
        }

        HPX_TEST_EQ(first, first);
        HPX_TEST_NEQ(first, last);

        HPX_TEST_LT(first, last);
        HPX_TEST_LTE(first, last);
        HPX_TEST_LTE(first, first);
    }
}

void std_flat_set_ctors_iterators()
{
    using fmap_t = hpx::detail::flat_set<std::string>;
    using value_t = std::string;

    auto pair_cmp = [](auto lhs, auto rhs) { return lhs == rhs; };

    std::initializer_list<value_t> init_list = {"key1", "key2", "key0"};
    fmap_t::key_container_type const c = {"key1", "key2", "key0"};
    std::vector<value_t> const vec = {"key1", "key2", "key0"};
    std::vector<value_t> sorted_vec = vec;
    std::sort(sorted_vec.begin(), sorted_vec.end());

    fmap_t const map_from_c(c);
    fmap_t const map_from_vec(vec);

    HPX_TEST_EQ(map_from_c, map_from_vec);
    HPX_TEST(!(map_from_c != map_from_vec));

    fmap_t const map_from_const_map = map_from_c;
    fmap_t const map_from_map_rvalue = fmap_t(map_from_c);

    HPX_TEST_EQ(map_from_const_map, map_from_vec);
    HPX_TEST(!(map_from_map_rvalue != map_from_vec));

    {
        fmap_t map;
        HPX_TEST(map.empty());
        HPX_TEST_EQ(map.size(), 0u);
        HPX_TEST_EQ(map.begin(), map.end());
    }

    {
        fmap_t map(c);
        HPX_TEST_EQ(map.size(), 3u);
        HPX_TEST(!(map.empty()));
        HPX_TEST(std::equal(map.begin(), map.end(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }

    {
        fmap_t map(vec.begin(), vec.end());
        HPX_TEST(std::equal(map.begin(), map.end(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));

        HPX_TEST(std::equal(map.cbegin(), map.cend(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }

    {
        fmap_t map(vec);
        HPX_TEST(std::equal(map.begin(), map.end(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));

        HPX_TEST(std::equal(map.cbegin(), map.cend(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }

    {
        hpx::detail::flat_set<std::string, std::greater<>> map(vec);
        HPX_TEST(std::equal(map.rbegin(), map.rend(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));

        HPX_TEST(std::equal(map.crbegin(), map.crend(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }

    {
        // This breaks the ctor's contract, since c are not in sorted
        // order, but it's useful to verify that the resulting contents were
        // not touched.
        fmap_t map(hpx::detail::sorted_unique, c);
        HPX_TEST_EQ(map.size(), 3u);
        HPX_TEST(!(map.empty()));
        HPX_TEST(std::equal(
            map.begin(), map.end(), vec.begin(), vec.end(), pair_cmp));
    }

    {
        // Broken contract, as above.
        fmap_t map(hpx::detail::sorted_unique, vec.begin(), vec.end());
        HPX_TEST_EQ(map.size(), 3u);
        HPX_TEST(!(map.empty()));
        HPX_TEST(std::equal(
            map.begin(), map.end(), vec.begin(), vec.end(), pair_cmp));
    }

    {
        // Broken contract, as above.
        fmap_t map(hpx::detail::sorted_unique, vec);
        HPX_TEST_EQ(map.size(), 3u);
        HPX_TEST(!(map.empty()));
        HPX_TEST(std::equal(
            map.begin(), map.end(), vec.begin(), vec.end(), pair_cmp));
    }

    {
        hpx::detail::flat_set<std::string, std::greater<>> map;
        map.insert(vec.begin(), vec.end());
        HPX_TEST(std::equal(map.rbegin(), map.rend(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }

    {
        fmap_t map(init_list);
        HPX_TEST(std::equal(map.begin(), map.end(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }

    {
        // Broken contract, as above.
        fmap_t map(hpx::detail::sorted_unique, init_list);
        HPX_TEST(std::equal(
            map.begin(), map.end(), vec.begin(), vec.end(), pair_cmp));
    }

    {
        fmap_t map;
        map = {value_t{"key1"}, value_t{"key2"}, value_t{"key0"}};
        HPX_TEST(std::equal(map.begin(), map.end(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }
}

void std_flat_set_ctors_allocators_iterators()
{
    using fmap_t = hpx::detail::flat_set<std::string>;
    using value_t = std::string;

    auto pair_cmp = [](auto lhs, auto rhs) { return lhs == rhs; };

    std::initializer_list<value_t> init_list = {"key1", "key2", "key0"};
    fmap_t::key_container_type const c = {"key1", "key2", "key0"};
    std::vector<value_t> const vec = {"key1", "key2", "key0"};
    std::vector<value_t> sorted_vec = vec;
    std::sort(sorted_vec.begin(), sorted_vec.end());

    auto allocator = c.get_allocator();

    fmap_t const map_from_c(c, allocator);
    fmap_t const map_from_vec(vec, allocator);

    HPX_TEST_EQ(map_from_c, map_from_vec);
    HPX_TEST(!(map_from_c != map_from_vec));

    fmap_t const map_from_const_map(map_from_c, allocator);
    fmap_t const map_from_map_rvalue(fmap_t(map_from_c), allocator);

    HPX_TEST_EQ(map_from_const_map, map_from_vec);
    HPX_TEST(!(map_from_map_rvalue != map_from_vec));

    {
        fmap_t map(allocator);
        HPX_TEST(map.empty());
        HPX_TEST_EQ(map.size(), 0u);
        HPX_TEST_EQ(map.begin(), map.end());
    }

    {
        fmap_t map(c, allocator);
        HPX_TEST_EQ(map.size(), 3u);
        HPX_TEST(!(map.empty()));
        HPX_TEST(std::equal(map.begin(), map.end(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }

    {
        fmap_t map(vec.begin(), vec.end(), allocator);
        HPX_TEST(std::equal(map.begin(), map.end(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));

        HPX_TEST(std::equal(map.cbegin(), map.cend(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }

    {
        fmap_t map(vec, allocator);
        HPX_TEST(std::equal(map.begin(), map.end(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));

        HPX_TEST(std::equal(map.cbegin(), map.cend(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }

    {
        hpx::detail::flat_set<std::string, std::greater<>> map(vec, allocator);
        HPX_TEST(std::equal(map.rbegin(), map.rend(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));

        HPX_TEST(std::equal(map.crbegin(), map.crend(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }

    {
        // This breaks the ctor's contract, since c are not in sorted
        // order, but it's useful to verify that the resulting contents were
        // not touched.
        fmap_t map(hpx::detail::sorted_unique, c, allocator);
        HPX_TEST_EQ(map.size(), 3u);
        HPX_TEST(!(map.empty()));
        HPX_TEST(std::equal(
            map.begin(), map.end(), vec.begin(), vec.end(), pair_cmp));
    }

    {
        // Broken contract, as above.
        fmap_t map(
            hpx::detail::sorted_unique, vec.begin(), vec.end(), allocator);
        HPX_TEST_EQ(map.size(), 3u);
        HPX_TEST(!(map.empty()));
        HPX_TEST(std::equal(
            map.begin(), map.end(), vec.begin(), vec.end(), pair_cmp));
    }

    {
        // Broken contract, as above.
        fmap_t map(hpx::detail::sorted_unique, vec, allocator);
        HPX_TEST_EQ(map.size(), 3u);
        HPX_TEST(!(map.empty()));
        HPX_TEST(std::equal(
            map.begin(), map.end(), vec.begin(), vec.end(), pair_cmp));
    }

    {
        hpx::detail::flat_set<std::string, std::greater<>> map(allocator);
        map.insert(vec.begin(), vec.end());
        HPX_TEST(std::equal(map.rbegin(), map.rend(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }

    {
        fmap_t map(init_list, allocator);
        HPX_TEST(std::equal(map.begin(), map.end(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }

    {
        // Broken contract, as above.
        fmap_t map(hpx::detail::sorted_unique, init_list, allocator);
        HPX_TEST(std::equal(
            map.begin(), map.end(), vec.begin(), vec.end(), pair_cmp));
    }

    {
        fmap_t map(allocator);
        map = {value_t{"key1"}, value_t{"key2"}, value_t{"key0"}};
        HPX_TEST(std::equal(map.begin(), map.end(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }
}

void std_flat_set_emplace_insert()
{
    using fmap_t = hpx::detail::flat_set<std::string>;
    using rev_fmap_t = hpx::detail::flat_set<std::string, std::greater<>>;
    using value_t = std::string;

    auto pair_cmp = [](auto lhs, auto rhs) { return lhs == rhs; };

    fmap_t::key_container_type const c = {"key1", "key2", "key0"};
    std::vector<value_t> const vec = {"key1", "key2", "key0"};
    std::vector<value_t> sorted_vec = vec;
    std::sort(sorted_vec.begin(), sorted_vec.end());

    value_t const val1("q");
    value_t const val2("w");

    using foreign_value_t = char const*;

    {
        foreign_value_t foreign_pair("e");

        // NOTE: These calls exercise all emplace's and inserts before the range
        // inserts.

        fmap_t map;
        map.insert(val1);
        map.insert(value_t("r"));
        map.insert(map.begin(), val2);
        map.insert(map.begin(), value_t("t"));
        map.insert(foreign_pair);
        map.insert(static_cast<foreign_value_t>("y"));

        std::vector<value_t> const qwerty_vec = {"e", "q", "r", "t", "w", "y"};
        HPX_TEST(std::equal(map.begin(), map.end(), qwerty_vec.begin(),
            qwerty_vec.end(), pair_cmp));
    }

    // NOTE: Already exercised by ctor test:
    // template<class InputIterator>
    // void insert(InputIterator first, InputIterator last);
    // template<class InputIterator>
    // void insert(sorted_unique_t, InputIterator first, InputIterator last);

    {
        fmap_t map;
        map.insert({value_t{"key1"}, value_t{"key2"}, value_t{"key0"}});
        HPX_TEST_EQ(map.size(), 3u);
        HPX_TEST(!(map.empty()));
        HPX_TEST(std::equal(map.begin(), map.end(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }

    {
        rev_fmap_t map;
        map.insert({value_t{"key1"}, value_t{"key2"}, value_t{"key0"}});
        HPX_TEST_EQ(map.size(), 3u);
        HPX_TEST(!(map.empty()));
        HPX_TEST(std::equal(map.rbegin(), map.rend(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }

    {
        // This breaks the ctor's contract, since c are not in sorted
        // order, but it's useful to verify that the resulting contents were
        // not touched.
        fmap_t map;
        map.insert(hpx::detail::sorted_unique,
            {value_t{"key1"}, value_t{"key2"}, value_t{"key0"}});
        HPX_TEST_EQ(map.size(), 3u);
        HPX_TEST(!(map.empty()));
        HPX_TEST(std::equal(
            map.begin(), map.end(), vec.begin(), vec.end(), pair_cmp));
    }
}

void std_flat_set_extract_replace()
{
    using fmap_t = hpx::detail::flat_set<std::string>;
    using value_t = std::string;

    fmap_t::key_container_type const sorted_c = {"key0", "key1", "key2"};
    std::vector<value_t> const vec = {"key1", "key2", "key0"};

    fmap_t map1(vec);
    fmap_t const map1_copy = map1;

    fmap_t::key_container_type extracted_c = std::move(map1).extract();
    HPX_TEST_EQ(extracted_c, sorted_c);

    fmap_t map2;
    map2.replace(std::move(extracted_c));
    HPX_TEST_EQ(map2, map1_copy);
}

void std_flat_set_erase_swap()
{
    using fmap_t = hpx::detail::flat_set<std::string>;
    using value_t = std::string;

    auto pair_cmp = [](auto lhs, auto rhs) { return lhs == rhs; };

    std::vector<value_t> const vec = {"key0", "key1", "key2"};

    {
        fmap_t map(vec);
        map.erase(map.begin());
        std::vector<value_t> const local_vec = {"key1", "key2"};
        HPX_TEST(std::equal(map.begin(), map.end(), local_vec.begin(),
            local_vec.end(), pair_cmp));
    }

    {
        fmap_t map(vec);
        map.erase(map.cbegin());
        std::vector<value_t> const local_vec = {"key1", "key2"};
        HPX_TEST(std::equal(map.begin(), map.end(), local_vec.begin(),
            local_vec.end(), pair_cmp));
    }

    {
        fmap_t map(vec);
        map.erase("key0");
        std::vector<value_t> const local_vec = {"key1", "key2"};
        HPX_TEST(std::equal(map.begin(), map.end(), local_vec.begin(),
            local_vec.end(), pair_cmp));
    }

    {
        fmap_t map(vec);
        map.erase(map.begin(), map.end());
        HPX_TEST_EQ(map.begin(), map.end());
    }

    {
        fmap_t const orig_map(vec);
        fmap_t const orig_empty_map;

        auto map1 = orig_map;
        auto map2 = orig_empty_map;

        std::swap(map1, map2);

        HPX_TEST_EQ(map1, orig_empty_map);
        HPX_TEST_EQ(map2, orig_map);

        std::swap(map1, map2);

        HPX_TEST_EQ(map1, orig_map);
        HPX_TEST_EQ(map2, orig_empty_map);
    }
}

void std_flat_set_count_contains()
{
    using fmap_t = hpx::detail::flat_set<std::string>;
    using value_t = std::string;

    std::vector<value_t> const vec = {"key1", "key2", "key0"};

    fmap_t const map(vec);
    HPX_TEST_EQ(map.count("key0"), 1u);
    HPX_TEST_EQ(map.count("key10"), 0u);
    HPX_TEST(map.count("key0"));
    HPX_TEST(!(map.count("key10")));
}

void std_flat_set_equal_range()
{
    using fmap_t = hpx::detail::flat_set<std::string>;
    using value_t = std::string;

    std::vector<value_t> const vec = {"key1", "key2", "key0"};

    fmap_t const map(vec);
    auto const eq_range = map.equal_range("key0");
    HPX_TEST_EQ(eq_range.first, map.begin());
    HPX_TEST_EQ(eq_range.second, map.begin() + 1);
    auto const empty_eq_range = map.equal_range("");
    HPX_TEST_EQ(empty_eq_range.first, empty_eq_range.second);
}

void std_flat_set_comparisons()
{
    using fmap_t = hpx::detail::flat_set<std::string>;
    using value_t = std::string;

    std::vector<value_t> const vec = {"key0", "key1", "key2"};

    fmap_t const map_123(vec);
    fmap_t const map_12(vec.begin(), vec.begin() + 2);

    HPX_TEST_EQ(map_123, map_123);
    HPX_TEST_NEQ(map_123, map_12);

    HPX_TEST_LT(map_12, map_123);
    HPX_TEST_LTE(map_12, map_123);
    HPX_TEST_LTE(map_123, map_123);
}

int main()
{
    test_std_flat_set_iterator();
    std_flat_set_ctors_iterators();
    std_flat_set_ctors_allocators_iterators();
    std_flat_set_emplace_insert();
    std_flat_set_extract_replace();
    std_flat_set_erase_swap();
    std_flat_set_count_contains();
    std_flat_set_equal_range();
    std_flat_set_comparisons();

    return hpx::util::report_errors();
}
