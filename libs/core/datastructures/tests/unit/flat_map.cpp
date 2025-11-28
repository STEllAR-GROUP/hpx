//  Copyright (C) 2019-2022 T. Zachary Laine
//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/datastructures.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// dummy implementations for testing macros
namespace hpx::detail {

    template <typename Key, typename T, typename Compare, typename KeyContainer,
        typename MappedContainer>
    std::ostream& operator<<(std::ostream& os,
        hpx::detail::flat_map<Key, T, Compare, KeyContainer,
            MappedContainer> const&)
    {
        return os;
    }

    template <typename KeyRef, typename TRef, typename KeyIter,
        typename MappedIter>
    std::ostream& operator<<(std::ostream& os,
        hpx::detail::flat_map_iterator<KeyRef, TRef, KeyIter,
            MappedIter> const&)
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
template class hpx::detail::flat_map<std::string, int>;

void test_std_flat_map_iterator()
{
    using fmap_t = hpx::detail::flat_map<std::string, int>;

    {
        fmap_t map;
        fmap_t::iterator mutable_first = map.begin();
        fmap_t::const_iterator const_first = mutable_first;

        HPX_TEST_EQ(mutable_first, const_first);
    }

    {
        fmap_t::containers c = {{"key0", "key1", "key2"}, {0, 1, 2}};

        fmap_t::iterator const first(c.keys.begin(), c.values.begin());
        fmap_t::iterator const last(c.keys.end(), c.values.end());

        HPX_TEST_EQ(first + 3, last);
        HPX_TEST_EQ(first, last - 3);

        HPX_TEST_EQ(first[1].first, "key1");
        HPX_TEST_EQ(last[-3].second, 0);

        HPX_TEST_EQ((*(first + 1)).first, "key1");
        HPX_TEST_EQ((*(last - 3)).second, 0);

        HPX_TEST_EQ((first + 1)->first, "key1");
        HPX_TEST_EQ((last - 3)->second, 0);

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
        fmap_t::containers c = {{"key0", "key1", "key2"}, {0, 1, 2}};

        fmap_t::reverse_iterator const first(
            c.keys.rbegin(), c.values.rbegin());
        fmap_t::reverse_iterator const last(c.keys.rend(), c.values.rend());

        HPX_TEST_EQ(first + 3, last);
        HPX_TEST_EQ(first, last - 3);

        HPX_TEST_EQ(first[1].first, "key1");
        HPX_TEST_EQ(last[-3].second, 2);

        HPX_TEST_EQ((*(first + 1)).first, "key1");
        HPX_TEST_EQ((*(last - 3)).second, 2);

        HPX_TEST_EQ((first + 1)->first, "key1");
        HPX_TEST_EQ((last - 3)->second, 2);

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

void std_flat_map_ctors_iterators()
{
    using fmap_t = hpx::detail::flat_map<std::string, int>;
    using pair_t = std::pair<std::string, int>;

    auto pair_cmp = [](auto lhs, auto rhs) {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    };

    std::initializer_list<pair_t> init_list = {
        {"key1", 1}, {"key2", 2}, {"key0", 0}};
    fmap_t::containers const c = {{"key1", "key2", "key0"}, {1, 2, 0}};
    std::vector<pair_t> const vec = {{"key1", 1}, {"key2", 2}, {"key0", 0}};
    std::vector<pair_t> sorted_vec = vec;
    std::sort(sorted_vec.begin(), sorted_vec.end());

    fmap_t const map_from_c(c.keys, c.values);
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
        fmap_t map(c.keys, c.values);
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
        hpx::detail::flat_map<std::string, int, std::greater<std::string>> map(
            vec);
        HPX_TEST(std::equal(map.rbegin(), map.rend(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));

        HPX_TEST(std::equal(map.crbegin(), map.crend(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }

    {
        // This breaks the ctor's contract, since c.keys are not in sorted
        // order, but it's useful to verify that the resulting contents were
        // not touched.
        fmap_t map(hpx::detail::sorted_unique, c.keys, c.values);
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
        hpx::detail::flat_map<std::string, int, std::greater<std::string>> map;
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
        map = {pair_t{"key1", 1}, pair_t{"key2", 2}, pair_t{"key0", 0}};
        HPX_TEST(std::equal(map.begin(), map.end(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }
}

void std_flat_map_ctors_allocators_iterators()
{
    using fmap_t = hpx::detail::flat_map<std::string, int>;
    using pair_t = std::pair<std::string, int>;

    auto pair_cmp = [](auto lhs, auto rhs) {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    };

    std::initializer_list<pair_t> init_list = {
        {"key1", 1}, {"key2", 2}, {"key0", 0}};
    fmap_t::containers const c = {{"key1", "key2", "key0"}, {1, 2, 0}};
    std::vector<pair_t> const vec = {{"key1", 1}, {"key2", 2}, {"key0", 0}};
    std::vector<pair_t> sorted_vec = vec;
    std::sort(sorted_vec.begin(), sorted_vec.end());

    auto allocator = c.keys.get_allocator();

    fmap_t const map_from_c(c.keys, c.values, allocator);
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
        fmap_t map(c.keys, c.values, allocator);
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
        hpx::detail::flat_map<std::string, int, std::greater<>> map(
            vec, allocator);
        HPX_TEST(std::equal(map.rbegin(), map.rend(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));

        HPX_TEST(std::equal(map.crbegin(), map.crend(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }

    {
        // This breaks the ctor's contract, since c.keys are not in sorted
        // order, but it's useful to verify that the resulting contents were
        // not touched.
        fmap_t map(hpx::detail::sorted_unique, c.keys, c.values, allocator);
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
        hpx::detail::flat_map<std::string, int, std::greater<std::string>> map(
            allocator);
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
        map = {pair_t{"key1", 1}, pair_t{"key2", 2}, pair_t{"key0", 0}};
        HPX_TEST(std::equal(map.begin(), map.end(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }
}

void std_flat_map_index_at()
{
    using fmap_t = hpx::detail::flat_map<std::string, int>;
    using pair_t = std::pair<std::string, int>;

    std::vector<pair_t> const vec = {{"key1", 1}, {"key2", 2}, {"key-1", -1}};
    std::vector<pair_t> sorted_vec = vec;
    std::sort(sorted_vec.begin(), sorted_vec.end());

    {
        fmap_t map(vec);
        std::string const key_1("key1");
        HPX_TEST_EQ(map["key-1"], -1);
        HPX_TEST_EQ(map["key2"], 2);
        HPX_TEST_EQ(map[key_1], 1);
        HPX_TEST_EQ(map["foo"], 0);

        map["foo"] = 8;

        HPX_TEST_EQ(map.at("key-1"), -1);
        HPX_TEST_EQ(map.at("key2"), 2);
        HPX_TEST_EQ(map.at("key1"), 1);
        HPX_TEST_EQ(map.at("foo"), 8);
        HPX_TEST_THROW(map.at("bar"), std::out_of_range);

        fmap_t const& cmap = map;
        HPX_TEST_THROW(cmap.at("bar"), std::out_of_range);
    }
}

void std_flat_map_emplace_insert()
{
    using fmap_t = hpx::detail::flat_map<std::string, int>;
    using rev_fmap_t =
        hpx::detail::flat_map<std::string, int, std::greater<std::string>>;
    using pair_t = std::pair<std::string, int>;

    auto pair_cmp = [](auto lhs, auto rhs) {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    };

    fmap_t::containers const c = {{"key1", "key2", "key0"}, {1, 2, 0}};
    std::vector<pair_t> const vec = {{"key1", 1}, {"key2", 2}, {"key0", 0}};
    std::vector<pair_t> sorted_vec = vec;
    std::sort(sorted_vec.begin(), sorted_vec.end());

    pair_t const pair1("q", 3);
    pair_t const pair2("w", 4);

    using foreign_pair_t = std::pair<char const*, short>;

    {
        foreign_pair_t foreign_pair("e", 5);

        // NOTE: These calls exercise all emplace's and inserts before the range
        // inserts.

        fmap_t map;
        map.insert(pair1);
        map.insert(pair_t("r", 6));
        map.insert(map.begin(), pair2);
        map.insert(map.begin(), pair_t("t", 7));
        map.insert(foreign_pair);
        map.insert(foreign_pair_t("y", 8));

        std::vector<pair_t> const qwerty_vec = {
            {"e", 5}, {"q", 3}, {"r", 6}, {"t", 7}, {"w", 4}, {"y", 8}};
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
        map.insert({pair_t{"key1", 1}, pair_t{"key2", 2}, pair_t{"key0", 0}});
        HPX_TEST_EQ(map.size(), 3u);
        HPX_TEST(!(map.empty()));
        HPX_TEST(std::equal(map.begin(), map.end(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }

    {
        rev_fmap_t map;
        map.insert({pair_t{"key1", 1}, pair_t{"key2", 2}, pair_t{"key0", 0}});
        HPX_TEST_EQ(map.size(), 3u);
        HPX_TEST(!(map.empty()));
        HPX_TEST(std::equal(map.rbegin(), map.rend(), sorted_vec.begin(),
            sorted_vec.end(), pair_cmp));
    }

    {
        // This breaks the ctor's contract, since c.keys are not in sorted
        // order, but it's useful to verify that the resulting contents were
        // not touched.
        fmap_t map;
        map.insert(hpx::detail::sorted_unique,
            {pair_t{"key1", 1}, pair_t{"key2", 2}, pair_t{"key0", 0}});
        HPX_TEST_EQ(map.size(), 3u);
        HPX_TEST(!(map.empty()));
        HPX_TEST(std::equal(
            map.begin(), map.end(), vec.begin(), vec.end(), pair_cmp));
    }

    {
        // NOTE: These calls exercise all the try_emplace's.

        fmap_t map;
        map.try_emplace(map.begin(), pair1.first, pair1.second);
        map.try_emplace(map.begin(), std::string("lucky"), 13);

        std::vector<pair_t> const local_vec = {{"lucky", 13}, {"q", 3}};
        HPX_TEST(std::equal(map.begin(), map.end(), local_vec.begin(),
            local_vec.end(), pair_cmp));
    }

    {
        // NOTE: These calls exercise all the try_emplace's.

        fmap_t map;
        map.insert_or_assign(map.begin(), pair1.first, pair1.second);
        map.insert_or_assign(
            map.begin(), std::string(pair1.first), pair1.second);

        std::vector<pair_t> const local_vec = {{"q", 3}};
        HPX_TEST(std::equal(map.begin(), map.end(), local_vec.begin(),
            local_vec.end(), pair_cmp));
    }
}

void std_flat_map_extract_replace()
{
    using fmap_t = hpx::detail::flat_map<std::string, int>;
    using pair_t = std::pair<std::string, int>;

    fmap_t::containers const sorted_c = {{"key0", "key1", "key2"}, {0, 1, 2}};
    std::vector<pair_t> const vec = {{"key1", 1}, {"key2", 2}, {"key0", 0}};

    fmap_t map1(vec);
    fmap_t const map1_copy = map1;

    fmap_t::containers extracted_c = std::move(map1).extract();
    HPX_TEST_EQ(extracted_c.keys, sorted_c.keys);
    HPX_TEST_EQ(extracted_c.values, sorted_c.values);

    fmap_t map2;
    map2.replace(std::move(extracted_c.keys), std::move(extracted_c.values));
    HPX_TEST_EQ(map2, map1_copy);
}

void std_flat_map_erase_swap()
{
    using fmap_t = hpx::detail::flat_map<std::string, int>;
    using pair_t = std::pair<std::string, int>;

    auto pair_cmp = [](auto lhs, auto rhs) {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    };

    std::vector<pair_t> const vec = {{"key0", 0}, {"key1", 1}, {"key2", 2}};

    {
        fmap_t map(vec);
        map.erase(map.begin());
        std::vector<pair_t> const local_vec = {{"key1", 1}, {"key2", 2}};
        HPX_TEST(std::equal(map.begin(), map.end(), local_vec.begin(),
            local_vec.end(), pair_cmp));
    }

    {
        fmap_t map(vec);
        map.erase(map.cbegin());
        std::vector<pair_t> const local_vec = {{"key1", 1}, {"key2", 2}};
        HPX_TEST(std::equal(map.begin(), map.end(), local_vec.begin(),
            local_vec.end(), pair_cmp));
    }

    {
        fmap_t map(vec);
        map.erase("key0");
        std::vector<pair_t> const local_vec = {{"key1", 1}, {"key2", 2}};
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

void std_flat_map_count_contains()
{
    using fmap_t = hpx::detail::flat_map<std::string, int>;
    using pair_t = std::pair<std::string, int>;

    std::vector<pair_t> const vec = {{"key1", 1}, {"key2", 2}, {"key0", 0}};

    fmap_t const map(vec);
    HPX_TEST_EQ(map.count("key0"), 1u);
    HPX_TEST_EQ(map.count("key10"), 0u);
    HPX_TEST(map.count("key0"));
    HPX_TEST(!(map.count("key10")));
}

void std_flat_map_equal_range()
{
    using fmap_t = hpx::detail::flat_map<std::string, int>;
    using pair_t = std::pair<std::string, int>;

    std::vector<pair_t> const vec = {{"key1", 1}, {"key2", 2}, {"key0", 0}};

    fmap_t const map(vec);
    auto const eq_range = map.equal_range("key0");
    HPX_TEST_EQ(eq_range.first, map.begin());
    HPX_TEST_EQ(eq_range.second, map.begin() + 1);
    auto const empty_eq_range = map.equal_range("");
    HPX_TEST_EQ(empty_eq_range.first, empty_eq_range.second);
}

void std_flat_map_comparisons()
{
    using fmap_t = hpx::detail::flat_map<std::string, int>;
    using pair_t = std::pair<std::string, int>;

    std::vector<pair_t> const vec = {{"key0", 0}, {"key1", 1}, {"key2", 2}};

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
    test_std_flat_map_iterator();
    std_flat_map_ctors_iterators();
    std_flat_map_ctors_allocators_iterators();
    std_flat_map_index_at();
    std_flat_map_emplace_insert();
    std_flat_map_extract_replace();
    std_flat_map_erase_swap();
    std_flat_map_count_contains();
    std_flat_map_equal_range();
    std_flat_map_comparisons();

    return hpx::util::report_errors();
}
