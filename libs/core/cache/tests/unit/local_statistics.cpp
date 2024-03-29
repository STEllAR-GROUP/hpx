//  Copyright (c) 2008-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/cache/entries/lru_entry.hpp>
#include <hpx/cache/local_cache.hpp>
#include <hpx/cache/statistics/local_statistics.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <functional>
#include <map>
#include <string>

///////////////////////////////////////////////////////////////////////////////
struct data
{
    constexpr data(char const* const k, char const* const v) noexcept
      : key(k)
      , value(v)
    {
    }

    char const* const key;
    char const* const value;
};

data cache_entries[] = {data("white", "255,255,255"),
    data("yellow", "255,255,0"), data("green", "0,255,0"),
    data("blue", "0,0,255"), data("magenta", "255,0,255"),
    data("black", "0,0,0"), data(nullptr, nullptr)};

///////////////////////////////////////////////////////////////////////////////
void test_statistics_insert()
{
    using namespace hpx::util::cache;

    using entry_type = entries::lru_entry<std::string>;
    using cache_type = local_cache<std::string, entry_type,
        std::less<entry_type>, policies::always<entry_type>,
        std::map<std::string, entry_type>, statistics::local_statistics>;

    cache_type c(3);

    HPX_TEST_EQ(static_cast<cache_type::size_type>(3), c.capacity());

    // insert all items into the cache
    for (data* d = &cache_entries[0]; d->key != nullptr; ++d)
    {
        HPX_TEST(c.insert(d->key, d->value));
        HPX_TEST_LTE(c.size(), static_cast<cache_type::size_type>(3));
    }

    // there should be 3 items in the cache
    HPX_TEST_EQ(static_cast<cache_type::size_type>(3), c.size());

    // retrieve statistics
    statistics::local_statistics const& stats = c.get_statistics();
    HPX_TEST_EQ(static_cast<std::size_t>(0), stats.hits());
    HPX_TEST_EQ(static_cast<std::size_t>(0), stats.misses());
    HPX_TEST_EQ(static_cast<std::size_t>(6), stats.insertions());
    HPX_TEST_EQ(static_cast<std::size_t>(3), stats.evictions());
}

///////////////////////////////////////////////////////////////////////////////
void test_statistics_insert_with_touch()
{
    using namespace hpx::util::cache;

    using entry_type = entries::lru_entry<std::string>;
    using cache_type = local_cache<std::string, entry_type,
        std::less<entry_type>, policies::always<entry_type>,
        std::map<std::string, entry_type>, statistics::local_statistics>;

    cache_type c(3);

    HPX_TEST_EQ(static_cast<cache_type::size_type>(3), c.capacity());

    // insert 3 items into the cache
    int i = 0;
    data* d = &cache_entries[0];

    for (/**/; i < 3 && d->key != nullptr; ++d, ++i)
    {
        HPX_TEST(c.insert(d->key, d->value));
        HPX_TEST_LTE(c.size(), static_cast<cache_type::size_type>(3));
    }

    HPX_TEST_EQ(static_cast<cache_type::size_type>(3), c.size());

    // now touch the first item
    std::string white;
    HPX_TEST(c.get_entry("white", white));
    HPX_TEST_EQ(white, "255,255,255");

    // add two more items
    for (i = 0; i < 2 && d->key != nullptr; ++d, ++i)
    {
        HPX_TEST(c.insert(d->key, d->value));
        HPX_TEST_EQ(static_cast<cache_type::size_type>(3), c.size());
    }

    // provoke a miss
    std::string yellow;
    HPX_TEST(!c.get_entry("yellow", yellow));

    // there should be 3 items in the cache, and white should be there as well
    HPX_TEST_EQ(static_cast<cache_type::size_type>(3), c.size());
    HPX_TEST(c.holds_key("white"));    // does not call the entry's touch()

    // retrieve statistics
    statistics::local_statistics const& stats = c.get_statistics();
    HPX_TEST_EQ(static_cast<std::size_t>(1), stats.hits());
    HPX_TEST_EQ(static_cast<std::size_t>(1), stats.misses());
    HPX_TEST_EQ(static_cast<std::size_t>(5), stats.insertions());
    HPX_TEST_EQ(static_cast<std::size_t>(2), stats.evictions());
}

///////////////////////////////////////////////////////////////////////////////
void test_statistics_update()
{
    using namespace hpx::util::cache;

    using entry_type = entries::lru_entry<std::string>;
    using cache_type = local_cache<std::string, entry_type,
        std::less<entry_type>, policies::always<entry_type>,
        std::map<std::string, entry_type>, statistics::local_statistics>;

    cache_type c(4);    // this time we can hold 4 items

    HPX_TEST_EQ(static_cast<cache_type::size_type>(4), c.capacity());

    // insert 3 items into the cache
    int i = 0;
    data* d = &cache_entries[0];

    for (/**/; i < 3 && d->key != nullptr; ++d, ++i)
    {
        HPX_TEST(c.insert(d->key, d->value));
        HPX_TEST_LTE(c.size(), static_cast<cache_type::size_type>(3));
    }

    // there should be 3 items in the cache
    HPX_TEST_EQ(static_cast<cache_type::size_type>(3), c.size());

    // now update some items
    HPX_TEST(c.update("black", "255,0,0"));    // isn't in the cache
    HPX_TEST_EQ(static_cast<cache_type::size_type>(4), c.size());

    HPX_TEST(c.update("yellow", "255,0,0"));
    HPX_TEST_EQ(static_cast<cache_type::size_type>(4), c.size());

    std::string yellow;
    HPX_TEST(c.get_entry("yellow", yellow));
    HPX_TEST_EQ(yellow, "255,0,0");

    // retrieve statistics
    statistics::local_statistics const& stats = c.get_statistics();
    HPX_TEST_EQ(static_cast<std::size_t>(2), stats.hits());
    HPX_TEST_EQ(static_cast<std::size_t>(1), stats.misses());
    HPX_TEST_EQ(static_cast<std::size_t>(4), stats.insertions());
    HPX_TEST_EQ(static_cast<std::size_t>(0), stats.evictions());
}

///////////////////////////////////////////////////////////////////////////////
struct erase_func
{
    erase_func(std::string const& key)
      : key_(key)
    {
    }

    template <typename Entry>
    bool operator()(Entry const& e) const
    {
        return key_ == e.first;
    }

    std::string key_;
};

void test_statistics_erase_one()
{
    using namespace hpx::util::cache;

    using entry_type = entries::lru_entry<std::string>;
    using cache_type = local_cache<std::string, entry_type,
        std::less<entry_type>, policies::always<entry_type>,
        std::map<std::string, entry_type>, statistics::local_statistics>;

    cache_type c(3);

    HPX_TEST_EQ(static_cast<cache_type::size_type>(3), c.capacity());

    // insert all items into the cache
    for (data* d = &cache_entries[0]; d->key != nullptr; ++d)
    {
        HPX_TEST(c.insert(d->key, d->value));
        HPX_TEST_LTE(c.size(), static_cast<cache_type::size_type>(3));
    }

    entry_type blue;
    HPX_TEST(c.get_entry("blue", blue));

    c.erase(erase_func("blue"));    // removals count as eviction

    // there should be 2 items in the cache
    HPX_TEST(!c.get_entry("blue", blue));
    HPX_TEST_EQ(static_cast<cache_type::size_type>(2), c.size());

    // retrieve statistics
    statistics::local_statistics const& stats = c.get_statistics();
    HPX_TEST_EQ(static_cast<std::size_t>(1), stats.hits());
    HPX_TEST_EQ(static_cast<std::size_t>(1), stats.misses());
    HPX_TEST_EQ(static_cast<std::size_t>(6), stats.insertions());
    HPX_TEST_EQ(static_cast<std::size_t>(4), stats.evictions());
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    test_statistics_insert();
    test_statistics_insert_with_touch();
    test_statistics_update();
    test_statistics_erase_one();

    return hpx::util::report_errors();
}
