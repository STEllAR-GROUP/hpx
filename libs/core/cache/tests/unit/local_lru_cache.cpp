//  Copyright (c) 2008-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/cache/entries/lru_entry.hpp>
#include <hpx/cache/local_cache.hpp>
#include <hpx/cache/statistics/local_statistics.hpp>
#include <hpx/modules/testing.hpp>

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
void test_lru_insert()
{
    using entry_type = hpx::util::cache::entries::lru_entry<std::string>;
    using cache_type = hpx::util::cache::local_cache<std::string, entry_type>;

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
}

///////////////////////////////////////////////////////////////////////////////
void test_lru_insert_with_touch()
{
    using entry_type = hpx::util::cache::entries::lru_entry<std::string>;
    using cache_type = hpx::util::cache::local_cache<std::string, entry_type>;

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

    // there should be 3 items in the cache, and white should be there as well
    HPX_TEST_EQ(static_cast<cache_type::size_type>(3), c.size());
    HPX_TEST(c.holds_key("white"));
}

///////////////////////////////////////////////////////////////////////////////
void test_lru_clear()
{
    using entry_type = hpx::util::cache::entries::lru_entry<std::string>;
    using cache_type = hpx::util::cache::local_cache<std::string, entry_type>;

    cache_type c(3);

    HPX_TEST_EQ(static_cast<cache_type::size_type>(3), c.capacity());

    // insert all items into the cache
    for (data* d = &cache_entries[0]; d->key != nullptr; ++d)
    {
        HPX_TEST(c.insert(d->key, d->value));
        HPX_TEST_LTE(c.size(), static_cast<cache_type::size_type>(3));
    }

    c.clear();

    // there should be no items in the cache
    HPX_TEST_EQ(static_cast<cache_type::size_type>(0), c.size());
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

void test_lru_erase_one()
{
    using entry_type = hpx::util::cache::entries::lru_entry<std::string>;
    using cache_type = hpx::util::cache::local_cache<std::string, entry_type>;

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

    c.erase(erase_func("blue"));

    // there should be 2 items in the cache
    HPX_TEST(!c.get_entry("blue", blue));
    HPX_TEST_EQ(static_cast<cache_type::size_type>(2), c.size());
}

///////////////////////////////////////////////////////////////////////////////
void test_lru_update()
{
    using entry_type = hpx::util::cache::entries::lru_entry<std::string>;
    using cache_type = hpx::util::cache::local_cache<std::string, entry_type>;

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
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    test_lru_insert();
    test_lru_insert_with_touch();
    test_lru_clear();
    test_lru_erase_one();
    test_lru_update();

    return hpx::util::report_errors();
}
