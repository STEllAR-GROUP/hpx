//  Copyright (c) 2008 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <string>

#include <boost/cache/entries/lru_entry.hpp>
#include <boost/cache/local_cache.hpp>

#include <boost/detail/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
struct data
{
    data(char const* const k, char const* const v)
      : key(k), value(v)
    {}

    char const* const key;
    char const* const value;
};

data entries[] =
{
    data ("white", "255,255,255"),
    data ("yellow", "255,255,0"),
    data ("green", "0,255,0"),
    data ("blue", "0,0,255"),
    data ("magenta", "255,0,255"),
    data ("black", "0,0,0"),
    data (NULL, NULL)
};

///////////////////////////////////////////////////////////////////////////////
void test_lru_insert()
{
    typedef boost::cache::entries::lru_entry<std::string> entry_type;
    typedef boost::cache::local_cache<std::string, entry_type> cache_type;

    cache_type c(3);

    BOOST_TEST(3 == c.capacity());

    // insert all items into the cache
    for (data* d = &entries[0]; d->key != NULL; ++d) {
        BOOST_TEST(c.insert(d->key, d->value));
        BOOST_TEST(3 >= c.size());
    }

    // there should be 3 items in the cache
    BOOST_TEST(3 == c.size());
}

///////////////////////////////////////////////////////////////////////////////
void test_lru_insert_with_touch()
{
    typedef boost::cache::entries::lru_entry<std::string> entry_type;
    typedef boost::cache::local_cache<std::string, entry_type> cache_type;

    cache_type c(3);

    BOOST_TEST(3 == c.capacity());

    // insert 3 items into the cache
    int i = 0;
    data* d = &entries[0];

    for (/**/; i < 3 && d->key != NULL; ++d, ++i) {
        BOOST_TEST(c.insert(d->key, d->value));
        BOOST_TEST(3 >= c.size());
    }

    BOOST_TEST(3 == c.size());

    // now touch the first item
    std::string white;
    BOOST_TEST(c.get_entry("white", white));
    BOOST_TEST(white == "255,255,255");

    // add two more items
    for (i = 0; i < 2 && d->key != NULL; ++d, ++i) {
        BOOST_TEST(c.insert(d->key, d->value));
        BOOST_TEST(3 == c.size());
    }

    // there should be 3 items in the cache, and white should be there as well
    BOOST_TEST(3 == c.size());
    BOOST_TEST(c.holds_key("white"));
}

///////////////////////////////////////////////////////////////////////////////
void test_lru_clear()
{
    typedef boost::cache::entries::lru_entry<std::string> entry_type;
    typedef boost::cache::local_cache<std::string, entry_type> cache_type;

    cache_type c(3);

    BOOST_TEST(3 == c.capacity());

    // insert all items into the cache
    for (data* d = &entries[0]; d->key != NULL; ++d) {
        BOOST_TEST(c.insert(d->key, d->value));
        BOOST_TEST(3 >= c.size());
    }

    c.clear();

    // there should be no items in the cache
    BOOST_TEST(0 == c.size());
}

///////////////////////////////////////////////////////////////////////////////
struct erase_func
{
    erase_func(std::string const& key)
      : key_(key)
    {}

    template <typename Entry>
    bool operator()(Entry const& e) const
    {
        return key_ == e.first;
    }

    std::string key_;
};

void test_lru_erase_one()
{
    typedef boost::cache::entries::lru_entry<std::string> entry_type;
    typedef boost::cache::local_cache<std::string, entry_type> cache_type;

    cache_type c(3);

    BOOST_TEST(3 == c.capacity());

    // insert all items into the cache
    for (data* d = &entries[0]; d->key != NULL; ++d) {
        BOOST_TEST(c.insert(d->key, d->value));
        BOOST_TEST(3 >= c.size());
    }

    entry_type blue;
    BOOST_TEST(c.get_entry("blue", blue));

    c.erase(erase_func("blue"));

    // there should be 2 items in the cache
    BOOST_TEST(!c.get_entry("blue", blue));
    BOOST_TEST(2 == c.size());
}

///////////////////////////////////////////////////////////////////////////////
void test_lru_update()
{
    typedef boost::cache::entries::lru_entry<std::string> entry_type;
    typedef boost::cache::local_cache<std::string, entry_type> cache_type;

    cache_type c(4);    // this time we can hold 4 items

    BOOST_TEST(4 == c.capacity());

    // insert 3 items into the cache
    int i = 0;
    data* d = &entries[0];

    for (/**/; i < 3 && d->key != NULL; ++d, ++i) {
        BOOST_TEST(c.insert(d->key, d->value));
        BOOST_TEST(3 >= c.size());
    }

    // there should be 3 items in the cache
    BOOST_TEST(3 == c.size());

    // now update some items
    BOOST_TEST(c.update("black", "255,0,0"));     // isn't in the cache
    BOOST_TEST(4 == c.size());

    BOOST_TEST(c.update("yellow", "255,0,0"));
    BOOST_TEST(4 == c.size());

    std::string yellow;
    BOOST_TEST(c.get_entry("yellow", yellow));
    BOOST_TEST(yellow == "255,0,0");
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    test_lru_insert();
    test_lru_insert_with_touch();
    test_lru_clear();
    test_lru_erase_one();
    test_lru_update();
    return boost::report_errors();
}

