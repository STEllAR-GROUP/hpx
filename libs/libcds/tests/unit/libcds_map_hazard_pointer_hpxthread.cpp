//  Copyright (c) 2020 Weile Wei
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/libcds/hpx_tls_manager.hpp>
#include <hpx/modules/testing.hpp>

#include <cds/container/feldman_hashmap_hp.h>
#include <cds/container/michael_kvlist_hp.h>
#include <cds/container/michael_list_hp.h>
#include <cds/container/michael_map.h>
#include <cds/container/split_list_map.h>
#include <cds/init.h>    // for cds::Initialize and cds::Terminate

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <deque>
#include <functional>
#include <iterator>
#include <random>
#include <string>
#include <vector>

template <typename T>
struct atomwrapper
{
    std::atomic<T> value;

    atomwrapper()
      : value()
    {
    }

    atomwrapper(const std::atomic<T>& a)
      : value(a.load())
    {
    }

    atomwrapper(const atomwrapper& other)
      : value(other.value.load())
    {
    }

    atomwrapper& operator=(const atomwrapper& other)
    {
        value.store(other.value.load());
    }
};

using gc_type = cds::gc::custom_HP<cds::gc::hp::details::HPXDataHolder>;
using key_type = std::size_t;
using value_type = atomwrapper<std::size_t>;

// Declare michael_list traits
struct michael_list_trait : public cds::container::michael_list::traits
{
    typedef std::less<std::size_t> less;
};
// Declare traits-based list
using michaelkvlist = cds::container::MichaelKVList<gc_type, key_type,
    value_type, michael_list_trait>;

// Declare split_list_traits
struct split_list_traits : public cds::container::split_list::traits
{
    // what type of ordered list we want to use
    typedef cds::container::michael_list_tag ordered_list;
    // hash functor for the key stored in split-list map
    typedef std::hash<key_type> hash;

    // Type traits for our MichaelList class
    struct ordered_list_traits : public cds::container::michael_list::traits
    {
        // use our std::less predicate as comparator to order list nodes
        typedef std::less<key_type> less;
    };
};

template <typename Map>
void run(Map& map, const std::size_t n_items, const std::size_t n_threads)
{
    // a reference counter vector to keep track of counter on
    // value [0, 1000)
    std::array<std::atomic<std::size_t>, 1000> counter_vec;
    std::fill(std::begin(counter_vec), std::end(counter_vec), 0);

    std::vector<hpx::thread> threads;

    // each thread inserts number of n_items/n_threads items to the map
    std::vector<std::vector<std::size_t>> numbers_vec(
        n_threads, std::vector<std::size_t>(n_items / n_threads, 0));

    // map init
    std::size_t val = 0;
    while (val < 1000)
    {
        std::atomic<std::size_t> counter(0);
        map.insert(val, counter);
        val++;
    }

    auto insert_val_and_increase_counter =
        [&](std::vector<std::size_t>& number_vec) {
            hpx::cds::hpxthread_manager_wrapper cds_hpx_wrap;
            for (auto val : number_vec)
            {
                if (rand() % 10 == 0)
                    hpx::this_thread::yield();
                typename Map::guarded_ptr gp;

                gp = map.get(val);
                HPX_ASSERT(gp);
                (gp->second).value++;
                counter_vec[gp->first]++;
            }
        };

    for (auto& v : numbers_vec)
    {
        std::generate(v.begin(), v.end(),
            [&]() { return rand() % (n_items / n_threads); });

        threads.emplace_back(insert_val_and_increase_counter, std::ref(v));
    }

    // wait for all threads to complete
    for (auto& t : threads)
    {
        if (t.joinable())
            t.join();
    }

    for (auto it = map.cbegin(); it != map.cend(); ++it)
    {
        HPX_TEST_EQ(counter_vec[it->first], (it->second).value);
    }
}

int hpx_main(int, char**)
{
    using feldmanhash_map_type =
        cds::container::FeldmanHashMap<gc_type, key_type, value_type>;
    using michaelhash_map_type =
        cds::container::MichaelHashMap<gc_type, michaelkvlist>;
    using splitlist_map_type = cds::container::SplitListMap<gc_type, key_type,
        value_type, split_list_traits>;

    const std::size_t max_hazard_pointer =
        (std::max)({feldmanhash_map_type::c_nHazardPtrCount + 1,
            michaelhash_map_type::c_nHazardPtrCount + 1,
            splitlist_map_type::c_nHazardPtrCount + 1});

    // Initialize libcds and hazard pointer
    const std::size_t n_threads = 128;
    hpx::cds::libcds_wrapper cds_init_wrapper(
        hpx::cds::smr_t::hazard_pointer_hpxthread, max_hazard_pointer, n_threads, 16);

    {
        // enable this thread/task to run using libcds support
        hpx::cds::hpxthread_manager_wrapper cds_hpx_wrap;

        const std::size_t n_items = 10000;
        // load factor: estimation of max number of items in the bucket
        const std::size_t n_load_factor = 100;

        feldmanhash_map_type feldmanhash_map;
        michaelhash_map_type michaelhash_map(n_items, n_load_factor);
        splitlist_map_type splitlist_map(n_items, n_load_factor);

        run(feldmanhash_map, n_items, n_threads);
        run(michaelhash_map, n_items, n_threads);
        run(splitlist_map, n_items, n_threads);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}
