//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#if defined(_MSC_VER)
// conversion from uint64_t -> double, possible loss of precision
#pragma warning (disable: 4244)
#endif

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/cache/entries/lfu_entry.hpp>
#include <hpx/cache/local_cache.hpp>
#include <hpx/cache/statistics/local_full_statistics.hpp>
#include <hpx/preprocessor/stringize.hpp>
#include <hpx/statistics/histogram.hpp>
#include <hpx/modules/testing.hpp>

#include <hpx/modules/program_options.hpp>
#include <boost/accumulators/accumulators.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Below is a copy of the original AGAS cache
typedef hpx::util::cache::entries::lfu_entry<hpx::agas::gva> gva_entry_type;

struct gva_cache_key
{
private:
    typedef std::pair<hpx::naming::gid_type, hpx::naming::gid_type> key_type;

    key_type key_;

public:
    gva_cache_key()
      : key_()
    {
    }

    explicit gva_cache_key(
            hpx::naming::gid_type const& id, std::uint64_t count = 1)
      : key_(hpx::naming::detail::get_stripped_gid(id),
            hpx::naming::detail::get_stripped_gid(id) + (count - 1))
    {
        HPX_TEST(count);
    }

    hpx::naming::gid_type get_gid() const
    {
        return key_.first;
    }

    std::uint64_t get_count() const
    {
        hpx::naming::gid_type const size = key_.second - key_.first;
        HPX_TEST_EQ(size.get_msb(), std::uint64_t(0));
        return size.get_lsb();
    }

    friend bool operator<(gva_cache_key const& lhs, gva_cache_key const& rhs)
    {
        return lhs.key_.second < rhs.key_.first;
    }

    friend bool operator==(gva_cache_key const& lhs, gva_cache_key const& rhs)
    {
        // Direct hit
        if (lhs.key_ == rhs.key_)
        {
            return true;
        }

        // Is lhs in rhs?
        if (1 == lhs.get_count() && 1 != rhs.get_count())
        {
            return rhs.key_.first <= lhs.key_.first &&
                lhs.key_.second <= rhs.key_.second;
        }

        // Is rhs in lhs?
        else if (1 != lhs.get_count() && 1 == rhs.get_count())
        {
            return lhs.key_.first <= rhs.key_.first &&
                rhs.key_.second <= lhs.key_.second;
        }

        return false;
    }
};

struct gva_erase_policy
{
    gva_erase_policy(hpx::naming::gid_type const& id, std::uint64_t count)
      : entry(id, count)
    {}

    typedef std::pair<gva_cache_key, gva_entry_type> entry_type;

    bool operator()(entry_type const& p) const
    {
        return p.first == entry;
    }

    gva_cache_key entry;
};

typedef hpx::util::cache::local_cache<
    gva_cache_key, gva_entry_type, std::less<gva_entry_type>,
    hpx::util::cache::policies::always<gva_entry_type>,
    std::map<gva_cache_key, gva_entry_type>,
    hpx::util::cache::statistics::local_full_statistics
> gva_cache_type;

///////////////////////////////////////////////////////////////////////////////
void calculate_histogram(std::string const& prefix,
    std::vector<std::uint64_t> const& timings)
{
    auto minmax = std::minmax_element(timings.begin(), timings.end());

    typedef boost::accumulators::accumulator_set<
            std::uint64_t,
            boost::accumulators::features<hpx::util::tag::histogram>
        > histogram_collector_type;

    histogram_collector_type hist(
        hpx::util::tag::histogram::num_bins = 20,
        hpx::util::tag::histogram::min_range = *minmax.first,
        hpx::util::tag::histogram::max_range = *minmax.second);

    for (std::int64_t t : timings)
    {
        hist(t);
    }

    std::cout << prefix << ": ";

    bool first = true;
    auto data = hpx::util::histogram(hist);
    for (auto const& item : data)
    {
        if (!first)
        {
            std::cout << ", ";
        }
        first = false;
        std::cout << std::setprecision(3) << std::setw(6) << item.second;
    }

    std::cout << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
void test_insert(gva_cache_type& cache, std::size_t num_entries)
{
    hpx::naming::gid_type locality = hpx::get_locality();
    std::uint32_t ct = hpx::components::component_invalid;

    std::vector<std::uint64_t> timings;
    timings.reserve(num_entries);

    for (std::size_t i = 0; i != num_entries; ++i)
    {
        gva_cache_key key(hpx::detail::get_next_id(), 1);
        hpx::agas::gva value(locality, ct, 1, std::uint64_t(0), 0);

        std::uint64_t t = hpx::chrono::high_resolution_clock::now();

        cache.insert(key, value);

        timings.push_back(hpx::chrono::high_resolution_clock::now() - t);
    }

    calculate_histogram("insert", timings);
}

void test_get(gva_cache_type& cache, hpx::naming::gid_type first_key)
{
    std::vector<std::uint64_t> timings;
    timings.reserve(cache.size());

    for (std::size_t i = 0; i != cache.size(); ++i)
    {
        gva_cache_key key(++first_key, 1);
        gva_cache_key idbase;
        gva_cache_type::entry_type e;

        std::uint64_t t = hpx::chrono::high_resolution_clock::now();

        cache.get_entry(key, idbase, e);

        timings.push_back(hpx::chrono::high_resolution_clock::now() - t);
    }

    calculate_histogram("   get", timings);
}

void test_update(gva_cache_type& cache, hpx::naming::gid_type first_key)
{
    hpx::naming::gid_type locality = hpx::get_locality();
    std::uint32_t ct = hpx::components::component_invalid;

    std::vector<std::uint64_t> timings;
    timings.reserve(cache.size());

    for (std::size_t i = 0; i != cache.size(); ++i)
    {
        gva_cache_key key(++first_key, 1);
        hpx::agas::gva value(locality, ct, 1, std::uint64_t(1), 1);

        std::uint64_t t = hpx::chrono::high_resolution_clock::now();

        cache.update(key, value);

        timings.push_back(hpx::chrono::high_resolution_clock::now() - t);
    }

    calculate_histogram("update", timings);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    std::size_t cache_size = HPX_AGAS_LOCAL_CACHE_SIZE;
    if (vm.count("cache_size"))
        cache_size = vm["cache_size"].as<std::size_t>();

    std::size_t num_entries = 1000;
    if (vm.count("num_entries"))
        num_entries = vm["num_entries"].as<std::size_t>();

    gva_cache_type cache;
    cache.reserve(cache_size);

    hpx::naming::gid_type first_key = hpx::detail::get_next_id();

    hpx::chrono::high_resolution_timer t1;

    test_insert(cache, num_entries);
    test_get(cache, first_key);
    test_update(cache, first_key);

    double elapsed = t1.elapsed();
    hpx::util::print_cdash_timing("AGASCache", elapsed);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("cache_size", value<std::size_t>(),
         "initial cache size (default: "
         HPX_PP_STRINGIZE(HPX_AGAS_LOCAL_CACHE_SIZE_PER_THREAD) ")")
        ("num_entries,n", value<std::size_t>(),
         "number of items to insert into cache (default: 1000)")
        ;

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args);
}
#endif
