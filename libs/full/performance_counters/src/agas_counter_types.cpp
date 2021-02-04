//  Copyright (c) 2011-2021 Hartmut Kaiser
//  Copyright (c) 2016 Parsa Amini
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/agas/addressing_service.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/performance_counters/agas_counter_types.hpp>
#include <hpx/performance_counters/component_namespace_counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/locality_namespace_counters.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/performance_counters/primary_namespace_counters.hpp>
#include <hpx/performance_counters/symbol_namespace_counters.hpp>

#include <cstdint>

namespace hpx { namespace performance_counters {

    /// Install performance counter types exposing properties from the local cache.
    void register_agas_counter_types(agas::addressing_service& client)
    {
        // install
        util::function_nonser<std::int64_t(bool)> cache_entries(
            util::bind_front(
                &agas::addressing_service::get_cache_entries, &client));
        util::function_nonser<std::int64_t(bool)> cache_hits(util::bind_front(
            &agas::addressing_service::get_cache_hits, &client));
        util::function_nonser<std::int64_t(bool)> cache_misses(util::bind_front(
            &agas::addressing_service::get_cache_misses, &client));
        util::function_nonser<std::int64_t(bool)> cache_evictions(
            util::bind_front(
                &agas::addressing_service::get_cache_evictions, &client));
        util::function_nonser<std::int64_t(bool)> cache_insertions(
            util::bind_front(
                &agas::addressing_service::get_cache_insertions, &client));

        util::function_nonser<std::int64_t(bool)> cache_get_entry_count(
            util::bind_front(
                &agas::addressing_service::get_cache_get_entry_count, &client));
        util::function_nonser<std::int64_t(bool)> cache_insertion_count(
            util::bind_front(
                &agas::addressing_service::get_cache_insertion_entry_count,
                &client));
        util::function_nonser<std::int64_t(bool)> cache_update_entry_count(
            util::bind_front(
                &agas::addressing_service::get_cache_update_entry_count,
                &client));
        util::function_nonser<std::int64_t(bool)> cache_erase_entry_count(
            util::bind_front(
                &agas::addressing_service::get_cache_erase_entry_count,
                &client));

        util::function_nonser<std::int64_t(bool)> cache_get_entry_time(
            util::bind_front(
                &agas::addressing_service::get_cache_get_entry_time, &client));
        util::function_nonser<std::int64_t(bool)> cache_insertion_time(
            util::bind_front(
                &agas::addressing_service::get_cache_insertion_entry_time,
                &client));
        util::function_nonser<std::int64_t(bool)> cache_update_entry_time(
            util::bind_front(
                &agas::addressing_service::get_cache_update_entry_time,
                &client));
        util::function_nonser<std::int64_t(bool)> cache_erase_entry_time(
            util::bind_front(
                &agas::addressing_service::get_cache_erase_entry_time,
                &client));

        using util::placeholders::_1;
        using util::placeholders::_2;
        performance_counters::generic_counter_type_data const counter_types[] =
            {
                {"/agas/count/cache/entries", performance_counters::counter_raw,
                    "returns the number of cache entries in the AGAS cache",
                    HPX_PERFORMANCE_COUNTER_V1,
                    util::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        cache_entries, _2),
                    &performance_counters::locality_counter_discoverer, ""},
                {"/agas/count/cache/hits",
                    performance_counters::counter_monotonically_increasing,
                    "returns the number of cache hits while accessing the AGAS "
                    "cache",
                    HPX_PERFORMANCE_COUNTER_V1,
                    util::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        cache_hits, _2),
                    &performance_counters::locality_counter_discoverer, ""},
                {"/agas/count/cache/misses",
                    performance_counters::counter_monotonically_increasing,
                    "returns the number of cache misses while accessing the "
                    "AGAS cache",
                    HPX_PERFORMANCE_COUNTER_V1,
                    util::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        cache_misses, _2),
                    &performance_counters::locality_counter_discoverer, ""},
                {"/agas/count/cache/evictions",
                    performance_counters::counter_monotonically_increasing,
                    "returns the number of cache evictions from the AGAS cache",
                    HPX_PERFORMANCE_COUNTER_V1,
                    util::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        cache_evictions, _2),
                    &performance_counters::locality_counter_discoverer, ""},
                {"/agas/count/cache/insertions",
                    performance_counters::counter_monotonically_increasing,
                    "returns the number of cache insertions into the AGAS "
                    "cache",
                    HPX_PERFORMANCE_COUNTER_V1,
                    util::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        cache_insertions, _2),
                    &performance_counters::locality_counter_discoverer, ""},
                {"/agas/count/cache/get_entry",
                    performance_counters::counter_monotonically_increasing,
                    "returns the number of invocations of get_entry function "
                    "of the AGAS cache",
                    HPX_PERFORMANCE_COUNTER_V1,
                    util::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        cache_get_entry_count, _2),
                    &performance_counters::locality_counter_discoverer, ""},
                {"/agas/count/cache/insert_entry",
                    performance_counters::counter_monotonically_increasing,
                    "returns the number of invocations of insert function of "
                    "the AGAS cache",
                    HPX_PERFORMANCE_COUNTER_V1,
                    util::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        cache_insertion_count, _2),
                    &performance_counters::locality_counter_discoverer, ""},
                {"/agas/count/cache/update_entry",
                    performance_counters::counter_monotonically_increasing,
                    "returns the number of invocations of update_entry "
                    "function of the AGAS cache",
                    HPX_PERFORMANCE_COUNTER_V1,
                    util::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        cache_update_entry_count, _2),
                    &performance_counters::locality_counter_discoverer, ""},
                {"/agas/count/cache/erase_entry",
                    performance_counters::counter_monotonically_increasing,
                    "returns the number of invocations of erase_entry function "
                    "of the AGAS cache",
                    HPX_PERFORMANCE_COUNTER_V1,
                    util::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        cache_erase_entry_count, _2),
                    &performance_counters::locality_counter_discoverer, ""},
                {"/agas/time/cache/get_entry",
                    performance_counters::counter_monotonically_increasing,
                    "returns the overall time spent executing of the get_entry "
                    "API function of the AGAS cache",
                    HPX_PERFORMANCE_COUNTER_V1,
                    util::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        cache_get_entry_time, _2),
                    &performance_counters::locality_counter_discoverer, "ns"},
                {"/agas/time/cache/insert_entry",
                    performance_counters::counter_monotonically_increasing,
                    "returns the overall time spent executing of the "
                    "insert_entry API function of the AGAS cache",
                    HPX_PERFORMANCE_COUNTER_V1,
                    util::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        cache_insertion_time, _2),
                    &performance_counters::locality_counter_discoverer, ""},
                {"/agas/time/cache/update_entry",
                    performance_counters::counter_monotonically_increasing,
                    "returns the overall time spent executing of the "
                    "update_entry API function of the AGAS cache",
                    HPX_PERFORMANCE_COUNTER_V1,
                    util::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        cache_update_entry_time, _2),
                    &performance_counters::locality_counter_discoverer, "ns"},
                {"/agas/time/cache/erase_entry",
                    performance_counters::counter_monotonically_increasing,
                    "returns the overall time spent executing of the "
                    "erase_entry API function of the AGAS cache",
                    HPX_PERFORMANCE_COUNTER_V1,
                    util::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        cache_erase_entry_time, _2),
                    &performance_counters::locality_counter_discoverer, ""},
            };

        performance_counters::install_counter_types(
            counter_types, sizeof(counter_types) / sizeof(counter_types[0]));

        // install counters for services
        agas::primary_namespace_register_counter_types();
        if (client.is_bootstrap())
        {
            agas::component_namespace_register_counter_types();
            agas::locality_namespace_register_counter_types();
        }
        agas::symbol_namespace_register_counter_types();
    }
}}    // namespace hpx::performance_counters
