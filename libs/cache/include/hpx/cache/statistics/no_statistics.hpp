//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_CACHE_NO_STATISTICS_NOV_20_2008_1229PM)
#define HPX_UTIL_CACHE_NO_STATISTICS_NOV_20_2008_1229PM

#include <hpx/config.hpp>

#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace cache { namespace statistics
{
    ///////////////////////////////////////////////////////////////////////////
    enum method
    {
        method_get_entry = 0,
        method_insert_entry = 1,
        method_update_entry = 2,
        method_erase_entry = 3
    };

    ///////////////////////////////////////////////////////////////////////////
    class no_statistics
    {
    public:
        /// \brief  The function \a got_hit will be called by a cache instance
        ///         whenever a entry got touched.
        void got_hit() {}

        /// \brief  The function \a got_miss will be called by a cache instance
        ///         whenever a requested entry has not been found in the cache.
        void got_miss() {}

        /// \brief  The function \a got_insertion will be called by a cache
        ///         instance whenever a new entry has been inserted.
        void got_insertion() {}

        /// \brief  The function \a got_eviction will be called by a cache
        ///         instance whenever an entry has been removed from the cache
        ///         because a new inserted entry let the cache grow beyond its
        ///         capacity.
        void got_eviction() {}

        /// \brief Reset all statistics
        void clear() {}

        /// Helper class to update timings and counts on function exit
        struct update_on_exit
        {
            update_on_exit(no_statistics const&, method) {}
        };

        /// The function \a get_get_entry_count returns the number of
        /// invocations of the get_entry() API function of the cache.
        std::int64_t get_get_entry_count(bool)
        {
            return 0;
        }

        /// The function \a get_insert_entry_count returns the number of
        /// invocations of the insert_entry() API function of the cache.
        std::int64_t get_insert_entry_count(bool)
        {
            return 0;
        }

        /// The function \a get_update_entry_count returns the number of
        /// invocations of the update_entry() API function of the cache.
        std::int64_t get_update_entry_count(bool)
        {
            return 0;
        }

        /// The function \a get_erase_entry_count returns the number of
        /// invocations of the erase() API function of the cache.
        std::int64_t get_erase_entry_count(bool)
        {
            return 0;
        }

        /// The function \a get_get_entry_time returns the overall time spent
        /// executing of the get_entry() API function of the cache.
        std::int64_t get_get_entry_time(bool)
        {
            return 0;
        }

        /// The function \a get_insert_entry_time returns the overall time
        /// spent executing of the insert_entry() API function of the cache.
        std::int64_t get_insert_entry_time(bool)
        {
            return 0;
        }

        /// The function \a get_update_entry_time returns the overall time
        /// spent executing of the update_entry() API function of the cache.
        std::int64_t get_update_entry_time(bool)
        {
            return 0;
        }

        /// The function \a get_erase_entry_time returns the overall time spent
        /// executing of the erase() API function of the cache.
        std::int64_t get_erase_entry_time(bool)
        {
            return 0;
        }
    };
}}}}

#endif

