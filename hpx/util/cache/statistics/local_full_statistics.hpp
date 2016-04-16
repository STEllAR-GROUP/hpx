//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_CACHE_LOCAL_FULL_STATISTICS_AUG_11_2014_1029AM)
#define HPX_UTIL_CACHE_LOCAL_FULL_STATISTICS_AUG_11_2014_1029AM

#include <hpx/config.hpp>
#include <hpx/util/cache/statistics/local_statistics.hpp>

#if defined(__bgq__)
#include <hwi/include/bqc/A2_inlines.h>
#else
#include <boost/chrono/chrono.hpp>
#include <boost/chrono/process_cpu_clocks.hpp>
#endif
#include <boost/cstdint.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace cache { namespace statistics
{
    ///////////////////////////////////////////////////////////////////////////
    class local_full_statistics : public local_statistics
    {
    private:
        boost::int64_t get_and_reset_value(boost::int64_t& value, bool reset)
        {
            boost::int64_t result = value;
            if (reset) value = 0;
            return result;
        }

        struct api_counter_data
        {
            api_counter_data()
              : count_(0), time_(0)
            {}

            boost::int64_t count_;
            boost::int64_t time_;
        };

    public:
        /// Helper class to update timings and counts on function exit
        struct update_on_exit
        {
        private:
            static api_counter_data& get_api_counter_data(
                local_full_statistics& stat, method m)
            {
                switch(m) {
                case method_get_entry:
                default:
                    break;

                case method_insert_entry:
                    return stat.insert_entry_;

                case method_update_entry:
                    return stat.update_entry_;

                case method_erase_entry:
                    return stat.erase_entry_;
                }

                return stat.get_entry_;
            }

            static boost::uint64_t now()
            {
#if defined(__bgq__)
                return GetTimeBase();
#else
                boost::chrono::nanoseconds ns =
                    boost::chrono::steady_clock::now().time_since_epoch();
                return static_cast<boost::uint64_t>(ns.count());
#endif
            }

        public:
            update_on_exit(local_full_statistics& stat, method m)
              : started_at_(now()),
                data_(get_api_counter_data(stat, m))
            {
            }

            ~update_on_exit()
            {
                data_.time_ += (now() - started_at_);
                ++data_.count_;
            }

            boost::int64_t started_at_;
            api_counter_data& data_;
        };

        /// The function \a get_get_entry_count returns the number of
        /// invocations of the get_entry() API function of the cache.
        boost::int64_t get_get_entry_count(bool reset)
        {
            return get_and_reset_value(get_entry_.count_, reset);
        }

        /// The function \a get_insert_entry_count returns the number of
        /// invocations of the insert_entry() API function of the cache.
        boost::int64_t get_insert_entry_count(bool reset)
        {
            return get_and_reset_value(insert_entry_.count_, reset);
        }

        /// The function \a get_update_entry_count returns the number of
        /// invocations of the update_entry() API function of the cache.
        boost::int64_t get_update_entry_count(bool reset)
        {
            return get_and_reset_value(update_entry_.count_, reset);
        }

        /// The function \a get_erase_entry_count returns the number of
        /// invocations of the erase() API function of the cache.
        boost::int64_t get_erase_entry_count(bool reset)
        {
            return get_and_reset_value(erase_entry_.count_, reset);
        }

        /// The function \a get_get_entry_time returns the overall time spent
        /// executing of the get_entry() API function of the cache.
        boost::int64_t get_get_entry_time(bool reset)
        {
            return get_and_reset_value(get_entry_.time_, reset);
        }

        /// The function \a get_insert_entry_time returns the overall time
        /// spent executing of the insert_entry() API function of the cache.
        boost::int64_t get_insert_entry_time(bool reset)
        {
            return get_and_reset_value(insert_entry_.time_, reset);
        }

        /// The function \a get_update_entry_time returns the overall time
        /// spent executing of the update_entry() API function of the cache.
        boost::int64_t get_update_entry_time(bool reset)
        {
            return get_and_reset_value(update_entry_.time_, reset);
        }

        /// The function \a get_erase_entry_time returns the overall time spent
        /// executing of the erase() API function of the cache.
        boost::int64_t get_erase_entry_time(bool reset)
        {
            return get_and_reset_value(erase_entry_.time_, reset);
        }

    private:
        friend struct update_on_exit;

        api_counter_data get_entry_;
        api_counter_data insert_entry_;
        api_counter_data update_entry_;
        api_counter_data erase_entry_;
    };
}}}}

#endif

