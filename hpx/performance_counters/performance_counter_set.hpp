//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_PERFORMANCE_COUNTER_SET_DEC_19_2016_1055AM)
#define HPX_PERFORMANCE_COUNTERS_PERFORMANCE_COUNTER_SET_DEC_19_2016_1055AM

#include <hpx/config.hpp>
#include <hpx/error_code.hpp>
#include <hpx/lcos/dataflow.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/util/unwrap.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters
{
    // Make a collection of performance counters available as a set
    class HPX_EXPORT performance_counter_set
    {
        typedef lcos::local::spinlock mutex_type;

    public:
        /// Create an empty set of performance counters
        performance_counter_set(bool print_counters_locally = false)
          : print_counters_locally_(print_counters_locally)
        {}

        /// Create a set of performance counters from a name, possibly
        /// containing wild-card characters
        explicit performance_counter_set(std::string const& names,
            bool reset = false);
        explicit performance_counter_set(std::vector<std::string> const& names,
            bool reset = false);

        /// Add more performance counters to the set based on the given name,
        /// possibly containing wild-card characters
        void add_counters(std::string const& names, bool reset = false,
            error_code& ec = throws);
        void add_counters(std::vector<std::string> const& names,
            bool reset = false, error_code& ec = throws);

        /// Retrieve the counter infos for all counters in this set
        std::vector<counter_info> get_counter_infos() const;

        /// Retrieve the values for all counters in this set supporting
        /// this operation
        std::vector<hpx::future<counter_value> > get_counter_values(
            bool reset = false) const;
        std::vector<counter_value> get_counter_values(launch::sync_policy,
            bool reset = false, error_code& ec = throws) const;

        /// Retrieve the array-values for all counters in this set supporting
        /// this operation
        std::vector<hpx::future<counter_values_array> >
            get_counter_values_array(bool reset = false) const;
        std::vector<counter_values_array> get_counter_values_array(
            launch::sync_policy, bool reset = false, error_code& ec = throws) const;

        /// Reset all counters in this set
        std::vector<hpx::future<void> > reset();
        void reset (launch::sync_policy, error_code& ec = throws);

        /// Start all counters in this set
        std::vector<hpx::future<bool> > start();
        bool start(launch::sync_policy, error_code& ec = throws);

        /// Stop all counters in this set
        std::vector<hpx::future<bool> > stop();
        bool stop(launch::sync_policy, error_code& ec = throws);

        /// Release all references to counters in the set
        void release();

        /// Return the number of counters in this set
        std::size_t size() const;

        template <typename T>
        hpx::future<std::vector<T> > get_values(bool reset = false) const
        {
            return hpx::dataflow(
                &performance_counter_set::extract_values<T>,
                get_counter_values(reset));
        }
        template <typename T>
        std::vector<T> get_values(launch::sync_policy, bool reset = false,
            error_code& ec = throws) const
        {
            return get_values<T>(reset).get(ec);
        }

        std::size_t get_invocation_count() const;

    protected:
        bool find_counter(counter_info const& info, bool reset, error_code& ec);

        template <typename T>
        static std::vector<T> extract_values(
            std::vector<hpx::future<counter_value> > && values)
        {
            std::vector<T> results;
            results.reserve(values.size());
            for (hpx::future<counter_value> & f : values)
                results.push_back(f.get().get_value<T>());
            return results;
        }

    private:
        mutable mutex_type mtx_;

        std::vector<counter_info> infos_;     // counter instance names
        std::vector<naming::id_type> ids_;    // global ids of counter instances
        std::vector<std::uint8_t> reset_;     // != 0 if counter should be reset

        mutable std::uint64_t invocation_count_;
        bool print_counters_locally_;         // handle only local counters
    };
}}

#include <hpx/config/warnings_suffix.hpp>

#endif

