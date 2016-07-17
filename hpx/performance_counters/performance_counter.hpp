//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_PERFORMANCE_COUNTER_JAN_18_2013_0939AM)
#define HPX_PERFORMANCE_COUNTERS_PERFORMANCE_COUNTER_JAN_18_2013_0939AM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/stubs/performance_counter.hpp>

#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters
{
    ///////////////////////////////////////////////////////////////////////////
    struct HPX_EXPORT performance_counter
      : components::client_base<performance_counter, stubs::performance_counter>
    {
        typedef components::client_base<
            performance_counter, stubs::performance_counter
        > base_type;

        performance_counter() {}

        performance_counter(std::string const& name);

        performance_counter(std::string const& name, hpx::id_type const& locality);

        performance_counter(future<id_type> && id)
          : base_type(std::move(id))
        {}

        performance_counter(hpx::future<performance_counter> && c)
          : base_type(std::move(c))
        {}

        ///////////////////////////////////////////////////////////////////////
        future<counter_info> get_info() const;
        counter_info get_info_sync(error_code& ec = throws) const;

        future<counter_value> get_counter_value(bool reset = false);
        future<counter_value> get_counter_value() const;

        counter_value get_counter_value_sync(bool reset = false,
            error_code& ec = throws);
        counter_value get_counter_value_sync(error_code& ec = throws) const;

        future<counter_values_array> get_counter_values_array(bool reset = false);
        future<counter_values_array> get_counter_values_array() const;

        counter_values_array get_counter_values_array_sync(bool reset = false,
            error_code& ec = throws);
        counter_values_array get_counter_values_array_sync(
            error_code& ec = throws) const;

        ///////////////////////////////////////////////////////////////////////
        future<bool> start();
        bool start_sync(error_code& ec = throws);

        future<bool> stop();
        bool stop_sync(error_code& ec = throws);

        future<void> reset();
        void reset_sync(error_code& ec = throws);

        ///////////////////////////////////////////////////////////////////////
        future<std::string> get_name() const;
        std::string get_name_sync() const;

    private:
        template <typename T>
        static T extract_value(future<counter_value> && value, error_code& ec)
        {
            return value.get().get_value<T>(ec);
        }

    public:
        template <typename T>
        future<T> get_value(bool reset = false, error_code& ec = throws)
        {
            return get_counter_value(reset).then(
                util::bind(&performance_counter::extract_value<T>,
                    util::placeholders::_1, boost::ref(ec)));
        }
        template <typename T>
        T get_value_sync(bool reset = false, error_code& ec = throws)
        {
            return get_counter_value_sync(reset).get_value<T>(ec);
        }

        template <typename T>
        future<T> get_value(error_code& ec = throws) const
        {
            return get_counter_value().then(
                util::bind(&performance_counter::extract_value<T>,
                    util::placeholders::_1, boost::ref(ec)));
        }
        template <typename T>
        T get_value_sync(error_code& ec = throws) const
        {
            return get_counter_value_sync().get_value<T>(ec);
        }
    };

    /// Return all counters matching the given name (with optional wildcards).
    HPX_API_EXPORT std::vector<performance_counter> discover_counters(
        std::string const& name, error_code& ec = throws);
}}

#endif
