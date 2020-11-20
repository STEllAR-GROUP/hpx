//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/performance_counters/server/base_performance_counter.hpp>
#include <hpx/runtime/components/server/component_base.hpp>
#include <hpx/runtime_local/interval_timer.hpp>
#include <hpx/synchronization/spinlock.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server {
    namespace detail {
        struct counter_type_from_statistic_base
        {
            virtual ~counter_type_from_statistic_base() {}

            virtual bool need_reset() const = 0;
            virtual double get_value() = 0;
            virtual void add_value(double value) = 0;
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // This counter exposes the average count of items processed during the
    // given base time interval. The counter relies on querying a steadily
    // growing counter value.
    template <typename Statistic>
    class statistics_counter
      : public base_performance_counter
      , public components::component_base<statistics_counter<Statistic>>
    {
        typedef components::component_base<statistics_counter<Statistic>>
            base_type;

        // avoid warnings about using this in member initializer list
        statistics_counter* this_()
        {
            return this;
        }

    public:
        typedef statistics_counter type_holder;
        typedef base_performance_counter base_type_holder;

        statistics_counter()
          : has_prev_value_(false)
          , parameter1_(0)
          , parameter2_(0)
          , reset_base_counter_(false)
        {
        }

        statistics_counter(counter_info const& info,
            std::string const& base_counter_name, std::size_t parameter1,
            std::size_t parameter2, bool reset_base_counter);

        /// Overloads from the base_counter base class.
        hpx::performance_counters::counter_value get_counter_value(
            bool reset = false) override;

        bool start() override;

        bool stop() override;

        void reset_counter_value() override;

        void on_terminate() {}

        /// \brief finalize() will be called just before the instance gets
        ///        destructed
        void finalize()
        {
            base_performance_counter::finalize();
            base_type::finalize();
        }

    protected:
        bool evaluate_base_counter(counter_value& value);
        bool evaluate();
        bool ensure_base_counter();

    private:
        typedef lcos::local::spinlock mutex_type;
        mutable mutex_type mtx_;

        hpx::util::interval_timer
            timer_;    ///< base time interval in milliseconds
        std::string
            base_counter_name_;    ///< name of base counter to be queried
        naming::id_type base_counter_id_;

        std::unique_ptr<detail::counter_type_from_statistic_base> value_;
        counter_value prev_value_;
        bool has_prev_value_;

        std::size_t parameter1_, parameter2_;
        bool reset_base_counter_;
    };
}}}    // namespace hpx::performance_counters::server
