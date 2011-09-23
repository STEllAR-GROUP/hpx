//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PERFORMANCE_COUNTERS_SINE_SEP_20_2011_0112PM)
#define PERFORMANCE_COUNTERS_SINE_SEP_20_2011_0112PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/performance_counters/server/base_performance_counter.hpp>

namespace performance_counters { namespace sine { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    struct sine_counter
      : public hpx::performance_counters::server::base_performance_counter,
        public hpx::components::managed_component_base<sine_counter> 
    {
        typedef hpx::components::managed_component_base<sine_counter> base_type;

    public:
        typedef sine_counter type_holder;
        typedef hpx::performance_counters::server::base_performance_counter 
            base_type_holder;

        sine_counter() : current_value_(0) {}
        sine_counter(hpx::performance_counters::counter_info const& info);

        /// This function will be called in order to query the current value of 
        /// this performance counter
        void get_counter_value(hpx::performance_counters::counter_value& value);

        ///////////////////////////////////////////////////////////////////////
        // Disambiguate several functions defined in both base classes
        
        /// \brief finalize() will be called just before the instance gets 
        ///        destructed
        void finalize();

        using base_type::get_component_type;
        using base_type::set_component_type;

    protected:
        hpx::threads::thread_state_enum evaluate(hpx::threads::thread_state_ex_enum);
        void schedule_thread(std::size_t secs);

    private:
        typedef hpx::util::spinlock mutex_type;

        mutable mutex_type mtx_;
        double current_value_;

        hpx::threads::thread_id_type id_;     // id of currently scheduled thread
        boost::chrono::high_resolution_clock::time_point started_at_;
    };
}}}

#endif
