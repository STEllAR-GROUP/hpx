//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2011-2012 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PERFORMANCE_COUNTERS_PAPI_SERVER_PAPI_201111181426)
#define PERFORMANCE_COUNTERS_PAPI_SERVER_PAPI_201111181426

#include <vector>

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/interval_timer.hpp>
#include <hpx/performance_counters/server/base_performance_counter.hpp>

#include <boost/chrono/chrono.hpp>

#include <papi.h>

namespace hpx { namespace performance_counters { namespace papi { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    // Remarks:
    // As of papi-4.2.0 using bundled pfmlib many PAPI functions are
    // subtly broken when accessing performance counter interface in Linux
    // kernel.
    // To maintain the HPX notion of single quantity per performance counter
    // instance, the events need to be added (and removed if counter "stop"
    // function is to be reliably implemented) individually.

    // What has been tried so far:
    // 1) Multiple PAPI event sets with one event per set are not properly
    // coordinated by the library. Probably the only way this can work is when
    // at most one event set is active at a time, but that doesn't provide the
    // necessary functionality.
    // 2) Using a shared event set with PAPI_add_event works, but the removal
    // via PAPI_remove_event causes the remaining counter(s) to stop counting,
    // even though the event set status is returned as PAPI_RUNNING.
    // 3) Shared event set that is initialized from scratch by explicitly
    // adding all remaining events when one of the existing events needs to be
    // removed seems to work. However, the collected counts will be lost
    // during event set cleanup. Maintaning the correct total counts requires
    // careful tracking of values at both PAPI library and HPX levels.
    // 4) To simplify (3), in theory one could use PAPI_write to reinstate
    // the counter values before resuming counting. PAPI_write doesn't seem
    // to be working on all platforms and in addition some kernels may
    // explicitly disable it for accesses from user space.
    // 5) The alternative to using PAPI_read is to invoke PAPI_accum to
    // accumulate the momentary counts in user space counter array, with the
    // side effect of resetting the values of PAPI counters. This simplifies
    // the handling of counter values when adjustments to event set need to be
    // performed. Unfortunately, just like for (2) the simple removal of events
    // does not seem to work correctly, so sticking with (3) for event set
    // shrinking is the preferred approach.


    ///////////////////////////////////////////////////////////////////////////
    enum counter_state
    {
        // counter explicitly stopped or never activated
        PAPI_COUNTER_STOPPED = 0,
        // counter is currently counting events
        PAPI_COUNTER_ACTIVE,
        // counter activation requested, but was rejected by lower levels
        PAPI_COUNTER_SUSPENDED
    };

    struct papi_counter;

    ///////////////////////////////////////////////////////////////////////////
    struct papi_counter_base: boost::noncopyable
    {
        typedef hpx::util::spinlock mutex_type;
        typedef std::vector<papi_counter *> cnttab_type;

        ///////////////////////////////////////////////////////////////////////
        // shared state
        // mutex to protect shared state
        static mutex_type base_mtx_;
        // shared PAPI event set
        static int evset_;
        // associates pointers to active papi_counter instances with the
        // corresponding raw counter indexes
        static cnttab_type cnttab_;
        // counter values array accessible to PAPI
        static std::vector<long long> counts_;

        // internal PAPI counter index
        unsigned index_;

    protected:
        // ctor/dtor
        papi_counter_base();
        ~papi_counter_base();
        // register and activate new single event counter
        bool add_counter(papi_counter *);
        // remove counter from active set
        bool remove_counter(long long& last_count);
        // obtain current value for active counter i
        bool read_value(long long &);
        // enable multiplexing with specified interval
        bool enable_multiplexing(long ival);

    private:
        //// functions below require locking that is managed by the caller

        // add new PAPI event to the event set
        bool add_event(papi_counter *);
        // remove an event from the event set
        bool remove_event();
        // start all active counters
        void start_all();
        // safely stop all counters in event set and accumulate values in counts_
        void stop_all();
    };


    ///////////////////////////////////////////////////////////////////////////
    struct HPX_COMPONENT_EXPORT papi_counter:
        public hpx::performance_counters::server::base_performance_counter,
        public hpx::components::managed_component_base<papi_counter>,
        papi_counter_base
    {
        typedef hpx::components::managed_component_base<papi_counter> base_type;

    private:
        typedef hpx::util::spinlock mutex_type;

        // mutex for internal state updates
        mutable mutex_type mtx_;
        // PAPI event associated with the counter
        int event_;
        // most recent counter value
        long long value_;
        // and corresponding timestamp
        boost::int64_t timestamp_;
        // counting status
        counter_state status_;

    public:
        enum actions
        {
            papi_counter_set_event = 0,
            papi_counter_start,
            papi_counter_stop,
            papi_counter_enable_multiplexing
        };

        typedef hpx::components::managed_component<papi_counter> wrapping_type;
        typedef papi_counter type_holder;
        typedef hpx::performance_counters::server::base_performance_counter
            base_type_holder;

        papi_counter():
            event_(PAPI_NULL), value_(0), timestamp_(-1),
            status_(PAPI_COUNTER_STOPPED) { }
        papi_counter(hpx::performance_counters::counter_info const& info):
            base_type_holder(info), event_(PAPI_NULL), value_(0), timestamp_(-1),
            status_(PAPI_COUNTER_STOPPED) { }

        /// Specify event to count
        bool set_event(int event, bool activate);

        /// Start the counter
        bool start();

        /// Stop the counter
        bool stop();

        /// Get the current value of this performance counter
        void get_counter_value(hpx::performance_counters::counter_value& value);

        /// Enable multiplexing on the underlying event set
        bool enable_multiplexing(long ival);

        ///////////////////////////////////////////////////////////////////////
        // Disambiguate several functions defined in both base classes

        /// \brief finalize() will be called just before the instance gets
        ///        destructed
        void finalize();

        using base_type::get_component_type;
        using base_type::set_component_type;

        /// types for additional actions
        typedef hpx::actions::result_action2<
            papi_counter, bool,
            papi_counter_set_event, int, bool, &papi_counter::set_event
        > set_event_action;
        typedef hpx::actions::result_action0<
            papi_counter, bool, papi_counter_start, &papi_counter::start
        > start_action;
        typedef hpx::actions::result_action0<
            papi_counter, bool, papi_counter_stop, &papi_counter::stop
        > stop_action;
        typedef hpx::actions::result_action1<
            papi_counter, bool,
            papi_counter_enable_multiplexing, long,
            &papi_counter::enable_multiplexing
        > enable_multiplexing_action;

        /// PAPI event associated with this counter instance
        int get_event() const {return event_;}
        /// counter status
        counter_state status() const {return status_;}
        /// currently cached value
        long long get_value() const {return value_;}

    private:
        //// methods below operate on shared state and require locking in the caller

        /// update counter value and timestamp
        void update_state(long long cnt)
        {
            using namespace boost::chrono;

            timestamp_ = high_resolution_clock::now().time_since_epoch().count();
            value_ = cnt;
        }

        /// copy state to counter_value (needs external lock)
        void update_value(hpx::performance_counters::counter_value& val)
        {
            val.time_ = timestamp_;
            val.value_ = value_;
            val.status_ = hpx::performance_counters::status_new_data;
            val.scaling_ = 1;
            val.scale_inverse_ = false;
        }

        /// start counting events
        bool start_counter();
        /// stop counting events
        bool stop_counter();
    };

}}}}


HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::performance_counters::papi::server::papi_counter::set_event_action,
    papi_counter_set_event_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::performance_counters::papi::server::papi_counter::start_action,
    papi_counter_start_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::performance_counters::papi::server::papi_counter::stop_action,
    papi_counter_stop_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::performance_counters::papi::server::papi_counter::enable_multiplexing_action,
    papi_counter_enable_multiplexing_action);

#endif
