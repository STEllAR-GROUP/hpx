//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2011-2012 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/performance_counters/counters.hpp>

#include <boost/version.hpp>

#include <hpx/components/papi/server/papi.hpp>
#include <hpx/components/papi/util/papi.hpp>

///////////////////////////////////////////////////////////////////////////////

namespace papi_ns = hpx::performance_counters::papi;

typedef hpx::components::managed_component<
    papi_ns::server::papi_counter
> papi_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    papi_counter_type, papi_counter, "base_performance_counter");
HPX_DEFINE_GET_COMPONENT_TYPE(
    papi_ns::server::papi_counter);

HPX_REGISTER_ACTION_EX(
    papi_ns::server::papi_counter::set_event_action,
    papi_counter_set_event_action);

HPX_REGISTER_ACTION_EX(
    papi_ns::server::papi_counter::start_action,
    papi_counter_start_action);

HPX_REGISTER_ACTION_EX(
    papi_ns::server::papi_counter::stop_action,
    papi_counter_stop_action);

HPX_REGISTER_ACTION_EX(
    papi_ns::server::papi_counter::enable_multiplexing_action,
    papi_counter_enable_multiplexing_action);


#define NS_STR "hpx::performance_counters::papi::server::"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace papi { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    // static members
    papi_counter_base::mutex_type papi_counter_base::base_mtx_;
    int papi_counter_base::evset_ = PAPI_NULL;
    papi_counter_base::cnttab_type papi_counter_base::cnttab_;
    std::vector<long long> papi_counter_base::counts_;


    using hpx::performance_counters::papi::util::papi_call;

    ///////////////////////////////////////////////////////////////////////////
    // methods
    papi_counter_base::papi_counter_base(): index_((unsigned)-1)
    {
        mutex_type::scoped_lock m(base_mtx_); // is locking required here?

        if (evset_ != PAPI_NULL) return;
        papi_call(PAPI_create_eventset(&evset_),
                  "unable to create new PAPI event set",
                  NS_STR "papi_counter_base()");
    }

    papi_counter_base::~papi_counter_base()
    {
        mutex_type::scoped_lock m(base_mtx_);

        if (evset_ != PAPI_NULL && counts_.size() == 0)
            papi_call(PAPI_destroy_eventset(&evset_),
                      "failed to destroy PAPI event set",
                      NS_STR "~papi_counter_base()");
    }

    bool papi_counter_base::add_counter(papi_counter *c)
    {
        mutex_type::scoped_lock m(base_mtx_);

        stop_all();
        bool rc = add_event(c);
        start_all();
        return rc;
    }

    bool papi_counter_base::remove_counter(long long& last_val)
    {
        mutex_type::scoped_lock m(base_mtx_);

        stop_all();
        last_val = counts_[index_];
        bool rc = remove_event();
        // resume counting if any events left
        if (cnttab_.size() > 0) start_all();
        return rc;
    }

    bool papi_counter_base::read_value(long long& val)
    {
        mutex_type::scoped_lock m(base_mtx_);

        papi_call(PAPI_accum(evset_, &counts_[0]), "PAPI_accum failed",
                             NS_STR "papi_counter_base::read_value()");
        val = counts_[index_];
        return true;
    }

    bool papi_counter_base::enable_multiplexing(long ival)
    {
        // multiplexing interval is ignored for now
        mutex_type::scoped_lock m(base_mtx_);

        switch (PAPI_get_multiplex(evset_))
        {
        case PAPI_OK:     // not yet multiplexed
            papi_call(PAPI_assign_eventset_component(evset_, 0),
                      "failed to assign component index to current event set",
                      NS_STR "papi_counter_base::enable_multiplexing()");
            papi_call(PAPI_set_multiplex(evset_),
                      "failed to enable multiplexing for event set",
                      NS_STR "papi_counter_base::enable_multiplexing()");
            return true;
        case PAPI_EINVAL: // already multiplexed
            return true;
        default:          // error
            return false;
        }
    }

    bool papi_counter_base::add_event(papi_counter *c)
    {
        papi_call(PAPI_add_event(evset_, c->get_event()),
                  "could not add event",
                  NS_STR "papi_counter_base::add_event()");
        counts_.push_back(c->get_value());
        cnttab_.push_back(c);
        index_ = counts_.size()-1;
        return true;
    }

    bool papi_counter_base::remove_event()
    {
        papi_call(PAPI_cleanup_eventset(evset_),
                  "could not clean up event set",
                  NS_STR "papi_counter_base::remove_event()");

        // For the lack of better strategy the events are added in the same
        // order as before. This avoids reordering of remaining counter values
        // and at least some surprises on architectures with asymmetric
        // functionality of counting registers.
        for (unsigned i = 0; i < counts_.size(); ++i)
        {
            papi_counter *c = cnttab_[i];
            if (i != index_)
            {
                papi_call(PAPI_add_event(evset_, c->get_event()),
                          "cannot add event to event set",
                          NS_STR "papi_counter_base::remove_event()");
                // adjust indices of remaining counters
                if (i > index_) c->index_--;
            }
        }
        // erase entries corresponding to removed event
        counts_.erase(counts_.begin()+index_);
        cnttab_.erase(cnttab_.begin()+index_);
        return true;
    }

    void papi_counter_base::start_all()
    {
        papi_call(PAPI_start(evset_), "cannot start PAPI counters",
                  NS_STR "papi_counter_base::start_all()");
    }

    void papi_counter_base::stop_all()
    {
        int stat;
        papi_call(PAPI_state(evset_, &stat), "PAPI_state failed",
                  NS_STR "papi_counter_base::stop_all()");

        if ((stat & PAPI_RUNNING) != 0)
        {
            std::vector<long long> tmp(counts_.size());
            papi_call(PAPI_stop(evset_, &tmp[0]), "PAPI_stop failed",
                      NS_STR "papi_counter_base::stop_all()");
            // accumulate existing counts before modifying event set
            papi_call(PAPI_accum(evset_, &counts_[0]), "PAPI_stop failed",
                      NS_STR "papi_counter_base::stop_all()");
        }
    }


    ///////////////////////////////////////////////////////////////////////////
    void papi_counter::get_counter_value(hpx::performance_counters::counter_value& value)
    {
        mutex_type::scoped_lock m(mtx_);

        if (status_ == PAPI_COUNTER_ACTIVE)
        {
            long long cnt;
            if (read_value(cnt)) update_state(cnt);
        }

        if (timestamp_ != -1) update_value(value);
        else value.status_ = hpx::performance_counters::status_invalid_data;
    }

    bool papi_counter::enable_multiplexing(long ival)
    {
        return papi_counter_base::enable_multiplexing(ival);
    }

    bool papi_counter::set_event(int event, bool activate)
    {
        mutex_type::scoped_lock m(mtx_);

        // check if anything needs to be done at all
        if (event == event_ && activate == (status_ == PAPI_COUNTER_ACTIVE))
            return true;

        // remove currently counted event from active set
        if (!stop_counter()) return false;
        // invalidate cached values on event change
        if (event != event_)
        {
            value_ = 0;
            timestamp_ = -1;
            event_ = event;
        }
        // insert new event to the active set
        if (activate) return start_counter();
        else status_ = PAPI_COUNTER_STOPPED;

        return true;
    }

    bool papi_counter::start()
    {
        mutex_type::scoped_lock m(mtx_);

        return start_counter();
    }

    bool papi_counter::stop()
    {
        mutex_type::scoped_lock m(mtx_);

        return stop_counter();
    }

    void papi_counter::finalize()
    {
        mutex_type::scoped_lock m(mtx_); // probably not needed

        stop_counter();
        base_type_holder::finalize();
        base_type::finalize();
    }

    bool papi_counter::start_counter()
    {
        if (status_ == PAPI_COUNTER_ACTIVE) return true;
        if (add_counter(this))
        {
            status_ = PAPI_COUNTER_ACTIVE;
            return true;
        }
        else
        {
            status_ = PAPI_COUNTER_SUSPENDED;
            return false;
        }
    }

    bool papi_counter::stop_counter()
    {
        if (status_ == PAPI_COUNTER_ACTIVE)
        {
            long long cnt;
            if (!remove_counter(cnt))
            {   // FIXME: revisit when more permissive handling is implemented
                return false;
            }

            update_state(cnt);
            status_ = PAPI_COUNTER_STOPPED;
        }
        return true;
    }

}}}}
