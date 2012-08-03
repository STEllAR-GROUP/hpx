//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2011-2012 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/components/papi/server/papi.hpp>
#include <hpx/components/papi/util/papi.hpp>
#include <hpx/exception.hpp>

#include <functional>

#include <boost/version.hpp>
#include <boost/format.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace papi_ns = hpx::performance_counters::papi;

typedef hpx::components::managed_component<
    papi_ns::server::papi_counter
> papi_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    papi_counter_type, papi_counter, "base_performance_counter");
HPX_DEFINE_GET_COMPONENT_TYPE(papi_ns::server::papi_counter);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace papi { namespace server
{
#define NS_STR "hpx::performance_counters::papi::server::"

    ///////////////////////////////////////////////////////////////////////////
    // static members
    papi_counter_base::mutex_type papi_counter_base::base_mtx_;
    papi_counter_base::ttable_type papi_counter_base::thread_state_;

    using hpx::performance_counters::papi::util::papi_call;

    ///////////////////////////////////////////////////////////////////////////
    // methods
    thread_counters::thread_counters(boost::uint32_t tix):
        thread_index_(tix), evset_(PAPI_NULL)
    {
        char const *locstr =
            "hpx::performance_counters::papi::server::thread_counters()";
        hpx::util::thread_mapper& tm =
            hpx::get_runtime().get_thread_mapper();
        papi_call(PAPI_create_eventset(&evset_),
            "could not create PAPI event set", locstr);
        papi_call(PAPI_assign_eventset_component(evset_, 0),
            "cannot assign component index to event set", locstr);
        unsigned long tid = tm.get_thread_id(tix);
        if (tid == tm.invalid_tid)
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                NS_STR "thread_counters::thread_counters()",
                "unable to retrieve correct OS thread ID for profiling "
                "(perhaps thread was not registered)");
        papi_call(PAPI_attach(evset_, tm.get_thread_id(tix)),
            "failed to attach thread to PAPI event set", locstr);
        tm.register_callback(tix,
            std::bind1st(std::mem_fun(&thread_counters::terminate), this));
    }

    thread_counters::~thread_counters()
    {
        {
            mutex_type::scoped_lock m(mtx_);
            finalize();
        }
        // callback cancellation moved outside the mutex to avoid potential deadlock
        hpx::get_runtime().get_thread_mapper().revoke_callback(thread_index_);
    }

    bool thread_counters::add_event(papi_counter *cnt)
    {
        stop_all();
        if (PAPI_add_event(evset_, cnt->get_event()) != PAPI_OK)
            return false;
        counts_.push_back(cnt->get_value());
        cnttab_.push_back(cnt);
        cnt->update_index(counts_.size()-1);
        return start_all();
    }

    bool thread_counters::remove_event(papi_counter *cnt)
    {
        if (!stop_all() || PAPI_cleanup_eventset(evset_) != PAPI_OK)
            return false;

        // store the most recent value
        int rmix = cnt->get_counter_index();
        cnt->update_state(timestamp_, counts_[rmix], PAPI_COUNTER_STOPPED);
        // erase entries corresponding to removed event
        counts_.erase(counts_.begin()+rmix);
        cnttab_.erase(cnttab_.begin()+rmix);
        // For the lack of better strategy the remaining events are added in the
        // same order as before. This avoids reordering of remaining counter
        // values and at least some surprises on architectures with asymmetric
        // functionality of counting registers.
        unsigned i = 0;
        while (i < counts_.size())
        {
            papi_counter *c = cnttab_[i];
            if (PAPI_add_event(evset_, c->get_event()) != PAPI_OK)
            { // failed to add event
                c->update_state(timestamp_, counts_[c->get_counter_index()],
                                PAPI_COUNTER_SUSPENDED);
                counts_.erase(counts_.begin()+i);
                cnttab_.erase(cnttab_.begin()+i);
                continue;
            }
            // update indices of remaining counters
            c->update_index(i++);
        }
        return start_all();
    }

    bool thread_counters::read_value(papi_counter *cnt)
    {
        if (PAPI_accum(evset_, &counts_[0]) != PAPI_OK) return false;
        timestamp_ = hpx::util::high_resolution_clock::now();
        cnt->update_state(timestamp_, counts_[cnt->get_counter_index()]);
        return true;
    }

    bool thread_counters::terminate(boost::uint32_t tix)
    {
        mutex_type::scoped_lock m(mtx_);
        return finalize();
    }

    bool thread_counters::start_all()
    {
        if (counts_.empty()) return true; // nothing to count
        return PAPI_start(evset_) == PAPI_OK;
    }

    bool thread_counters::stop_all()
    {
        int stat;
        if (PAPI_state(evset_, &stat) != PAPI_OK) return false;

        if ((stat & PAPI_RUNNING) != 0)
        {
            std::vector<long long> tmp(counts_.size());
            if (PAPI_stop(evset_, &tmp[0]) != PAPI_OK) return false;
            // accumulate existing counts before modifying event set
            if (PAPI_accum(evset_, &counts_[0]) != PAPI_OK) return false;
            timestamp_ = hpx::util::high_resolution_clock::now();
        }
        return true;
    }

    bool thread_counters::finalize()
    {
        if (evset_ != PAPI_NULL)
        {
            if (stop_all())
            {
                for (boost::uint32_t i = 0; i < cnttab_.size(); ++i)
                { // retrieve the most recent values for the still active counters
                    papi_counter *c = cnttab_[i];
                    if (c->get_status() == PAPI_COUNTER_ACTIVE)
                        c->update_state(timestamp_, counts_[i], PAPI_COUNTER_TERMINATED);
                    else c->update_status(PAPI_COUNTER_TERMINATED);
                }
            }
            // release event set resources
            return PAPI_destroy_eventset(&evset_) == PAPI_OK;
        }
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_counters *papi_counter_base::get_thread_counters(boost::uint32_t tix)
    {
        mutex_type::scoped_lock m(base_mtx_);

        // create entry for the thread associated with the counter if it doesn't exist
        ttable_type::iterator it = thread_state_.find(tix);
        if (it == thread_state_.end())
            return thread_state_[tix] = new thread_counters(tix);
        else
            return it->second;
    }

    ///////////////////////////////////////////////////////////////////////////
    papi_counter::papi_counter(hpx::performance_counters::counter_info const& info):
        base_type_holder(info), event_(PAPI_NULL), index_(-1),
        value_(0), timestamp_(-1), status_(PAPI_COUNTER_STOPPED)
    {
        char const *locstr = NS_STR "papi_counter()";

        // extract path elements from counter name
        counter_path_elements cpe;
        get_counter_path_elements(info.fullname_, cpe);
        // convert event name to code and check availability
        papi_call(PAPI_event_name_to_code(const_cast<char *>(cpe.countername_.c_str()),
                                          const_cast<int *>(&event_)),
            cpe.countername_+" does not seem to be a valid event name", locstr);
        papi_call(PAPI_query_event(event_),
            "event "+cpe.countername_+" is not available on this platform", locstr);
        // find OS thread associated with the counter
        hpx::util::thread_mapper& tm = get_runtime().get_thread_mapper();
        boost::uint32_t tix = tm.get_thread_index(cpe.instancename_.c_str(),
                                                  cpe.instanceindex_);
        if (tix == hpx::util::thread_mapper::invalid_index)
        {
            boost::format fmt("could not find %s#%d");
            HPX_THROW_EXCEPTION(hpx::no_success, locstr,
                boost::str(fmt % cpe.instancename_ % cpe.instanceindex_));
        }
        // associate low level counters object
        counters_ = get_thread_counters(tix);
        if (!counters_)
        {
            boost::format fmt("failed to find low level counters for %s#%d");
            HPX_THROW_EXCEPTION(hpx::no_success, locstr,
                boost::str(fmt % cpe.instancename_ % cpe.instanceindex_));
        }
        // counting is not enabled here; it has to be started explicitly
    }

    void papi_counter::get_counter_value(hpx::performance_counters::counter_value& value)
    {
        thread_counters::mutex_type::scoped_lock m(counters_->get_lock());

        if (status_ == PAPI_COUNTER_ACTIVE)
            counters_->read_value(this);

        if (timestamp_ != -1) copy_value(value);
        else value.status_ = hpx::performance_counters::status_invalid_data;

        value.count_ = ++invocation_count_;
    }

    bool papi_counter::start()
    {
        thread_counters::mutex_type::scoped_lock m(counters_->get_lock());

        if (status_ == PAPI_COUNTER_ACTIVE) return true;
        if (counters_->add_event(this))
        {
            status_ = PAPI_COUNTER_ACTIVE;
            return true;
        }
        status_ = PAPI_COUNTER_SUSPENDED;
        return false;
    }

    bool papi_counter::stop()
    {
        thread_counters::mutex_type::scoped_lock m(counters_->get_lock());

        return stop_counter();
    }

    bool papi_counter::stop_counter()
    {
        if (status_ == PAPI_COUNTER_ACTIVE)
        {
            if (!counters_->remove_event(this)) return false;
        }
        return true;
    }

    void papi_counter::finalize()
    {
        thread_counters::mutex_type::scoped_lock m(counters_->get_lock());

        stop_counter();
        base_type_holder::finalize();
        base_type::finalize();
    }

}}}}
