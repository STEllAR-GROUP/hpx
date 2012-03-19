//  Copyright (c) 2012 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_PAPI_THREAD_MAPPER_FEB_17_2012_0109PM)
#define HPX_UTIL_PAPI_THREAD_MAPPER_FEB_17_2012_0109PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <map>
#include <vector>
#include <string>
#include <cstring>

#include <boost/thread.hpp>
#include <boost/noncopyable.hpp>
#include <boost/cstdint.hpp>
#include <boost/function.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // enumerates active OS threads and maintains their metadata
    class HPX_EXPORT thread_mapper : boost::noncopyable
    {
    public:
        // type for callback function invoked when thread is unregistered
        typedef boost::function1<bool, boost::uint32_t> callback_type;

    private:
        // null callback
        static bool null_cb(boost::uint32_t);

        // internal thread ID
        struct thread_id
        {
            char const *label_;         // label of thread group
            boost::uint32_t instance_;  // number of thread instance within group

            thread_id(): label_(0), instance_(-1) { }
            thread_id(char const *l, boost::uint32_t i):
                label_(l), instance_(i) { }
        };

        // thread-specific data
        struct thread_data: public thread_id
        {
            // associated thread ID that can be passed to PAPI_attach;
            // typically an ID of a kernel thread
            unsigned long tid_;
            // callback function invoked when unregistering a thread
            callback_type cleanup_;

            thread_data(): tid_(invalid_tid), cleanup_(boost::ref(null_cb)) { }
            thread_data(char const *gr, boost::uint32_t inst, unsigned long tid):
                thread_id(gr, inst), tid_(tid), cleanup_(boost::ref(null_cb)) { }
        };

        // comparator lexicographically ordering thread labels
        // (uniqueness of pointers to labels cannot be guaranteed)
        struct label_cmp
        {
            bool operator()(char const *s1, char const *s2) const
            {
                return strcmp(s1, s2) < 0;
            }
        };

        // thread_id comparator
        struct id_cmp
        {
            bool operator()(thread_id const& d1, thread_id const& d2) const
            {
                return (d1.instance_ < d2.instance_) ||
                       ((d1.instance_ == d2.instance_) &&
                        (strcmp(d1.label_, d2.label_) < 0));
            }
        };

        typedef hpx::lcos::local::spinlock mutex_type;
        typedef std::map<boost::thread::id, boost::uint32_t> thread_map_type;
        typedef std::map<char const *, boost::uint32_t, label_cmp> label_count_type;
        typedef std::map<thread_id, boost::uint32_t, id_cmp> label_map_type;

        // main lock
        mutable mutex_type mtx_;
        // mapping from boost thread IDs to small integer indices
        thread_map_type thread_map_;
        // thread counts for each thread category
        label_count_type label_count_;
        // label to thread lookup
        label_map_type label_map_;
        // table of thread specific data
        std::vector<thread_data> thread_info_;

    protected:
        // retrieve low level ID of caller thread (system dependent)
        unsigned long get_papi_thread_id();

        // unmap thread being unregistered
        bool unmap_thread(thread_map_type::iterator&);

    public:
        // erroneous thread index
        static boost::uint32_t invalid_index;
        // erroneous low-level thread ID
        static unsigned long invalid_tid;

        thread_mapper() { }
        ~thread_mapper();

        // registers invoking OS thread and assigns it a unique index
        boost::uint32_t register_thread(char const *label = "unspecified-thread");

        // unregisters the calling OS thread
        bool unregister_thread();

        // register callback function for a thread, invoked when unregistering
        // that thread
        bool register_callback(boost::uint32_t tix, callback_type);

        // cancel callback
        bool revoke_callback(boost::uint32_t tix);

        // returns low level thread id
        unsigned long get_thread_id(boost::uint32_t tix) const;

        // returns the group label of thread tix
        char const *get_thread_label(boost::uint32_t tix) const;

        // returns the instance number of thread tix
        boost::uint32_t get_thread_instance(boost::uint32_t tix) const;

        // returns unique index based on group label and instance number
        boost::uint32_t get_thread_index(char const *grp,
                                       boost::uint32_t inst) const;

        // returns the number of threads registered so far
        boost::uint32_t get_thread_count() const;

        // retrieve all registered thread labels
        void get_registered_labels(std::vector<char const *>&) const;
    };
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
