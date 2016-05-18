//  Copyright (c) 2012 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_THREAD_MAPPER_HPP
#define HPX_UTIL_THREAD_MAPPER_HPP

#include <hpx/config.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/function.hpp>

#include <boost/bimap.hpp>
#include <boost/thread/thread.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // enumerates active OS threads and maintains their metadata
    class HPX_EXPORT thread_mapper
    {
        HPX_NON_COPYABLE(thread_mapper);

    public:
        // type for callback function invoked when thread is unregistered
        typedef util::function_nonser<bool(std::uint32_t)> callback_type;

        // erroneous thread index
        static std::uint32_t const invalid_index;
        // erroneous low-level thread ID
        static long const invalid_tid;
        // empty label for failed lookups
        static const std::string invalid_label;

    private:
        // null callback
        static bool null_cb(std::uint32_t);

        // thread-specific data
        struct thread_data
        {
            // associated system thread ID that can be passed to PAPI_attach;
            // typically an ID of a kernel thread
            long int tid_;
            // callback function invoked when unregistering a thread
            callback_type cleanup_;

            thread_data(long int tid = invalid_tid):
                tid_(tid), cleanup_(&thread_mapper::null_cb) { }
        };

        typedef hpx::lcos::local::spinlock mutex_type;
        typedef std::map<boost::thread::id, std::uint32_t> thread_map_type;
        typedef boost::bimap<std::string, std::uint32_t> label_map_type;

        // main lock
        mutable mutex_type mtx_;
        // mapping from boost thread IDs to thread indices
        thread_map_type thread_map_;
        // mapping between HPX thread labels and thread indices
        label_map_type label_map_;
        // table of thread specific data
        std::vector<thread_data> thread_info_;

    protected:
        // retrieve low level ID of caller thread (system dependent)
        long int get_system_thread_id();

        // unmap thread being unregistered
        bool unmap_thread(thread_map_type::iterator&);

    public:
        thread_mapper() { }
        ~thread_mapper();

        // registers invoking OS thread with a unique label
        std::uint32_t register_thread(char const *label);

        // unregisters the calling OS thread
        bool unregister_thread();

        // register callback function for a thread, invoked when unregistering
        // that thread
        bool register_callback(std::uint32_t tix, callback_type const&);

        // cancel callback
        bool revoke_callback(std::uint32_t tix);

        // returns low level thread id
        long int get_thread_id(std::uint32_t tix) const;

        // returns the label of registered thread tix
        std::string const& get_thread_label(std::uint32_t tix) const;

        // returns unique index based on registered thread label
        std::uint32_t get_thread_index(std::string const&) const;

        // returns the number of threads registered so far
        std::uint32_t get_thread_count() const;
    };
}}

#include <hpx/config/warnings_suffix.hpp>

#endif /*HPX_UTIL_THREAD_MAPPER_HPP*/
