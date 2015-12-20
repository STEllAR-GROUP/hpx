// ts_resource.hpp

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details


#ifndef JT28092007_ts_resource_HPP_DEFINED
#define JT28092007_ts_resource_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/util/assert.hpp>
#include <hpx/util/logging/detail/ts/ts.hpp>
#include <hpx/util/logging/detail/tss/tss.hpp>
#include <time.h>

namespace hpx { namespace util { namespace logging {


/**
    @brief Contains implementations of locker objects.
    Such a locker can lock data for read or write.

    Assume you have
    @code
    struct data {
        std::string name;
        std::string desc;
    };

    some_locker<data> lk;
    @endcode


    In order to access the locked data for write, you do:

    @code
    some_locker<data>::write obj(lk);
    obj->name = "coolio";
    obj->desc = "it works!";
    @endcode

    In order to access the locked data for read, you do:
    @code
    some_locker<data>::read obj(lk);
    std::cout << obj->name << "/" << obj->desc;
    @endcode

*/
namespace locker {

    /**
        the data to be locked. It's locked using default thread-safety

        @sa locker
    */
    template<class type, class mutex = hpx::util::logging::threading::mutex >
    struct ts_resource {
        typedef ts_resource<type,mutex> self_type;

        ts_resource(const type& val = type() ) : m_val(val) {}

        struct read;
        struct write;
        friend struct read;
        friend struct write;

        struct write {
            self_type & self ;
            typename mutex::scoped_lock locker;
            write(self_type & self_) : self(self_), locker(self_.m_cs) {}

            ~write() {}

            type & use() { return self.m_val ; }
            type* operator->() { return &use(); }
        };

        struct read {
            const self_type & self ;
            typename mutex::scoped_lock locker;
            read(const self_type & self_) : self(self_), locker(self_.m_cs) {}

            ~read() {}

            const type & use() { return self.m_val ; }
            const type* operator->() { return &use(); }
        };


    private:
        mutable mutex m_cs;
        type m_val;
    };


    template<class type> struct ts_resource_single_thread {
        typedef ts_resource_single_thread<type> self_type;

        ts_resource_single_thread(const type& val = type() ) : m_val(val) {}

        struct read;
        struct write;
        friend struct read;
        friend struct write;

        struct write {
            self_type & self ;
            write(self_type & self_) : self(self_) {}

            ~write() {}

            type & use() { return self.m_val ; }
            type* operator->() { return &use(); }
        };

        struct read {
            const self_type & self ;
            read(const self_type & self_) : self(self_) { }

            ~read() {}

            const type & use() { return self.m_val ; }
            const type* operator->() { return &use(); }
        };
    private:
        type m_val;
    };


#ifndef HPX_HAVE_LOG_NO_TSS

    /**
        Locks a resource, and uses TSS (Thread-specific storage).
        This holds the value, and each thread caches it.
        Once at a given period (like, every 5 seconds), when used,
        the latest object is copied.

        @sa locker
        @sa default_cache_millis how many secs to cache the data. By default, 5
    */
    template<class type, int default_cache_secs = 5,
    class mutex = hpx::util::logging::threading::mutex >
    struct tss_resource_with_cache {
        typedef tss_resource_with_cache<type, default_cache_secs, mutex> self_type;

    private:
        struct value_and_time {
            value_and_time()
                // so that the first time it's used, it'll be refreshed
                : val( type() ), time_(0) {
            }
            type val;
            ::time_t time_;
        };

    public:
        tss_resource_with_cache(const type& val = type() ,
            int cache_secs = default_cache_secs ) : m_val(val),
            m_cache_secs(cache_secs) {}

        struct read;
        struct write;
        friend struct read;
        friend struct write;

        struct write {
            type & val;
            typename mutex::scoped_lock locker;
            write(self_type & self) : val(self.m_val), locker(self.m_cs) {
            }
            ~write() {
            }

            type & use() { return val ; }
            type* operator->() { return &use(); }
        };

        struct read {
            const type *val ;
            read(const self_type & self) : val( &(self.m_cache->val) ) {
                ::time_t now = time(0);
                value_and_time & cached = *(self.m_cache);
                if ( cached.time_ + self.m_cache_secs < now) {
                    // cache has expired
                    typename mutex::scoped_lock lk(self.m_cs);
                    // see if another thread has updated the cache...
                    if ( cached.time_ + self.m_cache_secs < now) {
                        cached.val = self.m_val;
#ifndef HPX_LOG_TEST_TSS
                        cached.time_ = now;
#else
                        // for testing , make sure we always refresh at a fixed time
                        if ( cached.time_ != 0)
                            cached.time_ += self.m_cache_secs;
                        else
                            cached.time_ = now;
#endif
                    }
                }
            }
            ~read() {
            }

            const type & use() { return *val ; }
            const type* operator->() { return &use(); }
        };

    private:
        mutable tss_value<value_and_time> m_cache;
        type m_val;
        mutable mutex m_cs;
        int m_cache_secs;
    };




    /**
        Locks a resource, and uses TSS.

        The resource can be initialized once, at any time, no matter how many threads.
        Once the resource is initialized (basically, someone used resource::write),
        that is <b>the final value</b>

        All other threads will use and cached the initialized value.

        @sa locker
        @sa default_cache_millis how many secs to cache the data. By default, 5
    */
    template<class type, class mutex = hpx::util::logging::threading::mutex >
    struct tss_resource_once_init {
        typedef tss_resource_once_init<type, mutex> self_type;

    private:
        struct cached_value {
            cached_value( ) : val( type() ), is_cached(false) {}
            type val;
            bool is_cached;
        };

    public:
        tss_resource_once_init(const type& val = type() )
            : m_val(val), m_initialized(false) {}

        struct read;
        struct write;
        friend struct read;
        friend struct write;

        struct write {
            type & val;
            typename mutex::scoped_lock locker;
            write(self_type & self) : val(self.m_val), locker(self.m_cs) {
                self.m_initialized = true;
            }
            ~write() {
            }

            type & use() { return val ; }
            type* operator->() { return &use(); }
        };

        struct read {
            const type *val ;
            read(const self_type & self) {
                cached_value & cached = *(self.m_cache);
                val = &cached.val;
                if ( !cached.is_cached) {
                    typename mutex::scoped_lock lk(self.m_cs);
                    if ( self.m_initialized) {
                        cached.val = self.m_val;
                        cached.is_cached = true;
                    }
                }
            }
            ~read() {
            }

            const type & use() { return *val ; }
            const type* operator->() { return &use(); }
        };

    private:
        type m_val;
        mutable tss_value<cached_value> m_cache;
        mutable mutex m_cs;
        bool m_initialized;
    };


#endif

}}}}

#include <hpx/util/logging/detail/ts/resource_finder.hpp>

#endif

