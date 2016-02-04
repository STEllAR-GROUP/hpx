// tss_ensure_proper_delete.hpp

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


#ifndef JT28092007_tss_ensure_proper_delete_HPP_DEFINED
#define JT28092007_tss_ensure_proper_delete_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>
#include <vector>
#include <stdlib.h>

namespace hpx { namespace util { namespace logging { namespace detail {

struct do_delete_base {
    virtual ~do_delete_base () {}
};

template<class type> struct do_delete : do_delete_base {
    do_delete(type * val) : m_val(val) {}
    ~do_delete() { delete m_val; }
    type * m_val;
};

#ifdef HPX_LOG_TEST_TSS
// just for testing
void on_end_delete_objects();
#endif

struct delete_array : std::vector< do_delete_base* > {
    typedef hpx::util::logging::threading::mutex mutex;
    typedef std::vector< do_delete_base* > vector_base;

    delete_array() {}
    ~delete_array () {
        for ( const_iterator b = begin(), e = end(); b != e; ++b)
            delete *b;

#ifdef HPX_LOG_TEST_TSS
        on_end_delete_objects();
#endif
    }
    void push_back(do_delete_base* p) {
        mutex::scoped_lock  lk(cs);
        vector_base::push_back(p);
    }
private:
    mutex cs;
};


inline delete_array & object_deleter() {
    static delete_array a ;
    return a;
}


template<class type> inline type * new_object_ensure_delete() {
    type * val = new type;
    delete_array & del = object_deleter();
    del.push_back( new do_delete<type>(val) );
    return val;
}

template<class type> inline type * new_object_ensure_delete(const type & init) {
    type * val = new type(init);
    delete_array & del = object_deleter();
    del.push_back( new do_delete<type>(val) );
    return val;
}

}}}}

#endif

