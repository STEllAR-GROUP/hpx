// Copyright (C) 2001-2003 William E. Kempf
// Copyright (C) 2006 Roland Schwarz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

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

#ifndef JT28092007_BOOST_TSS_IMPL_PTHREAD
#define JT28092007_BOOST_TSS_IMPL_PTHREAD

#include <memory>
#include <vector>
#include <pthread.h>

namespace hpx { namespace util { namespace logging { namespace detail {

typedef std::vector<void*> tss_slots;

inline pthread_key_t & tss_data_native_key() {
    static pthread_key_t key;
    return key;
}

inline tss_slots* get_slots()
{
    tss_slots* slots = nullptr;
    slots = static_cast<tss_slots*>( pthread_getspecific(tss_data_native_key()));
    if (slots == nullptr)
    {
        std::auto_ptr<tss_slots> temp( new_object_ensure_delete<tss_slots>() );
        // pre-allocate a few elems, so that we'll be fast
        temp->resize(HPX_LOG_TSS_SLOTS_SIZE);

        pthread_setspecific(tss_data_native_key(), temp.get());
        slots = temp.release();
    }

    return slots;
}

extern "C" inline void cleanup_slots(void* ) {}

inline void init_tss_data()
{
    pthread_key_create(&tss_data_native_key(), &cleanup_slots);

    // make sure the static gets created
    object_deleter();
}



inline unsigned int slot_idx() {
    typedef hpx::util::logging::threading::mutex mutex;
    static mutex cs;
    static unsigned int idx = 0;

    mutex::scoped_lock lk(cs);

    // note: if the Logging Lib is used with TLS,
    //       I'm guaranteed this will be called before main(),
    //       and that this will work
    if ( !idx)
        init_tss_data();

    ++idx;
    return idx;
}

inline tss::tss() : m_slot( slot_idx() )
{
}

inline tss::~tss()
{
}

inline void* tss::get() const
{
    tss_slots* slots = get_slots();

    if (!slots)
        return 0;

    if (m_slot >= slots->size())
        return 0;

    return (*slots)[m_slot];
}

inline void tss::set(void* value)
{
    tss_slots* slots = get_slots();

    if (m_slot >= slots->size())
    {
        slots->resize(m_slot + 1);
    }

    (*slots)[m_slot] = value;
}

}}}}


#endif
