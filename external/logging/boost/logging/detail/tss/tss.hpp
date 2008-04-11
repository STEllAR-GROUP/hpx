// tss.hpp

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


#ifndef JT28092007_tss_HPP_DEFINED
#define JT28092007_tss_HPP_DEFINED

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#if !defined(BOOST_LOG_NO_TSS)

#include <boost/logging/detail/fwd.hpp>

#if defined(BOOST_LOG_TSS_USE_INTERNAL)
// use internal implementation
#include <boost/logging/detail/tss/tss_impl.hpp>
#define BOOST_LOG_TSS_DEFAULT_CLASS = ::boost::logging::thread_specific_ptr

#elif defined(BOOST_LOG_TSS_USE_BOOST)
// use the boost implementation
#include <boost/thread/tss.hpp>
#define BOOST_LOG_TSS_DEFAULT_CLASS = ::boost::thread_specific_ptr

#else

// in case the user specified a custom class, maybe he specified its name as well
#define BOOST_LOG_TSS_DEFAULT_CLASS BOOST_LOG_TSS_USE_CUSTOM
#endif

namespace boost { namespace logging {


template<class type, template<typename> class thread_specific_ptr_type BOOST_LOG_TSS_DEFAULT_CLASS > struct tss_value {
    tss_value() {}

    type * get() const {
        type * result = m_value.get();
        if ( !result) {
#if defined(BOOST_LOG_TSS_USE_INTERNAL)
            result = detail::new_object_ensure_delete<type>();
#else
            result = new type;
#endif
            m_value.reset( result );
        }
        return result;
    }

    type* operator->() const { return get(); }
    type& operator*() const { return *get(); }
private:
    mutable thread_specific_ptr_type<type> m_value;
};




template<class type, template<typename> class thread_specific_ptr_type BOOST_LOG_TSS_DEFAULT_CLASS > struct tss_value_with_default {
    tss_value_with_default(const type & default_ ) : m_default( default_) {}

    type * get() const {
        type * result = m_value.get();
        if ( !result) {
#if defined(BOOST_LOG_TSS_USE_INTERNAL)
            result = detail::new_object_ensure_delete<type>(m_default) ;
#else
            result = new type(m_default);
#endif
            m_value.reset( result );
        }
        return result;
    }

    type* operator->() const { return get(); }
    type& operator*() const { return *get(); }
private:
    mutable thread_specific_ptr_type<type> m_value;
    // the default value - to assign each time a new value is created
    type m_default;
};


}}

#endif // !BOOST_LOG_NO_TSS

#endif

