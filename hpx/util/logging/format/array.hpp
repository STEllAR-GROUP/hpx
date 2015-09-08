// array_holder.hpp

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

#ifndef JT28092007_array_holder_HPP_DEFINED
#define JT28092007_array_holder_HPP_DEFINED

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace hpx { namespace util { namespace logging {

    ///////////////////////////////////////////////////////////////////////////
    // array holder class
    namespace array {

    /**
        Holds an array of manipulators (formatters or destinations). It owns them,
        holding them internally as smart pointers
        Each function call is locked.

        The base_type must implement operator==

        When you call get_ptr() or del(), the type you provide, must implement
        operator==(const type& , const base_type&)
    */
    template <class base_type, class mutex = hpx::util::logging::threading::mutex >
    class shared_ptr_holder {
        typedef typename mutex::scoped_lock scoped_lock;
    public:
        typedef base_type value_type;
        typedef boost::shared_ptr<value_type> ptr_type;
        typedef std::vector<ptr_type> array_type;

        template<class derived> base_type* append(derived val) {
            // FIXME try/catch
            derived * copy = new derived(val);
            scoped_lock lk(m_cs);
            m_array.push_back( ptr_type(copy));
            return copy;
        }

        template<class derived> base_type * get_ptr(derived val) const {
            scoped_lock lk(m_cs);
            for ( typename array_type::const_iterator b = m_array.begin(),
                e = m_array.end(); b != e; ++b)
                if ( val == (*(b->get())) )
                    return b->get();

            // not found
            return 0;
        }

        template<class derived> void del(derived val) {
            base_type * p = get_ptr(val);
            del(p);
        }

        void del(base_type * p) {
            scoped_lock lk(m_cs);
            for ( typename array_type::iterator b = m_array.begin(),
                e = m_array.end(); b != e; ++b)
                if ( b->get() == p) {
                    m_array.erase(b);
                    return ;
                }
        }

    private:
        mutable mutex m_cs;
        array_type m_array;
    };

    } // namespace array

}}}

#endif

