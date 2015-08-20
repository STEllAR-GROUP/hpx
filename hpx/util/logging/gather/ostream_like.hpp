// ostream_like.hpp

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


#ifndef JT28092007_ostream_like_HPP_DEFINED
#define JT28092007_ostream_like_HPP_DEFINED

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>

#include <iostream>
#include <sstream>

namespace hpx { namespace util { namespace logging {

/** @page gather_the_message Gathering the message

A class that implements gathering the message needs 2 things:
- a function that will gather the data - called <tt>.out()</tt>
- define a function called <tt>.msg()</tt> that will return the gathered data
(once all data has been gathered).
- have a public type named "msg_type" - be it a class or a typedef
  - this contains what the gather_msg class returns, as non-reference,
  non-const (that is, msg_type != const msg_type,
  "msg_type&" is a not a reference-to-reference)


*/

namespace optimize {
    template<class char_type_ > struct cache_string_one_str ;
}

/**
    @brief Classes that implement gathering the message

    @copydoc gather_the_message

    Implementing a gather class is rather easy, here's a simple example:

    @code
    struct return_str {
        typedef std::basic_ostringstream<char_type> stream_type;

        stream_type & out() { return m_out; }
        std::basic_string<char_type> msg() { return m_out.str(); }
    private:
        stream_type m_out;
    };
    @endcode

    @sa gather::ostream_like
*/
namespace gather {

/**
    @brief In case your gather class returns anything else than a std::basic_ostream,
    that returned class @b must derive from this.

    This is needed when :
    - in your application, you could have messages logged before your logs are
      initialized
    - you want filters to work even "in advance" - that is,
    if a message was logged before your log was initialized,
      and when you initialize the log, its corresponding filter is turned on,
      that message will be logged.
      Otherwise, it will @b not be logged.
*/
struct out_base {
    operator const void*() const { return this; }
};


/**
    @brief Gathering the message: Allows you to write to a log using
    the cool "<<" operator.

    The <tt>.msg()</tt> function returns the gathered message.

    @copydoc gather_the_message

*/
namespace ostream_like {

/**
    @brief Allows you to write to a log using the cool "<<" operator.
    The @c .msg() returns the stream itself.

    Note that this is a very simple class - meant only as an example.

    @copydoc gather_the_message

    See also:
    - hpx::util::logging::gather
    - ostream_like

*/
template<class stream_type = std::basic_ostringstream<char_type> >
struct return_raw_stream { //-V690
    // what does the gather_msg class return?
    typedef stream_type msg_type;

    return_raw_stream() {}
    return_raw_stream(const return_raw_stream& other) : m_out( other.m_out.str() ) {}

    /**
    note: we return the whole stream - we don't return out().str() ,
    because the user might use a better ostream class,
    which could have better access to its buffer/internals
    */
    stream_type & msg() { return m_out; }
    stream_type & out() { return m_out; }
private:
    stream_type m_out;
};




/**
    @brief Allows you to write to a log using the cool "<<" operator.
    The .msg() returns a string - whatever you set as first template param.

    By default, it's @ref hpx::util::logging::optimize::cache_string_one_str
    "cache_string".

    @copydoc gather_the_message

    See also:
    - hpx::util::logging::gather
    - ostream_like


    @bug right now prepend_size and append_size are ignored;
    because we can also return a cache_string_several_str<>.
    When fixing, watch the find_gather class!
*/
template<
        class string = hpx::util::logging::optimize
            ::cache_string_one_str<hold_string_type> ,
        class stream_type = std::basic_ostringstream<char_type>
> struct return_str { //-V690

    // what does the gather_msg class return?
    typedef string msg_type;

    return_str() {}
    return_str(const return_str& other) : m_out(other.m_out.str()) {}

    stream_type & out() { return m_out; }
    /** @brief returns a string */
    string msg() { return string( m_out.str() ); }
private:
    stream_type m_out;
};



/**
    @brief Returns a tag holder

    See @ref hpx::util::logging::tag namespace
*/
template<class holder_type, class stream_type> struct return_tag_holder
    : out_base { //-V690
    // what does the gather_msg class return?
    typedef holder_type msg_type;

    return_tag_holder() {}
    return_tag_holder(const return_tag_holder& other) : m_out(other.m_out.str()),
        m_val(other.m_val) {}


    return_tag_holder & out() { return *this; }
    template<class tag_type> return_tag_holder& set_tag(const tag_type & tag) {
        m_val.set_tag(tag);
        return *this;
    }

    template<class val_type> stream_type & operator<<(const val_type& val)
    { m_out << val; return m_out; }

    /** @brief returns the holder */
    holder_type & msg() { m_val.set_string( m_out.str() ); return m_val; }
private:
    holder_type m_val;
    stream_type m_out;

};

}}}}}

#endif

