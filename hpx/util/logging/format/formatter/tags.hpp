// formatter_tags.hpp

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


#ifndef JT28092007_formatter_tags_HPP_DEFINED
#define JT28092007_formatter_tags_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>
#include <hpx/util/logging/detail/manipulator.hpp>
#include <hpx/util/logging/format/formatter/time.hpp>
#include <sstream>
#include <hpx/util/logging/format.hpp>

namespace hpx { namespace util { namespace logging { namespace formatter {



/**
    @brief Specifies that a formatter class handles a certain tag class

    @param type The class itself
    @param tag_type The tag class it handles
*/
template<class type, class tag_type> struct uses_tag {
    template<class tag_holder_type> void operator()(tag_holder_type & str) const {
        typedef typename tag_holder_type::string_type string_type;
        // automatic conversion - tag holder provides this
        const tag_type & tag = str;

        const type & self = static_cast<const type&>(*this);
        self.write_tag(str, tag);
    }
};

/** @brief Classes that process the @ref hpx::util::logging::tag "tags"
coming with the library

See @ref hpx::util::logging::tag "how to use tags".
*/
namespace tag {

/** @brief Dumps file/line information (corresponds to
hpx::util::logging::tag::file_line tag class)

See @ref hpx::util::logging::tag "how to use tags".
*/
template<class convert = do_convert_format::prepend>
struct file_line_t : is_generic, uses_tag< file_line_t<convert>,
    ::hpx::util::logging::tag::file_line >, hpx::util::logging::op_equal::always_equal  {
    typedef convert convert_type;
    template<class msg_type, class tag_type> void write_tag(msg_type & str,
        const tag_type & tag) const {
        convert::write( tag.val, str);
    }
};



/** @brief Dumps function name information
(corresponds to hpx::util::logging::tag::function tag class)

See @ref hpx::util::logging::tag "how to use tags".
*/
template<class convert = do_convert_format::prepend> struct function_t : is_generic,
uses_tag< function_t<convert>, ::hpx::util::logging::tag::function >,
hpx::util::logging::op_equal::always_equal  {
    typedef convert convert_type;
    template<class msg_type, class tag_type> void write_tag(msg_type & str,
        const tag_type & tag) const {
        convert::write( tag.val, str);
    }
};



/** @brief Dumps level (corresponds to hpx::util::logging::tag::level tag class)

See @ref hpx::util::logging::tag "how to use tags".
*/
template<class convert = do_convert_format::prepend>
struct level_t : is_generic, uses_tag< level_t<convert>,
    ::hpx::util::logging::tag::level >, hpx::util::logging::op_equal::always_equal  {
    typedef convert convert_type;
    template<class msg_type, class tag_type> void write_tag(msg_type & str,
        const tag_type & tag) const {
        typedef typename hpx::util::logging::dump_level<>::type dump_type;
        convert::write( dump_type::dump(tag.val) , str);
    }
};



/** @brief Dumps current time information
(corresponds to hpx::util::logging::tag::time tag class)

Similar to hpx::util::logging::formatter::time_t class - only that this one uses tags.

See @ref hpx::util::logging::tag "how to use tags".
*/
template<class convert = do_convert_format::prepend> struct time_t
    : is_generic, uses_tag< time_t<convert>, ::hpx::util::logging::tag::time > {
    typedef convert convert_type;
    typedef hpx::util::logging::formatter::time_t<convert> time_write_type;
    time_write_type m_writer;

    time_t(const hold_string_type & format) : m_writer(format) {}

    template<class msg_type, class tag_type>
    void write_tag(msg_type & str, const tag_type & tag) const {
        m_writer.write_time(str, tag.val);
    }

    bool operator==(const time_t & other) const {
        return m_writer == other.m_writer ;
    }

private:
};



/** @brief Dumps module information
(corresponds to hpx::util::logging::tag::module tag class)

See @ref hpx::util::logging::tag "how to use tags".
*/
template<class convert = do_convert_format::prepend> struct module_t
    : is_generic, uses_tag< module_t<convert>, ::hpx::util::logging::tag::module >,
    hpx::util::logging::op_equal::always_equal  {
    typedef convert convert_type;
    template<class msg_type, class tag_type> void write_tag(msg_type & str,
        const tag_type & tag) const {
        convert::write( tag.val, str);
    }
};



/** @brief Dumps thread id information
(corresponds to hpx::util::logging::tag::thread_id tag class)

See @ref hpx::util::logging::tag "how to use tags".
*/
template<class stream_type = ::std::basic_ostringstream<char_type> ,
class convert = do_convert_format::prepend> struct thread_id_t
        : is_generic, uses_tag< thread_id_t< ::std::basic_ostringstream<char_type>,
    convert>, ::hpx::util::logging::tag::thread_id >,
    hpx::util::logging::op_equal::always_equal  {

    typedef convert convert_type;
    template<class msg_type, class tag_type> void write_tag(msg_type & str,
        const tag_type & tag) const {
        stream_type out;
        out << tag.val ;
        convert::write( out.str(), str);
    }
};







/** @brief file_line_t with default values. See file_line_t

@copydoc file_line_t
*/
typedef file_line_t<> file_line;

/** @brief function_t with default values. See function_t

@copydoc function_t
*/
typedef function_t<> function;

/** @brief level_t with default values. See level_t

@copydoc level_t
*/
typedef level_t<> level;

/** @brief time_t with default values. See time_t

@copydoc time_t
*/
typedef time_t<> time;

/** @brief module_t with default values. See module_t

@copydoc module_t
*/
typedef module_t<> module;

/** @brief thread_id_t with default values. See thread_id_t

@copydoc thread_id_t
*/
typedef thread_id_t<> thread_id;


}}}}}

#endif

