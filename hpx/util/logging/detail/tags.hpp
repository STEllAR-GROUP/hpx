// tags.hpp

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


#ifndef JT28092007_tags_HPP_DEFINED
#define JT28092007_tags_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#ifndef JT28092007_format_fwd_HPP_DEFINED
#error Donot include directly. Please include <hpx/util/logging/format_fwd.hpp>
#endif


#include <hpx/util/logging/detail/fwd.hpp>

namespace hpx { namespace util { namespace logging {

namespace detail {
    struct void_1 {};
    struct void_2 {};
    struct void_3 {};
    struct void_4 {};
    struct void_5 {};
    struct void_6 {};
    struct void_7 {};
    struct void_8 {};
    struct void_9 {};
    struct void_10 {};

    template<class string_type> struct tag_holder_base {
        // assumes m_string is convertible to string
        operator const hold_string_type & () const { return m_string; }
    protected:
        string_type m_string;
    };
    template<> struct tag_holder_base<default_> {
        // it's for the default string
    protected:
        hold_string_type m_string;
    };
}

/**
@brief Allows you to use tags (extra information about the context of the logged message:
file/line, function name, thread id, etc.), and log this information as well

- @ref tag_need
- @ref tag_explained
    - @ref tag_classes
    - @ref tag_tag_holder
    - @ref tag_adding_tags
    - @ref tag_process_tags
    - @ref tag_see_example

@section tag_need Do you need tags?

First of all, note that the easiest way to log some extra context is to simply append it,
when definining your macro:

@code
#define LDBG_ HPX_LOG_USE_LOG_IF_LEVEL(g_l(), g_log_level(), debug ) << __FILE__ \
<< ":" << __LINE__ << " [dbg] "
@endcode

In the above case, you appended file & line, and the level of the logged message.
Usage is just the same:

@code
std::string hello = "hello", world = "world";
LAPP_ << hello << ", " << world;
@endcode

The output could look like:

@code
my_cool_sample:234 [dbg] hello, world
@endcode

I can see a few issues with the above
- The context formatting is fixed
  - You can't choose at runtime - what if I want to see the level first,
    then the file & line?
  - You can't choose at runtime if you want to ignore some of that context
    (to speed up the app in some cases, you might decide not to log the file & line)
  - You can't mix the context formatting with the rest of the formatting.
    For example, what if I want to log info like this : \n
    <tt>[idx] file_and_line [time] message [level]</tt> ?
  - You can't do extra formatting to any of the context.
    For example, when dumping file/line,
    what if you want to strip some information from the file
    (the file name could be pretty big). Or, you might want to @em normalize
    the file/line (like, fix it at 50 chars - by stripping or padding information)
- If you want to be efficient and do the logging on a
    @ref hpx::util::logging::writer::on_dedicated_thread "dedicated thread"
  - You can't use formatter::thread_id, because the thread_id is computed when
  being written (which when used on a dedicated thread, would always
    return the same value)
  - Logging the context takes time as well. For instance, <tt>" << __FILE__
    << ":" << __LINE__ << " [dbg] "</tt> , in the above case,
    takes time. It is much faster to only @em gather the context on the current thread,
    and then dump it on the dedicated thread. You can use tags for that.

If you're ok with the above issues, no need to delve into tags.
You can dump context like shown above, and be fine with it.

Otherwise, welcome to the world of @b tags!

@section tag_explained Tags - explained

@code
#include <hpx/util/logging/format_fwd.hpp>
@endcode



@subsection tag_classes Tag classes

Each single context information you need to hold, is identified by a tag class.
Tag classes are always found in the hpx::util::logging::tag namespace.

A tag class is deadly simple. Here are a few examples:

@code
struct file_line {
    file_line(const char * val = "") : val(val) {}
    const char * val;
};

struct time {
    time() : val( ::time(0) ) {}
    ::time_t val;
};
@endcode

They only allow holding the context, and making sure you can get to it
- when doing formatting. You can of course add your own tag clases.



@subsection tag_tag_holder Tag Holder - holding the tags

Now, you have to decide what tags you need. You will use templated class tag::holder:
- first param: the string class
- the next params: the tags you need

You will replace your old <tt>HPX_LOG_FORMAT_MSG(string_class)</tt> usage,
with tags. In case you don't have a HPX_LOG_FORMAT_MSG in your
application, the string_class is std::(w)string.

@code
// old
HPX_LOG_FORMAT_MSG( optimize::cache_string_one_str<> )

// new - use tags
//
//       In our case, time, file/line, function name
typedef tag::holder< optimize::cache_string_one_str<>,
tag::time, tag::file_line, tag::function> string;
HPX_LOG_FORMAT_MSG( string )
@endcode



@subsection tag_adding_tags Adding tags to your LOG macros

Some tag classes compute their context automatically (for instance, the tag::time class).
However, some tag classes need you to manually specify it,
in your LOG macros. This is the case for file/line, function, level, etc.

In your LOG macros, you need to append the tags like this:
- add <tt>.set_tag( <em>tag_class( tag_init_values)</em> ) </tt>
- if it's a tag class defined in the hpx::util::logging::tag namespace,
  you can use HPX_LOG_TAG(class_name) \n
  (which is just a shortcut for ::hpx::util::logging::tag::class_name)
- some tags that come with the lib have shortcuts :
  - HPX_LOG_TAG_LEVEL(lvl) - append the level
  - HPX_LOG_TAG_FILELINE - append file/line
  - HPX_LOG_TAG_FUNCTION - append function

Examples:

@code
// add file/line and function tags
#define L_ HPX_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) \
.set_tag(HPX_LOG_TAG_FILELINE) .set_tag(HPX_LOG_TAG_FUNCTION)

// add function and level
#define LDBG_ HPX_LOG_USE_LOG_IF_LEVEL(g_log_dbg(), g_log_level(), debug ) \
.set_tag(HPX_LOG_TAG_FUNCTION) .set_tag( HPX_LOG_TAG_LEVEL(debug) )

// add module information - you specify the module name whe using the L_ macro. Example:
// L_("chart") << "Initializing environment";
#define L_(module_name) HPX_LOG_USE_LOG_IF_FILTER(g_l(), \
g_log_filter()->is_enabled() ) .set_tag( HPX_LOG_TAG(module)(module_name) )

@endcode



@subsection tag_process_tags Processing the tags

Now, you're ready to process these tags
- where you're specifying your formatters and/or destinations,
add the tag formatters that will process your tags.
Example:

@code
#include <hpx/util/logging/format/formatter/tags.hpp>
...

g_l()->writer().add_formatter( formatter::idx() );
g_l()->writer().add_formatter( formatter::append_newline() );

// formatters to add the file/line and level
g_l()->writer().add_formatter( formatter::tag::file_line() );
g_l()->writer().add_formatter( formatter::tag::level() );

g_l()->writer().add_destination( destination::file("out.txt") );
g_l()->writer().add_destination( destination::cout() );
g_l()->writer().add_destination( destination::dbg_window() );
@endcode

Note that the library comes with default formatters for each tag class.
However, you can create your own formatter class, for a given tag class.

The formatters that come with the library, have the same name as the tag class itself,
only that they're in the @c formatter::tag namespace.

Examples:
- for tag::file_line, we have formatter::tag::file_line
- for tag::function, we have formatter::tag::function

When adding the formatters, don't forget to:

@code
#include <hpx/util/logging/format/formatter/tags.hpp>
@endcode


@subsection tag_see_example Example using Tags

@copydoc using_tags

@include using_tags.cpp

That's it, enjoy!

*/
namespace tag {

/** @brief Holds up to 10 @ref tag "tags".

@param string_ (required) The string class we use for holding logged messages.
By default, std::(w)string. What you used to specify using HPX_LOG_FORMAT_MSG.

@param param1 (optional) First tag
@param param2 (optional) Second tag
@param param3 (optional) Third tag
@param param4 (optional) Fourth tag
@param param5 (optional) Fifth tag
@param param6 (optional) Sixth tag
@param param7 (optional) Seventh tag
@param param8 (optional) Eigth tag
@param param9 (optional) Nineth tag
@param param10 (optional) Tenth tag
*/
template<
        class string_ = default_,
        class param1 = detail::void_1,
        class param2 = detail::void_2,
        class param3 = detail::void_3,
        class param4 = detail::void_4,
        class param5 = detail::void_5,
        class param6 = detail::void_6,
        class param7 = detail::void_7,
        class param8 = detail::void_8,
        class param9 = detail::void_9,
        class param10 = detail::void_10> struct holder
            : detail::tag_holder_base<string_> {
    typedef typename use_default<string_, hold_string_type>::type string_type;
    typedef detail::tag_holder_base<string_> tag_base_type;

    operator string_type & () { return tag_base_type::m_string; }
    operator const string_type & () const { return tag_base_type::m_string; }

    operator const param1& () const { return m_tag1; }
    operator const param2& () const { return m_tag2; }
    operator const param3& () const { return m_tag3; }
    operator const param4& () const { return m_tag4; }
    operator const param5& () const { return m_tag5; }
    operator const param6& () const { return m_tag6; }
    operator const param7& () const { return m_tag7; }
    operator const param8& () const { return m_tag8; }
    operator const param9& () const { return m_tag9; }
    operator const param10& () const { return m_tag10; }

    template<class tag_type> holder& set_tag(const tag_type & val) {
        set_tag_impl(val);
        return *this;
    }

    template<class tag_type> const tag_type & get_tag() const {
        return this->operator const tag_type&();
    }

    void set_string(const string_type & str) {
        tag_base_type::m_string = str;
    }

private:
    void set_tag_impl(const param1 & tag) {
        m_tag1 = tag;
    }
    void set_tag_impl(const param2 & tag) {
        m_tag2 = tag;
    }
    void set_tag_impl(const param3 & tag) {
        m_tag3 = tag;
    }
    void set_tag_impl(const param4 & tag) {
        m_tag4 = tag;
    }
    void set_tag_impl(const param5 & tag) {
        m_tag5 = tag;
    }
    void set_tag_impl(const param6 & tag) {
        m_tag6 = tag;
    }
    void set_tag_impl(const param7 & tag) {
        m_tag7 = tag;
    }
    void set_tag_impl(const param8 & tag) {
        m_tag8 = tag;
    }
    void set_tag_impl(const param9 & tag) {
        m_tag9 = tag;
    }
    void set_tag_impl(const param10 & tag) {
        m_tag10 = tag;
    }

private:
    param1 m_tag1;
    param2 m_tag2;
    param3 m_tag3;
    param4 m_tag4;
    param5 m_tag5;
    param6 m_tag6;
    param7 m_tag7;
    param8 m_tag8;
    param9 m_tag9;
    param10 m_tag10;
};


}}}}

#include <hpx/util/logging/tag/defaults.hpp>

#endif

