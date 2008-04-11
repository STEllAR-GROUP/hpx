/**
 Boost Logging library

 Author: John Torjo, www.torjo.com

 Copyright (C) 2007 John Torjo (see www.torjo.com for email)

 Distributed under the Boost Software License, Version 1.0.
    (See accompanying file LICENSE_1_0.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)

 See http://www.boost.org for updates, documentation, and revision history.
 See http://www.torjo.com/log2/ for more details
*/

/**
@example custom_fmt_dest.cpp

@copydoc custom_fmt_dest

@page custom_fmt_dest custom_fmt_dest.cpp Example

This example shows you how easy it is to add your custom formatter /destination classes.

This usage:
- You have one logger
- You have one filter, which can be turned on or off
- You want to format the message before it's written 
- The logger has several log destinations
    - The output goes to console, debug output window, and a file called out.txt - as XML
    - Formatting - prefix each message by its start time, its index, and append newline

\n\n
Custom classes:
- secs_since_start - custom formatter
- as_xml - custom destination


\n\n
Optimizations:
- use a cache string (from optimize namespace), in order to make formatting the message faster


\n\n
The output will look similar to this one:


The console and the debug window will be the same:
@code
+6s [1] this is so cool 1
+6s [2] this is so cool again 2
+7s [3] hello, world
+7s [4] good to be back ;) 3
@endcode


The out.txt file will look like this:


@code
<msg>+6s [1] this is so cool 1
</msg>
<msg>+6s [2] this is so cool again 2
</msg>
<msg>+7s [3] hello, world
</msg>
<msg>+7s [4] good to be back ;) 3
</msg>
@endcode

*/



#include <boost/logging/format_fwd.hpp>

BOOST_LOG_FORMAT_MSG( optimize::cache_string_one_str<> )

#include <boost/logging/format.hpp>
using namespace boost::logging;

typedef logger_format_write< default_, default_, writer::threading::no_ts > logger_type;


BOOST_DECLARE_LOG_FILTER(g_log_filter, filter::no_ts )
BOOST_DECLARE_LOG(g_l, logger_type) 

#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) 

BOOST_DEFINE_LOG(g_l, logger_type)
BOOST_DEFINE_LOG_FILTER(g_log_filter, filter::no_ts )



// Example of custom formatter:
// dump the no. of seconds since start of program
struct secs_since_start : formatter::class_<secs_since_start, formatter::implement_op_equal::no_context> {
    ::time_t m_start;
    secs_since_start() : m_start( ::time(0) ) {}
    void operator()(param str) const {
        ::time_t now = ::time(0);
        std::stringstream out;
        out << "+" << (int)(now-m_start) << "s ";
        str.prepend_string( out.str() );
    }
};

// Example of custom destination:
// Dump each message as XML
struct as_xml : 
        destination::class_<as_xml, destination::implement_op_equal::has_context>, 
        destination::non_const_context<std::ofstream> {

    std::string m_name;
    as_xml(const char* name) : non_const_context_base(name), m_name(name) {}
    void operator()(param str) const {
        context() << "<msg>" << str << "</msg>" << std::endl; 
    }

    bool operator==(const as_xml& other) const { return m_name == other.m_name; }
};


void custom_fmt_dest_example() {
    //         add formatters and destinations
    //         That is, how the message is to be formatted and where should it be written to
    g_l()->writer().add_formatter( formatter::idx(), "[%] " );
    g_l()->writer().add_formatter( formatter::append_newline() );
    g_l()->writer().add_formatter( secs_since_start() );

    g_l()->writer().add_destination( destination::cout() );
    g_l()->writer().add_destination( destination::dbg_window() );
    g_l()->writer().add_destination( as_xml("out.txt") );
    g_l()->mark_as_initialized();

    int i = 1;
    L_ << "this is so cool " << i++;
    L_ << "this is so cool again " << i++;

    std::string hello = "hello", world = "world";
    L_ << hello << ", " << world;

    g_log_filter()->set_enabled(false);
    L_ << "this will not be written to the log";
    L_ << "this won't be written to the log";

    g_log_filter()->set_enabled(true);
    L_ << "good to be back ;) " << i++;
}



int main() {
    custom_fmt_dest_example();
}


// End of file
