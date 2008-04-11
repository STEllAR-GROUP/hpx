namespace boost { namespace logging {

/** 
@page workflow Logging workflow


- @ref workflow_introduction 
- @ref workflow_filter 
- @ref workflow_processing 
- @ref workflow_2a 
- @ref workflow_2b 
- @ref workflow_formatters_destinations 


@section workflow_introduction Introduction


What happens when a message is written to the log?
- the message is filtered : is the filter enabled?
    - if so (in other words, the log is turned on), process the message:
        - gather the message
        - write the message to the destination(s)
    - if not (in other words, the log is turned off)
        - completely ignore the message: <em>if the log is not enabled, no processing takes place</em>.

For instance, say you have:

@code
LDBG_ << "user count = " << some_func_taking_a_lot_of_cpu_time();
@endcode

If @c LDBG_ is disabled, everything after "LDBG_" is ignored. Thus, @c some_func_taking_a_lot_of_cpu_time() will not be called.

First of all, we have 2 concepts:
- logger : a "logical" log - something you write to; it knows its destination(s), that is, where to write to
- filter : this provides a way to say if a logger is enabled or not. Whatever that "way to say a logger is enabled or not" means,
  is up to the designer of the filter class.

Note that the logger is a templated class, and the filter is a @ref namespace_concepts "namespace". I've provided
several implementations of the filter concept - you can use them, or define your own.


@section workflow_filter Step 1: Filtering the message

As said above, the filter just provides a way to say if a logger is enabled or not. The %logger and the %filter are completely
separated concepts. No %logger owns a %filter, or the other way around. You can have a %filter per %logger, but most likely
you'll have one %filter, and several loggers:

@code
// Example 1 : 1 filter, 1 logger
BOOST_DECLARE_LOG_FILTER(g_log_filter, filter::no_ts ) 
BOOST_DECLARE_LOG(g_l, logger_type) 

#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) 


// Example 2 : 1 filter (containing a level), several loggers
BOOST_DECLARE_LOG_FILTER(g_log_level, level::holder ) 
BOOST_DECLARE_LOG(g_log_err, logger_type) 
BOOST_DECLARE_LOG(g_log_app, logger_type)
BOOST_DECLARE_LOG(g_log_dbg, logger_type)

#define LDBG_ BOOST_LOG_USE_LOG_IF_LEVEL(g_log_dbg(), g_log_level(), debug ) 
#define LERR_ BOOST_LOG_USE_LOG_IF_LEVEL(g_log_err(), g_log_level(), error )
#define LAPP_ BOOST_LOG_USE_LOG_IF_LEVEL(g_log_app(), g_log_level(), info ) 
@endcode

Every time, before anything gets written to the log, the filter is asked if <em>it's enabled</em>. If so, the processing of the message takes place
(gathering the message and then writing it). Otherwise, the log message is completely ignored.

What <em>it's enabled</em> is depends on the filter class you use:
- if it's a simple class (filter::no_ts, filter::ts, filter::use_tss_with_cache), it's simply the @c is_enabled function (Example 1, above)
- if it's a more complex class, it's up to you
  - for instance, the level::holder_no_ts exposes an <tt>is_enabled(level)</tt>, so you can ask if a certain level is enabled (Example 2, above)
    Thus, logging takes place only if that certain level is enabled (@c debug for LDBG_, @c info for LAPP_, @c error for LERR_)




\n\n
@section workflow_processing Step 2: Processing the message

Once we've established that the logger is enabled, we'll @em process the message. This is divided into 2 smaller steps:
- gathering the message
- writing the message



@section workflow_2a Step 2A: Gathering the message

The meaning of "gathering the message" depends on your application. The message can:
- be a simple string,
- it can contain extra info, like: level, category, etc.
- it can be written all at once, or using the cool "<<" operator
- or any combination of the above

Depending on your needs, gathering can be complex or not. However, it's completely decoupled from the other steps.
Gathering goes hand in hand with @ref macros_use "macros".

The cool thing is that you decide how the <i>Logging syntax</i> is - depending on how you want to gather the message.
All of the below are viable options:

@code
L_("reading " + word);
L_ << "this " << " is " << "cool";
L_(dbg) << "happily debugging";
L_(err,"chart")("Cannot load chart")(chart_path);
@endcode

How you gather your message, depends on how you @ref macros_use "#define L_ ...".

In other words, gathering the message means getting all the message in "one piece", so that it can be written. \n
See the 
- the gather namespace - classes for gathering
- the gather::ostream_like - classes for gathering, using the cool "<<" operator



\n\n
@section workflow_2b Step 2B: Writing the message

Now that you have the message, you're ready to write it. Writing is done by calling @c operator() on the writer object.

What you choose as the writer object is completely up to you. It can be as simple as this:

@code
// dump message to cout
struct write_to_cout {
    void operator()(const std::string & msg) const {
        std::cout << msg << std::endl ;
    }
};

typedef logger< gather::ostream_like::return_str<std::string>, write_to_cout> logger_type;
BOOST_DECLARE_LOG(g_single_log, logger_type)
BOOST_DECLARE_LOG_FILTER(g_filter, filter::no_ts)

#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_single_log, g_filter->is_enabled() ) 

// usage
int i = 100;
L_ << "this is " << i << " times cooler than the average log";
@endcode

You can define your own types of writers. The %writer classes that come with this library are in <tt>namespace writer</tt>.

At this time, I've defined the concept of writer::format_write - writing using @ref manipulator "Formatters and Destinations".
Simply put, this means formatting the message, and then writing it to destination(s).

For each log, you decide how messages are formatted and to what destinations they are written. Example:

@code
typedef logger_format_write< > logger_type;

BOOST_DECLARE_LOG_FILTER(g_log_filter, filter::no_ts ) 
BOOST_DECLARE_LOG(g_l, logger_type) 

#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) 

// add formatters : [idx] [time] message <enter>
g_l()->writer().add_formatter( formatter::idx() );
g_l()->writer().add_formatter( formatter::time("$hh:$mm.$ss ") );
g_l()->writer().add_formatter( formatter::append_newline() );
// add destinations : console, output debug window, and a file called "out.txt"
g_l()->writer().add_destination( destination::cout() );
g_l()->writer().add_destination( destination::dbg_window() );
g_l()->writer().add_destination( destination::file("out.txt") );

// usage
int i = 1;
L_ << "this is so cool " << i++;
L_ << "this is so cool again " << i++;

// possible output:
// [1] 12:32:10 this is so cool 1
// [2] 12:32:10 this is so cool again 2
@endcode




\n\n
@section workflow_formatters_destinations Workflow when using formatters and destinations

When using @ref manipulator "formatters and destinations", there are some steps you'll usually take.

Remember:
- formatter - allows formatting the message before writing it (like, prepending extra information - an index, the time, thread id, etc)
- destination - is a place where the message is to be written to (like, the console, a file, a socket, etc)


@copydoc common_usage_steps_fd

There are plenty of @ref common_scenarios "examples" together with @ref scenarios_code "code".

*/

}}
