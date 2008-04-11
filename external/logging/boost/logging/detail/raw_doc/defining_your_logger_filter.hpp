namespace boost { namespace logging {

/** 
@page defining_your_logger_filter Declaring/Defining your logger/filter class(es)

- @ref defining_prerequisites
    - @ref defining_prerequisites_typedef 
    - @ref defining_prerequisites_dd

- @ref typedefing_your_filter 
    - @ref typedefing_your_filter_scenario 
    - @ref typedefing_your_filter_manually 
- @ref typedefing_your_logger 
    - @ref typedefing_your_logger_scenario 
    - @ref typedefing_your_logger_format_write 
    - @ref typedefing_your_logger_use_logger 

- @ref declare_define 
    - @ref declare_define_use_macros 
        - @ref declare_define_use_macros_under_the_hood 
        - @ref declare_define_use_macros_as_functions 
        - @ref declare_define_use_macros_fast_compile 
        - @ref declare_define_use_macros_before_main 
        - @ref declare_define_use_macros_no_after_destroyed 
    - @ref declare_define_manually 
        - @ref declare_define_manually_f_vs_v 
        - @ref declare_define_manually_logger_holder 
        - @ref declare_define_manually_before_main 





@section defining_prerequisites Prerequisites

When using the Boost Logging Lib, you need 2 things (see @ref workflow "Workflow"):
- a filter : which tells you if a logger is enabled or not. Note that you can use the same filter for multiple loggers - if you want. 
- a logger : which does the actual logging, once it's enabled

In order to declare/define filters and loggers:
- you have to choose the filter and logger based on your needs (@ref defining_prerequisites_typedef "typedefing")
- you have to declare and define your fiter and logger (@ref defining_prerequisites_dd "declare and define")


\n\n
@subsection defining_prerequisites_typedef Prerequisites - Typedefing

Typedefing your filter/logger is the process where you find the @c type of your filter/logger. 

Example 1:
@code
using namespace boost::logging::scenario::usage;
typedef use< filter_::change::often<10> > finder;
// now, finder::logger contains the logger type and finder::filter contains the filter type
@endcode

Example 2:
@code
namespace bl = boost::logging;
typedef bl::logger_format_write< > logger_type;
typedef bl::filter::no_ts filter_type;
@endcode


\n\n
@subsection defining_prerequisites_dd Prerequisites - Declare and define

Now that you know the types of your logger(s) and filter(s), you have to declare and define them.
The easiest way is to use the @c BOOST_DECLARE_LOG* and @c BOOST_DEFINE_LOG* macros:

@code
// in a header file
BOOST_DECLARE_LOG_FILTER(g_log_filter, filter_type ) 
BOOST_DECLARE_LOG(g_l, logger_type) 

// in a source file
BOOST_DEFINE_LOG_FILTER(g_log_filter, filter_type ) 
BOOST_DEFINE_LOG(g_l, logger_type) 

// ... manipulating the logger/filter in the code
g_l()->writer().add_formatter( formatter::idx(), "[%] "  );
g_log_filter()->set_enabled(false);
@endcode

However, there are other ways to declare/define your loggers/filters. We'll explore them @ref declare_define "later".




\n\n
@section typedefing_your_filter Typedefing your filter class

@subsection typedefing_your_filter_scenario Typedefing your filter using scenarios (the easy way)

You can declare/define both your logger and filter based on how you'll use them (scenario::usage).
Thus, you'll deal with the filter like this:

@code
#include <boost/logging/format_fwd.hpp>
using namespace boost::logging::scenario::usage;
typedef use<
        // how often does the filter change?
        filter_::change::often<10>, 
        // does the filter use levels?
        filter_::level::no_levels, 
        // logger info
        ...
        > finder;

// declare filter
BOOST_DECLARE_LOG_FILTER(g_log_filter, finder::filter ) 

// define filter
BOOST_DEFINE_LOG_FILTER(g_log_filter, finder::filter ) 
@endcode


@subsection typedefing_your_filter_manually Typedefing your filter manually

This is where you manually specify the filter class you want. There are multiple filter implementations:
- not using levels - the classes from the filter namespace
- using levels - the classes from the level namespace

Choose any you wish:

@code
#include <boost/logging/format_fwd.hpp>

// declare filter
BOOST_DECLARE_LOG_FILTER(g_log_filter, filter::no_ts ) 

BOOST_DEFINE_LOG_FILTER(g_log_filter, filter::no_ts ) 
@endcode



@section typedefing_your_logger Typedefing your logger class(es)

@subsection typedefing_your_logger_scenario Typedefing your logger using scenarios (the very easy way)

When you use formatters and destinations, you can declare/define both your logger and filter based on how you'll use them (scenario::usage).
Thus, you'll deal with the logger like this:

@code
#include <boost/logging/format_fwd.hpp>

using namespace boost::logging::scenario::usage;
typedef use<
        // filter info
        ...,
        // how often does the logger change?
        logger_::change::often<10>, 
        // what does the logger favor?
        logger_::favor::speed> finder;

// declare
BOOST_DECLARE_LOG(g_log_err, finder::logger ) 

// define
BOOST_DEFINE_LOG(g_log_err, finder::logger ) 

@endcode




@subsection typedefing_your_logger_format_write Typedefing your logger using logger_format_write (the easy way)

When you use formatters and destinations, you can use the logger_format_write class. The template params you don't want to set,
just leave them @c default_.

@code
#include <boost/logging/format_fwd.hpp>

namespace bl = boost::logging;
typedef bl::logger_format_write< bl::default_, bl::default_, bl::writer::threading::on_dedicated_thread > logger_type;

// declare
BOOST_DECLARE_LOG(g_l, logger_type) 

// define
BOOST_DEFINE_LOG(g_l, logger_type)
@endcode



@subsection typedefing_your_logger_use_logger Typedefing your logger using the logger class

In case you don't use formatters and destinations, or have custom needs that the above methods can't satisfy, or
just like to do things very manually, you can use the logger class directly:

@code
#include <boost/logging/logging.hpp>

typedef logger< gather::ostream_like::return_str<>, destination::cout> logger_type;

// declare
BOOST_DECLARE_LOG(g_l, logger_type) 

// define
BOOST_DEFINE_LOG(g_l, logger_type)
@endcode








\n\n
@section declare_define Declaring and defining your logger and filter

At this point, you @em have your logger class and your filter class. Lets assume they are @c logger_type and @c filter_type. 
You could have obtained them like this:

@code
namespace bl = boost::logging;
typedef bl::logger_format_write< > logger_type;
typedef bl::filter::no_ts filter_type;
@endcode




\n\n
@subsection declare_define_use_macros Declaring and defining your logger/filter using macros

This is the simplest way to declare/define your filters. 

Declaring:

@code
// in a header file

#include <boost/logging/format_fwd.hpp>
// if you don't use formatters/destinations, you can include only <boost/logging/logging.hpp>

// declare a filter, called g_log_filter
BOOST_DECLARE_LOG_FILTER(g_log_filter, filter_type) 

// declare a logger, called g_log
BOOST_DECLARE_LOG(g_log, logger_type)
@endcode



Defining:

@code
// in a source file
#include <boost/logging/format.hpp>

// define a filter, called g_log_filter
BOOST_DEFINE_LOG_FILTER(g_log_filter, filter_type) 

// define a logger, called g_log
BOOST_DEFINE_LOG(g_log, logger_type)
@endcode



Specifying some arguments when defining the logger/filter:
@code
// in a source file
#include <boost/logging/format.hpp>

// define a filter, called g_log_filter - assuming it needs an 2 arguments at construction
BOOST_DEFINE_LOG_FILTER_WITH_ARGS(g_log_filter, filter_type, (level::debug, true) ) 

// define a logger, called g_log - assuming it needs an extra argument at construction
BOOST_DEFINE_LOG_WITH_ARGS(g_log, logger_type, ("log.txt") )
@endcode


\n\n
@subsection declare_define_use_macros_under_the_hood Declaring and defining your logger/filter using macros - what happens under the hood?

When using the @c BOOST_DECLARE_LOG* and @c BOOST_DEFINE_LOG* macros, this is what the lib takes care for you:
-# it declares and defines a logger/filter name, as a @b function (which internally contains a static variable)
-# cares about @ref BOOST_LOG_COMPILE_FAST_ON "Fast Compile" (default) or @ref BOOST_LOG_COMPILE_FAST_OFF "not"
-# ensures the logger/filter object is instantiated before @c main(), even if not used before main(). This is very useful,
   since we can assume that before main() there won't be more than 1 threads, thus no problems at initializing the static variable.
-# ensures we don't @ref after_destruction "use a logger and/or filter after it's been destroyed"

\n\n
@subsubsection declare_define_use_macros_as_functions logger/filter as functions


We declare/define the logger/filter as a function, in order to avoid being used before it's initialized. Example:

@code
// Translation unit 1:
logger<...> g_l;

// Translation unit 2:
struct widget {
    widget() {
        // use g_l
        g_l.writer() ....
    }
} g_global_widget;

@endcode

In the above code we have 2 global variables (g_l and g_global_widget) in 2 different translation units. In this case, it's unspecified
which will be constructed first - thus, we could end up having g_global_widget constructed first, and using g_l before g_l is initialized.

To avoid this, g_l should be a function, like this:

@code
// Translation unit 1:
logger<...> & g_l() { static logger<...> l; return l; }

// Translation unit 2:
struct widget {
    widget() {
        // use g_l
        g_l().writer() ....
    }
} g_global_widget;

@endcode

In the above case, when g_l() is used for the first time, it constructs the local @c l, and it all works.
The @c BOOST_DECLARE_LOG* and @c BOOST_DEFINE_LOG* macros take care of this automatically.



\n\n
@subsubsection declare_define_use_macros_fast_compile Fast compiling : On/Off

The @c BOOST_DECLARE_LOG* and @c BOOST_DEFINE_LOG* macros also automatically take care of fast compiling or not.

Fast compiling (on by default) applies only to loggers. It means that you can use the loggers in code, even without knowing their definition.
More to the point, you can log messages throughout your application, even if you don't know the full type of the logger
(a @em typedef is enough).
This avoids inclusion of a lot of header files, speeding the compile process.

If fast compile is on, you only need this when using logs:

@code
// usually in a header file
#include <boost/logging/format_fwd.hpp>
typedef logger_format_write< > logger_type;

BOOST_DECLARE_LOG(g_l, logger_type) 

// macro used for logging
#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) 


// in your code, only by #including boost/logging/format_fwd.hpp, you can log messages
L_ << "this is so cool " << i++;
std::string hello = "hello", world = "world";
L_ << hello << ", " << world;
@endcode

If fast compile is off, when using the logs, you'll need to know the full type of the logger (the definition of the logger class). \n
When using formatters/destinations, this means <tt>\#include <boost/logging/format.hpp></tt>. Also, when logging a message,
the code for doing the actual logging will be generated inline, this taking a bit of compilation time.

@ref macros_compile_time "More details here".

In short, 

When fast compile is off, BOOST_DEFINE_LOG will generate code similar to this:

@code
logger_type * g_l() { static logger_type l; return &l; }
@endcode

When fast compile is on, BOOST_DEFINE_LOG will generate code similar to this:

@code
logger_holder<logger_type> & g_l() { static logger_holder_by_value<logger_type> l; return l; }
@endcode

In the latter case, logger_holder<> holds a pointer to the original log, and when a message is logged, 
it forwards it to the real logger (implemented in logger_holder_by_value).



\n\n
@subsubsection declare_define_use_macros_before_main Ensuring instantiation before main()

The @c BOOST_DECLARE_LOG* and @c BOOST_DEFINE_LOG* macros also automatically ensure that the logger/filter is instantiated before main().
This is very useful, since we can assume that before main() there won't be more than 1 threads, thus no problems at initializing the static variable.

For this, it uses the @c ensure_early_log_creation class, like this:

@code
// this will ensure g_l() is called before main(), even if not used anywhere else before main()
ensure_early_log_creation ensure( g_l() );
@endcode



\n\n
@subsubsection declare_define_use_macros_no_after_destroyed Ensuring you don't use a logger and/or filter after it's been destroyed

See @ref after_destruction "this" for more details.



\n\n
@subsection declare_define_manually Declaring and defining your logger/filter manually

As explained @ref declare_define_use_macros "above", you can use macros to declare/define your loggers/filters.

Of course, you can declare/define them manually. If you decide to do it, please read the @ref declare_define_use_macros "Define/declare ... macros" section
throughly, so that you know what you should watch for.

For example, declaring/defining your logger can be as easy as:

@code
// in a header file
logger_type * g_l();

// example of macro used for logging
#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) 


// in a source file
logger_type * g_l() { static logger_type l; return &l; } 

// example of usage
L_ << "this is so cool " << i++;
std::string hello = "hello", world = "world";
L_ << hello << ", " << world;
@endcode



\n\n
@subsubsection declare_define_manually_f_vs_v Functions versus variables

As I said, you should prefer @ref declare_define_use_macros_as_functions "functions instead of variables" for the obvious reasons.

Thus (when using functions), your code should look like:

@code
// in a header file
logger_type * g_l();

// example of macro used for logging
#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) 


// in a source file
logger_type * g_l() { static logger_type l; return &l; } 

// example of usage
L_ << "this is so cool " << i++;
std::string hello = "hello", world = "world";
L_ << hello << ", " << world;
@endcode

\n
You can use variables, provided that @ref declare_define_use_macros_as_functions "you know the risks".

@code
// in a header file
extern logger_type g_l;

// example of macro used for logging
#define L_ BOOST_LOG_USE_LOG_IF_FILTER((*g_l), g_log_filter()->is_enabled() ) 


// in a source file
logger_type g_l;

// example of usage
L_ << "this is so cool " << i++;
std::string hello = "hello", world = "world";
L_ << hello << ", " << world;
@endcode




\n\n
@subsubsection declare_define_manually_logger_holder Using logger_holder class

You should use @c logger_holder<> when you want to be able to use the logger without knowing its definition (in other words, you only have a typedef).
Thus, you'll only need to #include <boost/logging/format_fwd.hpp> throughout the application. 

In case you're using formatters and destinations, you'll need to #include <boost/logging/format.hpp> :
- when defining the logger
- when initializing it

Note that this will involve a virtual function call for each logged message - when performing the actual logging.

<tt>logger_holder<logger></tt> is the base class - the one that will be used in code/presented to clients. 
The possible implementations are :
- logger_holder_by_value<logger> - holds the original logger by value
  - in case you think the logger could be used after it's been destroyed, you should use this
- logger_holder_by_ptr<logger> - holds the original logger as a pointer (allocates it constructor, deallocates it in destructor)

Example of using logger_holder<> :

@code
// in a header file
logger_holder<logger_type> & g_l();

// example of macro used for logging
#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) 


// in a source file
logger_holder<logger_type> & g_l() {
  static logger_holder_by_value<logger_type> l;
  return l;
}

// example of usage
L_ << "this is so cool " << i++;
std::string hello = "hello", world = "world";
L_ << hello << ", " << world;
@endcode


\n\n
@subsubsection declare_define_manually_before_main  Ensure initialized before main()

If you use loggers/filters as global variables, you don't need to worry about this.

If you use loggers/filters as functions with static variables, they will be initialized on first usage.

This could be problematic, in case the variable is initialized when more than one thread is running.
In some current implementations , if 2 threads are calling the function at the same time (and when each function enters, needs to construct the variable), 
you might end up  with 2 different instances of the same static variable. Thus, trouble.

The easy solution is to use @c ensure_early_log_creation class, like this:

@code
// in the source file
logger_holder<logger_type> & g_l() {
  static logger_holder_by_value<logger_type> l;
  return l;
}
ensure_early_log_creation ensure( g_l() );

@endcode


This will ensure the logger is initialized before main(), thus the above problem does not happen.



*/

}}
