namespace boost { namespace logging {

/** 
@page common_usage_steps_fd Common steps when using Formatters and destinations

\n
<b>The easy way, use Named Formatters and Destinations</b>

You use a string to specify Formatters, and a string to specify Destinations. Thus, you use the @ref boost::logging::writer::named_write "writer::named_write".

First, the examples: @ref scenarios_code_mom "example 1", @ref scenarios_code_noo "example 2"



- Step 1: (optional) Specify your @ref BOOST_LOG_FORMAT_MSG "format message class" and/or @ref BOOST_LOG_DESTINATION_MSG "destination message class". By default, it's <tt>std::(w)string</tt>.
  You'll use this when you want a @ref optimize "optimize string class". Or, when @ref boost::logging::tag "using tags"
- Step 2: Specify your logging and filter classes
  - Step 2A: Typedef your logger as <tt>typedef boost::logging::named_logger<>::type logger_type;</tt> and @ref typedefing_your_filter "typedef your filter class"
  - Step 2B: Declare the @ref declare_define "filters and loggers" you'll use (in a header file)
  - Step 2C: Define the @ref declare_define "filters and loggers" you'll use (in a source file). We need this separation
    (into declaring and defining the logs/filters), in order to @ref macros_compile_time "make compilation times fast".
- Step 3: Define the @ref defining_logger_macros "macros through which you'll do logging"
- Step 4: Initialize the logger. 
  - Step 4A: Set the @ref boost::logging::writer::named_write "formatters and destinations", as strings. 
  - Step 4B: @ref boost::logging::logger_base::mark_as_initialized "Mark the logger as initialized"


\n
<b>The manual way</b>

First, the examples: @ref common_your_scenario_code "example 1", @ref common_your_mul_logger_one_filter "example 2"

- Step 1: (optional) Specify your @ref BOOST_LOG_FORMAT_MSG "format message class" and/or @ref BOOST_LOG_DESTINATION_MSG "destination message class". By default, it's <tt>std::(w)string</tt>.
  You'll use this when you want a @ref optimize "optimize string class". Or, when @ref boost::logging::tag "using tags"
- Step 2: (optional) Specify your @ref boost::logging::manipulator "formatter & destination base classes"
- Step 3: Specify your logging and filter classes
  - Step 3A: @ref typedefing_your_filter "Typedef your filter class(es)" and @ref typedefing_your_logger "Typedef your logger class(es)" 
  - Step 3B: Declare the @ref declare_define "filters and loggers" you'll use (in a header file)
  - Step 3C: Define the @ref declare_define "filters and loggers" you'll use (in a source file). We need this separation
  (into declaring and defining the logs/filters), in order to @ref macros_compile_time "make compilation times fast".
- Step 4: Define the @ref defining_logger_macros "macros through which you'll do logging"
- Step 5: Initialize the logger
  - Step 5A: Add @ref boost::logging::manipulator "formatters and destinations". That is, how the message is to be formatted...
  - Step 5B: @ref boost::logging::logger_base::mark_as_initialized "Mark the logger as initialized"


*/

}}
