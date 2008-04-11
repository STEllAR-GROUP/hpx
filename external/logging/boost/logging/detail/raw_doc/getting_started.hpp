namespace boost { namespace logging {

/** 
@page getting_started Getting started - the Fast & Furious way...


- @ref getting_started_basics 
- @ref getting_started_code
- @ref getting_started_example_no_levels 
- @ref getting_started_example_use_levels
- @ref getting_started_other_examples 



So, you don't have time to spend (at least not right now), to read the @ref log_tutorial "tutorial", but still want to use the Boost Logging Lib.

\n\n
@section getting_started_basics The basics

Here are the quick facts:
- We use the folloging concepts:
  - logger : the class that does the logging. You can have several logger objects in your code, but usually one will suffice.
    - writer : an object, part of the logger, that is in charge of writing the message
  - filter : the class that does the filtering - that is, it knows if a message should be logged or not

There are several @ref writer "writers". The most common uses the concept of formatters and destinations:
- formatter : formats the message before writing it (for instance, by prepending time to it, an index, etc.)
- destination : represents a place where the message is to be written to (like, console, a file, debug window, etc.)

Once you have a %writer, you can add several formatters and/or destinations. The %writer object knows how to call them.
The common (and default) one calls them like this:
- first, all formatters (in the order they were added)
- second, all destinations (in the order they were added)

The easiest %writer is the @ref boost::logging::writer::named_write "Named Writer". It's an easy interface to using formatters and destinations.
In other words, you set @ref format_string_syntax "a format string" and @ref dest_string_syntax "a destination string".


\n\n
@section getting_started_code Structuring your code

You'll use macros to:
- declare and define the filters and loggers
- do %logging.

You should structure your code like this:
- have a header file, in which you @c \#include the Boost Logging Lib forward classes, and you declare your loggers and filters
- have a source file, in which you define your loggers and filters, and eventually initialize them
- in the rest of the code, when you intend to do %logging, just include the above header file

\n\n 
@section getting_started_example_no_levels Example 1 : Have one Named Writer, No levels

Assume you want one logger and one filter - the filter is a simple filter that can only be turned on/off (the filter is not thread-safe). \n
This is something you can copy-paste into your program.

@include fast_start/no_levels.h


\n\n 
@section getting_started_example_use_levels Example 2 : Have one Named Writer , Use Levels

Assume you want one logger and one filter - the filter is a based on levels (the filter is not thread-safe). \n
This is something you can copy-paste into your program.

@include fast_start/use_levels.h


@section getting_started_other_examples Other examples...

Yup, we have @ref common_scenarios "other examples" as well. We also have a @ref starter_project "starter project".


*/

}}
