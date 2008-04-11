namespace boost { namespace logging {

/** 
@page main_intro Boost Logging Library v2 : Introduction

- @ref main_motivation
- @ref main_common_usage
- @ref main_feeback 
- @ref page_changelog 

@section main_motivation Motivation

Applications today are becoming increasingly complex. Part of making them easier to develop/maintain is to do logging. 
Logging allows you to later see what happened in your application. It can be a great help when debugging and/or testing it. 
The great thing about logging is that you can use it on systems in production and/or in use - if an error occurs, 
by examining the log, you can get a picture of where the problem is.

Good logging is mandatory in support projects, you simply can't live without it.

Used properly, logging is a very powerful tool. Besides aiding debugging/ testing, it can also show you 
how your application is used (which modules, etc.), how time-consuming certain parts of your program are, 
how much bandwidth your application consumes, etc. - it's up to you how much information you log, and where.

<b>Features</b>

- A simple and clear separation of @ref namespace_concepts "concepts"
    - concepts are also easily separated into namespaces
- A very flexible interface
- You don't pay for what you don't use.
- Allows for internationalization (i18n) - can be used with Unicode characters
- Fits a lot of @ref common_scenarios "scenarios": from @ref common_scenarios_6 "very simple" (dumping all to one log) 
  to @ref scenario::usage "very complex" (multiple logs, some enabled/some not, levels, etc).
- Allows you to choose how you use logs in your code (by defining your own LOG_ macros, suiting your application)
- Allows you to use Log levels (debug, error, fatal, etc). However this is an orthogonal concept - the library
  will work whether you use levels, categories or whatever , or not.
- Efficient filtering of log messages - that is, if a log is turned off, the message is not processed at all
- Thread-safe - the library allows you several degrees of thread-safety, as you'll see
- Allows for formatters and destinations 
    - formatters format the message (like, prepending extra information - an index, the time, thread id, etc)
    - destinations specify where the message is to be written
    - Formatters and Destinations are orthogonal to the rest of the library - if you want you can use them, otherwise
      you can define your own writing mechanism
- Easy manipulation of the logs (turning on/off, setting formatters, destinations, etc)
- Allows you to use @ref tag "tags" (extra information about the context of the log: file/line, function name, thread id, etc.)
- Allows for @ref scoped_logs "scoped logs"
- Has @ref formatter::high_precision_time_t "high precision time formatter"
- easy to configure at run-time, if you wish
  - @ref formatter::named_spacer
  - @ref destination::named
- @ref caching "cache messages before logs are initialized"
- Allows for @ref profile "profiling itself"


\n\n
@section main_common_usage Common Usage

To get you started, here's the <b>most common usage</b>:
\n

@copydoc mul_levels_one_logger 

@ref scenarios_code_1 "Click to see the code"
\n\n\n

To see more examples, check out @ref common_scenarios.


\n\n\n
@section main_feeback Feedback

I certainly welcome all feedback. So, be it a suggestion, or criticism, do write to me:
- contact me: http://torjo.com/contact.html
- If there's a feature you'd like, you can contact me, or drop a comment here:
  http://torjo.blogspot.com/2007/11/boost-logging-v2-your-killer-feature.html \n
  (this way, others can contribute with comments as well)


\n\n\n
@section main_changelog Changelog

@ref page_changelog "See the changelog".



*/

}}
