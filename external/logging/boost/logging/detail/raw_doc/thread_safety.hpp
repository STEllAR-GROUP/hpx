namespace boost { namespace logging {

/** 
@page thread_safety Thread safety

When talking about thread-safety, there are 2 types of things to consider:
- the logger class(es)
- the filter class(es)

Based on your application, you can fine tune any of the above to suit your needs:
- use it as single threaded
- use @ref macros_tss "TSS" (Thread Specific Storage)
  - use TSS - have some data you modify thread-safe(using mutexes), and have each thread cache the value, and refresh at a given period (very efficient)
  - initialize your logger/filter only once, and once it's initialized, always use that value (very efficient)
- thread-safe (use mutexes) - every access uses a mutex; very slow in comparison to the above methods

In addition to the above, for loggers, you can have an even faster method of writing the messages to their destinations: 
@ref writer::on_dedicated_thread "on a dedicated thread".

\n\n
The easiest way to specify the logger and filter classes, is to @ref scenario::usage "customize the lib to suit your application's needs".

Alternatively, you can @ref defining_your_logger_filter "define them manually":
- filters:
  - @c filter::no_ts - single threaded filter
  - @c filter::ts - thread-safe filter
  - @c filter::use_tss_with_cache - use TSS: have each thread cache the value, and refresh at a given period
  - @c filter::use_tss_once_init - use TSS: once the value is set, it will be used by each thread
- loggers:
  - <tt>logger<default_, default_, writer::threading::no_ts> </tt> - single threaded logger
  - <tt>logger<default_, default_, writer::threading::ts_write> </tt> - thread-safe logger
  - <tt>logger<default_, default_, writer::threading::on_dedicated_thread> </tt> - thread-safe logger, writing on dedicated thread

\n
Note: I recommend defining loggers manually only if you know the lib very well. Otherwise, choose one of these:
- @ref scenario::usage "customize the lib to suit your application's needs"
- @ref defining_your_logger_filter 
*/

}}
