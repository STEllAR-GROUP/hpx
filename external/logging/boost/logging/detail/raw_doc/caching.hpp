namespace boost { namespace logging {

/** 
@page caching Caching messages before logs are initialized

- @ref caching_why 
- @ref caching_BOOST_LOG_BEFORE_INIT_LOG_ALL 
- @ref caching_BOOST_LOG_BEFORE_INIT_CACHE_FILTER 
    - @ref caching_BOOST_LOG_BEFORE_INIT_CACHE_FILTER_the_catch 
- @ref caching_BOOST_LOG_BEFORE_INIT_IGNORE_BEFORE_INIT 

@section caching_why Caching - why is it needed?

Logging is all fine and dandy, but what if you do some %logging before the actual logs are initialized?
It's quite easy to end up doing this:
- usually you initialize logs somewhere within your @c main()
- as applications grow, you'll have global/static variables
- you'll do %logging from the body of the global/static variable's constructors 
    - direcly (from a constructor)
    - indirectly (from some constructor you call a function, which calls a function... which in turn does some logging)

You could even run into more complicated scenarios, where you create other threads, 
which, until you initialize your logs, do some logging. It's good practice to log a thread's begin and end, for instance.

Even though you think this'll never happen to you, usage of singletons and other static variables is quite common,
so better to guard against it.

One solution would be for the library to rely on an external function, like <tt>void boost::logging::init_logs()</tt>,
and have your application have to define it, and it its body, initialize the logs. The library would then make sure
the @c init_logs() is called before any log is used.

There are several problems with this solution:
- logs could be used too soon before you have all the needed data to initialize them (for instance, some could depend
  on command line arguments)
- before any log is used, I would need to make sure @c init_logs() has been called - thus, for each log usage,
  I would need to check if init_logs has been called or not - not very efficient
- within init_logs() I might end up using logs, thus ending in an infinite loop (log needs init_logs(), which uses log)
- you would not have any control over when @c init_logs() is called - what if they need some context information -
  they wouldn't have any context do rely on
- depending on your application, some logs could only be initialized later than others
- if your application has several modules, assume each module has its own log. Thus, each module should be able to 
  initialize its own log when the module is initialized

Thus, I came up with a caching mechanism. You can choose to:
- Cache messages that are written before logs are initialized. For each logged message, you will also cache its corresponding filter 
  (so that if, when initializing the logs, a certain filter is turned off, that message won't be logged)
- Cache messages that are written before logs are initialized. When logs are initialized, all these cached messages are logged
- Ignore messages that are written before the logs are initialized

<b>By default, for each log, cache is turned on. To turn cache off (mark the log as initialized), just call @c mark_as_initialized() on it.
You'll see that I'm doing this on all examples that come with the library.</b>






@section caching_BOOST_LOG_BEFORE_INIT_LOG_ALL Cache messages before logs are initialized regardless of their filter (BOOST_LOG_BEFORE_INIT_LOG_ALL)

This case is the @b default. When cache is on, all messages are cached, regardless of their filter (as if all filters are turned on).
Then, when cache is marked as initialized, all cached messages are logged.

If you want to force this setting, make sure you define the @c BOOST_LOG_BEFORE_INIT_LOG_ALL globally (it's on by default anyway).

@code
...
#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) 
...

L_ << "this message will be logged, even if filter will be turned off";
g_log_filter()->set_enabled(false);
g_l()->mark_as_initialized();
@endcode


\n
@section caching_BOOST_LOG_BEFORE_INIT_CACHE_FILTER Cache messages before logs are initialized/ cache their filter as well (BOOST_LOG_BEFORE_INIT_CACHE_FILTER)

It's a bit inefficient (after invoking the filter, it will always ask if cache is on or off). Also,
it increases the application's size a bit - for each log statement, I will generate a callback that I can call later to see if the filter
is still turned on or off.

@code
...
#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) 
...

L_ << "this message will not be logged";
g_log_filter()->set_enabled(false);
g_l()->mark_as_initialized();
@endcode

If you do want to use this setting, make sure you define the @c BOOST_LOG_BEFORE_INIT_CACHE_FILTER globally.


@subsection caching_BOOST_LOG_BEFORE_INIT_CACHE_FILTER_the_catch BOOST_LOG_BEFORE_INIT_CACHE_FILTER - the catch...

@note
If you don't want to cache the filter, just skip to the @ref caching_BOOST_LOG_BEFORE_INIT_IGNORE_BEFORE_INIT "next section".

If you cache the filter as well, in order for the caching process to work, all the parameters you pass to the filter need to be:
- either compile-time constants, or
- global values

Assume you have a logger with a filter based on levels:
@code
// for exposition only - normally you'd use BOOST_LOG_USE_LOG_IF_LEVEL
#define L_(lvl) BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_level()->is_enabled( lvl ) )
@endcode

If you cache the filter, the expression <tt>g_log_level()->is_enabled( lvl )</tt> needs to be recomputed at a later time 
(when the log is marked as initialized, and all messages that were cached, are logged).
Thus, all parameters that are passed to the your L_ macro need to be either compile-time constants or global values. Otherwise, a compile-time error will
be issued:

@code
void f() {
  boost::logging::level lvl = ...;
  // will generate a compile-time error : using a local variable as param
  L_(lvl) << "wish it could work";
}
@endcode


Normally you should not care about this, since whatever you pass to your logging macros should indeed be constant.




\n
@section caching_BOOST_LOG_BEFORE_INIT_IGNORE_BEFORE_INIT Ignore all messages before mark_as_initialized (BOOST_LOG_BEFORE_INIT_IGNORE_BEFORE_INIT)

In the last case, all messages before @c mark_as_initialized() are ignored.

If you do want to use this setting, make sure you define the @c BOOST_LOG_BEFORE_INIT_IGNORE_BEFORE_INIT globally.

@code
...
#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) 
...

L_ << "this message will NOT be logged";
g_l()->mark_as_initialized();
@endcode




*/

}}
