namespace boost { namespace logging {

/** 
@page known_issues Known Issues

- @ref known_issue_modifying_manipulator 



@note
This section assumes you're quite familiar with Boost Logging Lib v2, thus the concepts used here are not explained.


\n
@section known_issue_modifying_manipulator Modifying a manipulator while it's used

\n
@subsection known_issue_modifying_manipulator_code Sample code

@code
#define L_(lvl) BOOST_LOG_USE_LOG_IF_LEVEL(g_l(), g_log_level(), lvl )

destination::stream out(std::cout);
g_l()->writer().add_destination(out);
...


// Thread 1
out.stream(std::cerr);

// Thread 2
L_ << "whatever";
@endcode

\n
@subsection known_issue_modifying_manipulator_resolution Resolution

If the above code executes concurrently on Threads 1 and 2, we could get in trouble.

This happens because the thread-safe access is guaranteed :
- only if you specify it - when defining the writer (see @ref scenario::usage "scenario based on usage" and logger_format_write)
- it's guaranteed only for <em>adding and deleting</em> manipulators, not for modifying manipulators.

If we were to allow modifying manipulators, we'd have to:
-# either allow a way to pause()/resume() the logger(s) that use a given manipulator or
-# allow thread-safe access to the manipulator objects (meaning each public method would use a mutex, and keep it locked)

In case of the former, pitfalls:
- When modifying a manipulator, you'd need to pause() all loggers that use it. You need to know them.
- This pause()/resume() mechanism will slow the logging process a bit
- This will mean tight coupling between the writer and its member data. 
- In case a lot of logging happens while a logger is paused, could cause either a lot of caching, or a performance hit
  (a lot of threads would have to wait for the resume() to happen)

In case of the latter, pitfalls:
- This will incur a big speed penalty - each time you invoke operator() on the manipulator, will involve a mutex lock/unlock
- The more manipulators a logger uses, the more mutex lock/unlocks will happen. Thus, the speed penalty will be even bigger.

As a side-note, if I were a well known company, I'd just say "This behavior is by design".

\n
@subsection known_issue_modifying_manipulator_when When does it happen?

I would say seldom. This can happen to you only if you want to modify loggers @em after you've initialized - thus, while
you're logging. 

The usual scenario is :
- you initialize Logging once, at beginning of program
- you perform %logging (you don't modify the loggers once they've been initialized)

In the above scenario, this issue will never happen. However, if you do run into it, see @ref known_issue_modifying_manipulator_workaround "below".


\n
@subsection known_issue_modifying_manipulator_workaround Solution

The solution is dead-simple. Just delete this %manipulator, create one of the same type, modify that one, and then add it
to your logger(s). In the original @ref known_issue_modifying_manipulator_code "scenario", you'd do this:

@code
#define L_(lvl) BOOST_LOG_USE_LOG_IF_LEVEL(g_l(), g_log_level(), lvl )

destination::stream out(std::cout);
g_l()->writer().add_destination(out);
...


// Thread 1
destination::stream out2(std::cerr);
g_l()->writer().del_destination(out);
g_l()->writer().add_destination(out2);

// Thread 2 - all good
L_ << "whatever";

@endcode




*/

}}
