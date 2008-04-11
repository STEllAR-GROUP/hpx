namespace boost { namespace logging {

/** 
@page breaking_changes Breaking changes

- @ref breaking_change_v_20

@section breaking_change_v_20 v0.20.1 - Use filters/loggers as functions: append "()" to them

@subsection breaking_change_v_20_what_changed What changed?

Now, for every call to a filter/logger, you need to append "()" to it. You'll need to do this:
- when initializing the logs
- when you've defined your macros 
  - any usage of BOOST_LOG_USE_LOG_IF_LEVEL, BOOST_LOG_USE_LOG_IF_FILTER, BOOST_LOG_USE_SIMPLE_LOG_IF_FILTER
- when dealing with filters (turning them on/off, changing filter levels)


Example

Before:

@code
#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l, g_log_filter->is_enabled() ) 

g_l->writer().add_formatter( formatter::idx(), "[%] "  );
g_l->writer().add_formatter( formatter::append_newline_if_needed() );
g_l->writer().add_destination( destination::file("out.txt") );
g_l->mark_as_initialized();

g_log_filter->set_enabled(false);

@endcode

After:

@code
#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) 

g_l()->writer().add_formatter( formatter::idx(), "[%] "  );
g_l()->writer().add_formatter( formatter::append_newline_if_needed() );
g_l()->writer().add_destination( destination::file("out.txt") );
g_l()->mark_as_initialized();

g_log_filter()->set_enabled(false);

@endcode



@subsection breaking_change_v_20_why Why?

Yes, I know it's near review time and I should not do this, but...

I've gotten a bit of feedback on my extensive use of macros, and some of those being rather complex.
And lately I've re-analyzed the library, see where implementation could be simpler, etc., and the main part was - macros.

More to the point, declaring/defining a log/filter should be simpler - the macros for doing this should be simpler.

Previously, I wanted to abstract away the fact that when using BOOST_LOG_DECLARE/DEFINE, internally we'd be creating a @ref rationale_use_functions "function",
but use it as a variable:

@code
// before version v0.20

BOOST_DEFINE_LOG(g_l, logger_type)

// in code, "g_l" is used like a variable
g_l->writer().add_formatter( formatter::idx(), "[%] "  );
@endcode

However, @c g_l was a variable which always forwarded to a @ref rationale_use_functions "function". Making this possible required quite a bit of trickery,
and complicated the implementation of BOOST_LOG_DECLARE/DEFINE.

And even worse, extending them would be a nightmare: Basically I wanted to allow exporting a logger, for instance, from a DLL to an application.
With the previous syntax, it would be very complex to implement.

So, to make it easier, now, BOOST_LOG_DECLARE/DEFINE declare/define the function directly (as opposed to before, when they did define a function
with a different name, and then define a variable referencing that function).

To understand why BOOST_LOG_DECLARE/DEFINE declare/defines a @b function, as opposed to a @b variable, see @ref rationale_use_functions.


*/

}}
