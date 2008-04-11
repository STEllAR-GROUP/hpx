namespace boost { namespace logging {

/** 
@page boost_logging_requirements Boost.Logging Requirements

The following are my take on what this wiki page has to say:
http://www.crystalclearsoftware.com/cgi-bin/boost_wiki/wiki.pl?Boost.Logging

Last update of this page : 12 Nov 2007


- @ref bl_requirements_major 
    - @ref bl_requirements_ts
    - @ref bl_requirements_scope 
    - @ref bl_requirements_eliminate 
    - @ref bl_requirements_lazy 
    - @ref bl_requirements_sinks 
    - @ref bl_requirements_exception 
    - @ref bl_requirements_tags 
    - @ref bl_requirements_i18 
    - @ref bl_requirements_filt 
    - @ref bl_requirements_attr 
    - @ref bl_requirements_exc_support 
- @ref bl_requirements_design 
    - @ref bl_requirements_config_msg_attr 
    - @ref bl_requirements_global_macro 
    - @ref bl_requirements_general_thoughts 
- @ref bl_requirements_your_say 










@section bl_requirements_major Major requirements

Couldn't aggree more ;)

@section bl_requirements_functional Functional requirements

@subsection bl_requirements_ts Thread Safety

There should be no need for "log core". In my lib, there are no globals:
- the logger keeps all the logging information
- the filter keeps all the filter-related information

You have the ability to make either the logging thread-safe, or the filter, or both.
Also, thread-safety can be implemented in many ways: see scenario::usage

@subsection bl_requirements_scope Scope logging

Couldn't agree more. I will provide for scoped logging soon.

@subsection bl_requirements_eliminate Eliminate log statemets from generated code (Michael Lacher)

"Requirement: It should be possible to prevent all or a specified subset of log messages from generating any code, symbols or strings in the object file."

<b>About code/symbols:</b>

I have not tested this extensively, but just in case you want to fully eliminate statements from your code,
you should use filter::always_disabled.

Most likely, you'd have something like this:

@code
#ifndef ELIMINATE_LOG_STATEMENTS
BOOST_DECLARE_LOG_FILTER(g_log_filter, some_filter_class) 
#else
BOOST_DECLARE_LOG_FILTER(g_log_filter, filter::always_disabled) 
#endif
@endcode

<b>About strings in the object file: </b>

I assume you mean strings like file name, function name, etc.

Note that these are added @b only if you use @ref tag "tags". So, just don't use these tags when you want this information stripped.
Example:

@code
#ifndef ELIMINATE_LOG_STATEMENTS
#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) .set_tag( BOOST_LOG_TAG_FILELINE)
#else
#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) 
#endif
@endcode



@subsection bl_requirements_lazy Full lazy evaluation (Michael Lacher)

Yup, done.

@subsection bl_requirements_sinks Sinks (JD)

- Sink nature (JD) : @b yes
- Independent output formatting and filtering (JD) : @b yes

@subsection bl_requirements_exception Exception safety (abingham)

TODO

@subsection bl_requirements_tags Configurable log message attributes (abingham)

Done. See tag namespace.

@subsection bl_requirements_i18 The library shall manage i18n (JD)

Done. See @c BOOST_LOG_USE_WCHAR_T (if defined, use wchar_t). Or automatically on Windows, if @c UNICODE macro is defined.


@subsection bl_requirements_filt Filtering support (Andrey Semashev)

Filtering support is always provided by the filter. In my lib, the filter is completely separated from the logger.


@subsection bl_requirements_attr Attribute sets (Andrey Semashev)

Don't see the use for this.

@subsection bl_requirements_exc_support Exception logging support (Andrey Semashev)

I guess this could prove useful. I do see this somehow like this:

@code
template<class T> void throw_and_log(const T & exc) {
    DO_SOME_LOGGING;
    throw exc;
}
@endcode


@section bl_requirements_design Design Requirements

@subsection bl_requirements_config_msg_attr Configurable log message attributes (JD)

Yes, I do agree. At this time I have formatters, which you call in a different manner, but I guess
a wrapper can be provided.

@subsection bl_requirements_global_macro Macro access (JD)

Don't agree. You will need to have a logger defined somewhere. Note that when you define your macros, you can choose
which logger you use to log a certain message.  


@subsection bl_requirements_general_thoughts General Thoghths

- "In my experience, there are two completely unrelated tasks for logging with quite different requirements. Usually logging libraries map those two to different log levels, but this is not really correct. There are debug messages which need a very high level (like assertion failures) but which are nonetheless completely unneeded and maybe even harmfull in a release build. On the other hand many normal logging messages "user clicked button x" are not really important in most cases and it is cumbersome of having to enable those just to be able to see debug log messages. (Michael Lacher)"

I would assume you could use levels for this.

- "Another useful feature to support: sinks and sources should be independent enough to be able to run in the different processes/machines. It is not uncommon, to have all the logging sent via socket to a different machine where it is extracted from the socket and written on disk. It would be nice if architecture will allow to implement such separation. (Zigmar)"

That's completely doable ;) Just log "raw information" to memory, or to a file/socket, and have another program that just processes that input,
and performs user-friendly logging.


@section bl_requirements_your_say Your killer feature...

Just in case you have a feature you can't live without, let me know : http://torjo.blogspot.com/2007/11/boost-logging-v2-your-killer-feature.html


*/

}}
