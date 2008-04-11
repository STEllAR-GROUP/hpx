namespace boost { namespace logging {

/** 
@page headers_to_include Headers to #include

- @ref headers_to_include_declare 
- @ref headers_to_include_define 
- @ref headers_to_include_example 

\n
@section headers_to_include_declare When declaring the loggers

@attention
If you want to log a message using a certain logger, that logger needs to be declared.


- when using the @ref boost::logging::writer::named_write "Named Writer" (an easy interface to Formatters and Destinations)

@code
#include <boost/logging/format/named_write_fwd.hpp>
@endcode


- when using Formatters and Destinations

@code
#include <boost/logging/format_fwd.hpp>
@endcode



- when using Logging, without Formatters/Destinations

@code
#include <boost/logging/logging.hpp>
@endcode


- if in addition, if you want to do %logging on a dedicated thread

@code
// when you log messages on a dedicated thread (see writer::on_dedicated_thread class)
#include <boost/logging/writer/on_dedicated_thread.hpp> 
@endcode



\n\n
@section headers_to_include_define When defining/initializing the loggers

@attention
If you want to construct the logger, or to initialize it, the logger needs to be defined.
In other words, the corresponding logger class needs to be defined.


- when using the @ref boost::logging::writer::named_write "Named Writer" (an easy interface to Formatters and Destinations)

@code
#include <boost/logging/format/named_write.hpp>
@endcode


- when using Formatters and Destinations

@code
#include <boost/logging/format.hpp>
@endcode


- when using tags

@code
#include <boost/logging/formatter/tags.hpp>
@endcode


- when using Logging, without Formatters/Destinations

@code
#include <boost/logging/logging.hpp>
@endcode




\n\n
@section headers_to_include_example An Example - the starter project

Note that usually you'll have a header file of your own, where you declare the logs. And a source file where you define and initialize your logs.
You can take a look at the @ref starter_project "starter project".



*/

}}
