namespace boost { namespace logging {

/** 
@page miscelaneous Miscelaneous things

@section misc_use_defaults Template parameters - Using defaults

This parameter is optional. This means you don't need to set it, unless you want to.
Just leave it as @c default_, and move on to the paramers you're interested in.

Example:

@code
typedef logger_format_write< default_, default_, writer::threading::on_dedicated_thread > logger_type;
@endcode

@section misc_unicode Internationalization - Using Unicode charaters

In case you want to log unicode characters, it's very easy:

- just <tt>\#define BOOST_LOG_USE_WCHAR_T</tt> before including any Boost.Logging files
- For Windows, in case the @c UNICODE or @c _UNICODE is defined, the @c BOOST_LOG_USE_WCHAR_T is defined automatically for you
  - If you don't wish that, please <tt>#define BOOST_LOG_DONOT_USE_WCHAR_T</tt> globally, before including any Boost Logging Lib files.



@section misc_compilers Compilers it's been tested on

The Boost Logging Lib has been tested with the following compilers:

- VC 2005 
- VC 2003
- gcc 3.4.2
- gcc 4.1

I've tested it under Windows, and ran the tests successfully on Fedora 8.


@section misc_bjam Compiling with bjam

You can compile the scenarios and/or tests with bjam. The easy way:

- On Windows:
  - go the the scenarios or tests directory
  - run <tt>run_win.bat <em> path_to_boost toolset </em> </tt>
  - example: <tt>run_win d:/boosts/boost_1_34_1/ msvc </tt>
- On Linux
  - go the the scenarios or tests directory
  - run @c <tt>bash run_linux.sh <em> path_to_boost toolset</em> </tt>
  - example: <tt>sh run_linux.sh /home/jtorjo/Desktop/boost_1_34_1/ gcc </tt>

*/

}}
