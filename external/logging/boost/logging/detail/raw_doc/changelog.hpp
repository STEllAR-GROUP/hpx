/** 
@page page_changelog Changelog

@section changelog_cur_ver Current Version: v0.22.8, 4 feb 2008
- added license to all .h .cpp files (in scenarios/tests - they were missing)
- removed loggingvc8.cpp file (useless)
- lib -> libs
- doxygen.txt -> [lib -> libs]

- v0.22.7 - the version that is Boost Reviewed
- updated batches for building with bjam
- made mul_levels_mul_logers run successfully on Fedora8
- ts_posix.hpp - use BOOST_ASSERT instead of assert
- solved bug in using_tags - delete_array used after logger destroyed
- solved bug in no_levels_with_route - care about formatter::spacer
- solved gcc warning - m_writer initialized before m_original_logger
- fixed issue in Jamfile.v2 - runtime-link should refer only to msvc

v0.21.13, 1 feb 2008
- added known_issues section
- added getting_started section
- fixed scoped logs and added test
- added test_on_dedicated_thread
- updated Jamfile.v2 files for examples and test, for them to compile all files tests/examples
- updated Jamfile.v2 to actually build the tests + the tests to use BOOST_CHECK
- updated docs : about caching/destruction
- not using after_being_destroyed.hpp anymore - no need (see @ref after_destruction).
- made it much easier to include Boost Logging Lib headers: format_fwd.hpp, named_write_fwd.hpp, which internally accomodate for slow compilation
  (that is, even if slow compile, you won't need to include more headers in your your_app_log.h file)
- updated docs + named_writer -> named_write
- now can include <boost/logging/format.hpp> without including format_fwd.hpp before, and use BOOST_LOG* macros, and code still compiles and works as expected
- moved tags.hpp - not to be included directly by mistake
- added BOOST_LOG_DONOT_USE_WCHAR_T for Windows users
- added more docs to named_logger
- added named_logger - much easier to use the logging lib - doesn't require any advanced knowledge at all
- simplified examples, removed some of them from the "Scenarios" page; they're still found in the "Examples" section in the docs
- should only fix bugs & update docs until review

v0.20.16, 21 jan 2008
- docs about using loggers/filters are they were destroyed
- more docs about @ref defining_your_logger_filter 
- added example using profiling, @ref common_scenario_profile "see it"
- refactored finding the writer; added logger_to_writer class
- fully implemented profiling 
- added logger_to_gather, removed DEFINE_GATHER_CLASS; 
- modified compile_time example, to see compile time with precompiled header
- added profiler - allow profiling the logging code
- solved bug in logger_holder_by_value - many thanks to Olaf!
- refactored logger_holder - that is, we need to return logger_holder, 
  but keep logger_holder_by_value, for after_destroyed to work
- added test for using logger after it's been destroyed 
- handle using logger after it's been destroyed 
- simpler version of logger (derive from logger_base)
- added dll_and_exe - show how you can use a logger that is defined in a DLL, and use it from an EXE
- explained the breaking change in the docs
- updated the tests to compile (after the breaking change)
- @ref breaking_change_v_20 "BREAKING CHANGE" : when you use the filters/loggers, use them as function names (append "()" to them)
  - Please take a look at the updated examples or at the online help. Thanks for understanding.

v0.14.1, 17 jan 2008
- added test_after_destroyed - works on VC8
- moved logger::cached into a base class and made it virtual

v0.13.17, 16 jan 2008
- added sample compile_time (for testing compile time)
- added destination::named test
- automatically include writer/ts_write.hpp from format.hpp
- added test_tags test
- added test_named_spacer test
- solved bug if user doesn't use formatters/destinations, but uses cached string classes
- solved bug in rolling_file - it could not compile on gcc
- added test_rolling_file (tests rolling_file)
- solved bug - when used LOG_ in switch - thanks to Todd Neal!
- added test for formatter::file
- included <map> in destination::named - many thanks Jens Seidel!
- added Jamfiles for building examples/tests - not fully tested yet, since had problems buidling latest boost from SVN
- added docs about caching messages
- can cache messages before initialization of logs
- gather classes now have msg_type instead of param (msg_type contains the type of the returned msg - without const or &)

v0.12.13, 29 dec 2007
- added tss_ostringstream - allow faster creation of stringstreams (one stringstream per thread)
- added destination::named - similar to formatter::named_spacer, but for destinations
- added possibility to flush a rolling file, and fixed a bug
- added high precision_time tag
- added new includes for files to be able to be compiled standalone - many thanks Jens Seidel!
- added high precision timer 
- removed "#define BOOST_LOG_COMPILE_FAST_OFF" from samples - no needed and could be misleading
- added boost::logging::formatter::named_spacer
- created lock_resource_finder namespace - with existing resource lockers
- added boost::logging::formatter::spacer
- removed copyright from examples (scenarios) - so that they're easier to read in the documentation
- added scoped logs
- removed #ifdef SINGLE_TEST from scenarios
- added todo.html

v0.11.20, 16 nov 2007
- restructured some of the docs/added more - the docs should be rather ok now
- removed sink.hpp - no need anymore
- removed old_examples
- removed do_not_use tests
- more changes in the documentation ; also "refactored" table of contents
- finalized documentation about macros
- solved bugs in append_newline_if_needed() when using tags
- updated docs - added page showing how to declare/define your logger/filter classes
- updated macros documentation (still to do)
- added @ref starter_project
- solved bug when using append_newline_if_needed() - when not used with string
- solved bug at opening file for initial overwrite
- added destination::stream and test_log_output
- applied small patch from Jens Seidel - many thanks!
- moved a few files into detail/ directory - to avoid confusion as to what header should be included when
- fixed bug when including only macros.hpp file - many thanks Jens Seidel!
- added Boost.Logging Requirements page
- added tags - and documented them
- applied patch from Jens Seidel - many thanks!

v0.11.4, 10 nov 2007
- made basic_usage compile again (some late changes made the build break)
- removed #ifdef that would allow TSS only on Windows
- can now see the current version from the contents
- compiles with gcc 3.4.2, gcc 4.1, and VS 2005
- we have boost::logging::scenario namespace, which finds the right filter/logger, based on your application's needs

v0.10.12, 8 nov 2007
- added tss_resource_once_init and correspondent filter/logger class
- added scenario.hpp + documentation for it
- added and tested thread-safety (ts_write/on_dedicated_thread) for logger_format_write<>
- added 2 more scenarios: ts_loger_one_filter.cpp and ded_loger_one_filter.cpp

v0.10.9, 3 nov 2007
- added virtual destructor for logger<gather,default_> - many thanks Martin Baeker!
- added TSS tests 
  - test test_ts_resource_with_cache SUCCESSFUL on win32, vs2005
  - test test_tss SUCCESSFUL on win32, vs2005
  - test test_simple_tss SUCCESSFUL on win32, vs2005
- append_enter -> append_newline
- finalized tss_resouce_with_cache
- added TSS - however, not tested (TSS is off, for now)
- compiles on VS2005, gcc 3.4.2 , gcc 4.1 and gcc 4.3

v0.9.12.3, 1 nov 2007
- move ts/* files to detail directory
- doc - standalone page with changelog
- moved changelog.hpp to detail/raw_doc dir

v0.9.11, 31 oct 2007
- no more stdafx.* files

v0.9.10, 31 oct 2007
- destination::file - on linux don't add ios_base::in when not original overwrite. Many thanks to Martin Baeker!

v0.9.9, 31 oct 2007
- fixed fastest*.cpp scenarios so that they compile. Many thanks to Martin Baeker!

v0.9.8, 31 oct 2007
- added small things to make it compile with gcc 4.1 and 4.3
  (note: I just solved some errors that were pointed out by Jens Seidel, not sure if this fixed everything)

v0.9.7, 30 oct 2007
- added small things to make it compile with gcc 4.1 and 4.3
  (note: I just solved some errors that were pointed out by Jens Seidel, not sure if this fixed everything)

v0.9.6, 30 oct 2007
- removed garbage from stdafx.h files - so that it can be compiled under Linux

v0.9.5, 30 oct 2007
- compiles with gcc 4.1 and 4.3 
  (note: I just solved some errors that were pointed out by Jens Seidel, not sure if this fixed everything)

v0.9.4, 30 oct 2007
- compiles with gcc 3.4.2

v0.9.3, 30 oct 2007
- updated documentation 
    - updated workflow/most of the pages in the table of contents
    - removed most of references to @c process_msg class

v0.9.2, 30 oct 2007
- Breaking chage: 
  - @c process_msg class has been removed. Now @c logger takes its place
  - Instead of logger<use_format_write<>>, use logger_format_write<>

v0.9.1, 30 oct 2007
- implemented BOOST_DEFINE_LOG/BOOST_DECLARE_LOG for when not compiling fast
- BEFORE a breaking change that will happen :
  - logger as is now will dissapear
  - the new logger class will be the old process_msg class
  

v0.9.0, 28 oct 2007
- use_format_write has 4 params now, I've added 2 new params: thread_safety and gather 
  - if you want to leave something unchanged, use "default_" as argument
- added more complex example: Line Counter application
- Breaking change: 
  - filters are declared with BOOST_DECLARE_LOG_FILTER, and defined with BOOST_DEFINE_LOG_FILTER
  - filters are now used with operator->, instead of "."
  - Example: 
		BOOST_DEFINE_LOG_LEVEL(g_log_level(), level::holder ) 
		... 
		g_log_level()->set_enabled(level::error);
  - rationale: filters, same as levels, could be used before main


v0.8.3, 22 oct 2007
- now have 8 scenarios
- updated documentation


v0.8.2, 22 oct 2007
- added one more common scenario (with_route)


v0.8.1, 22 oct 2007
- added one more common scenario
- finally figured out how I can add examples using doxygen :)


v0.8, 20 oct 2007
- added use_format_write class
- removed the filter from the logger class (need to update documentation)
- added Common scenario


v0.7.3, 19 oct 2007
- added the manipulator::class_ concept
- added some documentation


v0.7.2, 18 oct 2007
- format_write cares if formatter/destination is generic or not
- solved bug when deleting manipulators (same_type needed to have a virtual destructor)

v0.4, 9 oct 2007
- ts_write and on_dedicated_thread work

v0.5, 11 oct 2007
- compiles
- added documentation

v0.5.2, 11 oct 2007
- changed license
- added to boost

v0.6, 13 oct 2007
- added formatters from v1. of the library
- added convert_format cool function :P

v0.7, 15 oct 2007
- compiles with gcc 3.4.2

*/
