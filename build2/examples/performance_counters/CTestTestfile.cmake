# CMake generated Testfile for 
# Source directory: /Users/harith/Desktop/Open Source/hpx/examples/performance_counters
# Build directory: /Users/harith/Desktop/Open Source/hpx/build2/examples/performance_counters
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(tests.examples.performance_counters.access_counter_set "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13" "/Users/harith/Desktop/Open Source/hpx/build2/bin/hpxrun.py" "/Users/harith/Desktop/Open Source/hpx/build2/bin/access_counter_set" "-e" "0" "-t" "4" "-l" "1" "-v" "--")
set_tests_properties(tests.examples.performance_counters.access_counter_set PROPERTIES  _BACKTRACE_TRIPLES "/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;126;add_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;269;add_hpx_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;292;add_test_and_deps_test;/Users/harith/Desktop/Open Source/hpx/examples/performance_counters/CMakeLists.txt;37;add_hpx_example_test;/Users/harith/Desktop/Open Source/hpx/examples/performance_counters/CMakeLists.txt;0;")
add_test(tests.examples.performance_counters.simplest_performance_counter "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13" "/Users/harith/Desktop/Open Source/hpx/build2/bin/hpxrun.py" "/Users/harith/Desktop/Open Source/hpx/build2/bin/simplest_performance_counter" "-e" "0" "-t" "4" "-l" "1" "-v" "--")
set_tests_properties(tests.examples.performance_counters.simplest_performance_counter PROPERTIES  _BACKTRACE_TRIPLES "/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;126;add_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;269;add_hpx_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;292;add_test_and_deps_test;/Users/harith/Desktop/Open Source/hpx/examples/performance_counters/CMakeLists.txt;37;add_hpx_example_test;/Users/harith/Desktop/Open Source/hpx/examples/performance_counters/CMakeLists.txt;0;")
subdirs("sine")
