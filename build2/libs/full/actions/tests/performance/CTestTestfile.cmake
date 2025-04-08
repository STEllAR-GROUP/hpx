# CMake generated Testfile for 
# Source directory: /Users/harith/Desktop/Open Source/hpx/libs/full/actions/tests/performance
# Build directory: /Users/harith/Desktop/Open Source/hpx/build2/libs/full/actions/tests/performance
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(tests.performance.modules.actions.serialization_overhead "/Users/harith/Desktop/Open Source/hpx/build2/bin/hpxrun.py" "/Users/harith/Desktop/Open Source/hpx/build2/bin/serialization_overhead_test" "-e" "0" "-t" "1" "-l" "1" "-v" "--")
set_tests_properties(tests.performance.modules.actions.serialization_overhead PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;126;add_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;269;add_hpx_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;286;add_test_and_deps_test;/Users/harith/Desktop/Open Source/hpx/libs/full/actions/tests/performance/CMakeLists.txt;29;add_hpx_performance_test;/Users/harith/Desktop/Open Source/hpx/libs/full/actions/tests/performance/CMakeLists.txt;0;")
