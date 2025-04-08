# CMake generated Testfile for 
# Source directory: /Users/harith/Desktop/Open Source/hpx/libs/core/serialization/tests/performance
# Build directory: /Users/harith/Desktop/Open Source/hpx/build2/libs/core/serialization/tests/performance
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(tests.performance.modules.serialization.serialization_performance "/Users/harith/Desktop/Open Source/hpx/build2/bin/hpxrun.py" "/Users/harith/Desktop/Open Source/hpx/build2/bin/serialization_performance_test" "-e" "0" "-t" "1" "-l" "1" "-v" "--" "100")
set_tests_properties(tests.performance.modules.serialization.serialization_performance PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;126;add_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;269;add_hpx_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;286;add_test_and_deps_test;/Users/harith/Desktop/Open Source/hpx/libs/core/serialization/tests/performance/CMakeLists.txt;25;add_hpx_performance_test;/Users/harith/Desktop/Open Source/hpx/libs/core/serialization/tests/performance/CMakeLists.txt;0;")
