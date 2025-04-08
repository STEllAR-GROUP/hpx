# CMake generated Testfile for 
# Source directory: /Users/harith/Desktop/Open Source/hpx/examples/balancing
# Build directory: /Users/harith/Desktop/Open Source/hpx/build2/examples/balancing
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(tests.examples.balancing.hpx_thread_phase "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13" "/Users/harith/Desktop/Open Source/hpx/build2/bin/hpxrun.py" "/Users/harith/Desktop/Open Source/hpx/build2/bin/hpx_thread_phase" "-e" "0" "-t" "4" "-l" "1" "-v" "--")
set_tests_properties(tests.examples.balancing.hpx_thread_phase PROPERTIES  _BACKTRACE_TRIPLES "/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;126;add_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;269;add_hpx_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;292;add_test_and_deps_test;/Users/harith/Desktop/Open Source/hpx/examples/balancing/CMakeLists.txt;27;add_hpx_example_test;/Users/harith/Desktop/Open Source/hpx/examples/balancing/CMakeLists.txt;0;")
add_test(tests.examples.balancing.os_thread_num "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13" "/Users/harith/Desktop/Open Source/hpx/build2/bin/hpxrun.py" "/Users/harith/Desktop/Open Source/hpx/build2/bin/os_thread_num" "-e" "0" "-t" "4" "-l" "1" "-v" "--")
set_tests_properties(tests.examples.balancing.os_thread_num PROPERTIES  _BACKTRACE_TRIPLES "/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;126;add_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;269;add_hpx_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;292;add_test_and_deps_test;/Users/harith/Desktop/Open Source/hpx/examples/balancing/CMakeLists.txt;27;add_hpx_example_test;/Users/harith/Desktop/Open Source/hpx/examples/balancing/CMakeLists.txt;0;")
