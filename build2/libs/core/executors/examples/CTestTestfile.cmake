# CMake generated Testfile for 
# Source directory: /Users/harith/Desktop/Open Source/hpx/libs/core/executors/examples
# Build directory: /Users/harith/Desktop/Open Source/hpx/build2/libs/core/executors/examples
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(tests.examples.modules.executors.disable_thread_stealing_executor "/Users/harith/Desktop/Open Source/hpx/build2/bin/hpxrun.py" "/Users/harith/Desktop/Open Source/hpx/build2/bin/disable_thread_stealing_executor" "-e" "0" "-t" "1" "-l" "1" "-v" "--")
set_tests_properties(tests.examples.modules.executors.disable_thread_stealing_executor PROPERTIES  _BACKTRACE_TRIPLES "/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;126;add_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;269;add_hpx_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;292;add_test_and_deps_test;/Users/harith/Desktop/Open Source/hpx/libs/core/executors/examples/CMakeLists.txt;39;add_hpx_example_test;/Users/harith/Desktop/Open Source/hpx/libs/core/executors/examples/CMakeLists.txt;0;")
add_test(tests.examples.modules.executors.executor_with_thread_hooks "/Users/harith/Desktop/Open Source/hpx/build2/bin/hpxrun.py" "/Users/harith/Desktop/Open Source/hpx/build2/bin/executor_with_thread_hooks" "-e" "0" "-t" "1" "-l" "1" "-v" "--")
set_tests_properties(tests.examples.modules.executors.executor_with_thread_hooks PROPERTIES  _BACKTRACE_TRIPLES "/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;126;add_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;269;add_hpx_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;292;add_test_and_deps_test;/Users/harith/Desktop/Open Source/hpx/libs/core/executors/examples/CMakeLists.txt;39;add_hpx_example_test;/Users/harith/Desktop/Open Source/hpx/libs/core/executors/examples/CMakeLists.txt;0;")
