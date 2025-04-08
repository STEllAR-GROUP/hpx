# CMake generated Testfile for 
# Source directory: /Users/harith/Desktop/Open Source/hpx/examples/cancelable_action
# Build directory: /Users/harith/Desktop/Open Source/hpx/build2/examples/cancelable_action
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(tests.examples.cancelable_action.cancelable_action_client "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13" "/Users/harith/Desktop/Open Source/hpx/build2/bin/hpxrun.py" "/Users/harith/Desktop/Open Source/hpx/build2/bin/cancelable_action_client" "-e" "0" "-t" "4" "-l" "1" "-v" "--")
set_tests_properties(tests.examples.cancelable_action.cancelable_action_client PROPERTIES  _BACKTRACE_TRIPLES "/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;126;add_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;269;add_hpx_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;292;add_test_and_deps_test;/Users/harith/Desktop/Open Source/hpx/examples/cancelable_action/CMakeLists.txt;32;add_hpx_example_test;/Users/harith/Desktop/Open Source/hpx/examples/cancelable_action/CMakeLists.txt;0;")
subdirs("cancelable_action")
