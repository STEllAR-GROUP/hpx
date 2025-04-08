# CMake generated Testfile for 
# Source directory: /Users/harith/Desktop/Open Source/hpx/tests/regressions
# Build directory: /Users/harith/Desktop/Open Source/hpx/build2/tests/regressions
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(tests.regressions.stack_size_config_4543 "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13" "/Users/harith/Desktop/Open Source/hpx/build2/bin/hpxrun.py" "/Users/harith/Desktop/Open Source/hpx/build2/bin/stack_size_config_4543_test" "-e" "0" "-t" "1" "-l" "1" "-v" "--")
set_tests_properties(tests.regressions.stack_size_config_4543 PROPERTIES  _BACKTRACE_TRIPLES "/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;126;add_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;266;add_hpx_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;282;add_test_and_deps_test;/Users/harith/Desktop/Open Source/hpx/tests/regressions/CMakeLists.txt;36;add_hpx_regression_test;/Users/harith/Desktop/Open Source/hpx/tests/regressions/CMakeLists.txt;0;")
subdirs("build")
subdirs("block_matrix")
subdirs("threads")
subdirs("util")
subdirs("component")
subdirs("lcos")
