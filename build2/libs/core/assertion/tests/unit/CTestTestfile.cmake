# CMake generated Testfile for 
# Source directory: /Users/harith/Desktop/Open Source/hpx/libs/core/assertion/tests/unit
# Build directory: /Users/harith/Desktop/Open Source/hpx/build2/libs/core/assertion/tests/unit
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(tests.unit.modules.assertion.assert_fail "/Users/harith/Desktop/Open Source/hpx/build2/bin/hpxrun.py" "/Users/harith/Desktop/Open Source/hpx/build2/bin/assert_fail_test" "-e" "0" "-t" "1" "-l" "1" "-v" "--")
set_tests_properties(tests.unit.modules.assertion.assert_fail PROPERTIES  WILL_FAIL "" _BACKTRACE_TRIPLES "/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;126;add_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;269;add_hpx_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;277;add_test_and_deps_test;/Users/harith/Desktop/Open Source/hpx/libs/core/assertion/tests/unit/CMakeLists.txt;23;add_hpx_unit_test;/Users/harith/Desktop/Open Source/hpx/libs/core/assertion/tests/unit/CMakeLists.txt;0;")
add_test(tests.unit.modules.assertion.assert_succeed "/Users/harith/Desktop/Open Source/hpx/build2/bin/hpxrun.py" "/Users/harith/Desktop/Open Source/hpx/build2/bin/assert_succeed_test" "-e" "0" "-t" "1" "-l" "1" "-v" "--")
set_tests_properties(tests.unit.modules.assertion.assert_succeed PROPERTIES  _BACKTRACE_TRIPLES "/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;126;add_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;269;add_hpx_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;277;add_test_and_deps_test;/Users/harith/Desktop/Open Source/hpx/libs/core/assertion/tests/unit/CMakeLists.txt;23;add_hpx_unit_test;/Users/harith/Desktop/Open Source/hpx/libs/core/assertion/tests/unit/CMakeLists.txt;0;")
