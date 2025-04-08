# CMake generated Testfile for 
# Source directory: /Users/harith/Desktop/Open Source/hpx/examples/spell_check
# Build directory: /Users/harith/Desktop/Open Source/hpx/build2/examples/spell_check
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(tests.examples.spell_check.spell_check_simple "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13" "/Users/harith/Desktop/Open Source/hpx/build2/bin/hpxrun.py" "/Users/harith/Desktop/Open Source/hpx/build2/bin/spell_check_simple" "-e" "0" "-t" "4" "-l" "1" "-v" "--")
set_tests_properties(tests.examples.spell_check.spell_check_simple PROPERTIES  _BACKTRACE_TRIPLES "/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;126;add_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;269;add_hpx_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;292;add_test_and_deps_test;/Users/harith/Desktop/Open Source/hpx/examples/spell_check/CMakeLists.txt;32;add_hpx_example_test;/Users/harith/Desktop/Open Source/hpx/examples/spell_check/CMakeLists.txt;0;")
add_test(tests.examples.spell_check.spell_check_file "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13" "/Users/harith/Desktop/Open Source/hpx/build2/bin/hpxrun.py" "/Users/harith/Desktop/Open Source/hpx/build2/bin/spell_check_file" "-e" "0" "-t" "4" "-l" "1" "-v" "--")
set_tests_properties(tests.examples.spell_check.spell_check_file PROPERTIES  _BACKTRACE_TRIPLES "/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;126;add_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;269;add_hpx_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;292;add_test_and_deps_test;/Users/harith/Desktop/Open Source/hpx/examples/spell_check/CMakeLists.txt;32;add_hpx_example_test;/Users/harith/Desktop/Open Source/hpx/examples/spell_check/CMakeLists.txt;0;")
