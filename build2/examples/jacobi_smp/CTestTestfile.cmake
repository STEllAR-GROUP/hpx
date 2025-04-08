# CMake generated Testfile for 
# Source directory: /Users/harith/Desktop/Open Source/hpx/examples/jacobi_smp
# Build directory: /Users/harith/Desktop/Open Source/hpx/build2/examples/jacobi_smp
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(tests.examples.jacobi_smp.jacobi_hpx "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13" "/Users/harith/Desktop/Open Source/hpx/build2/bin/hpxrun.py" "/Users/harith/Desktop/Open Source/hpx/build2/bin/jacobi_hpx" "-e" "0" "-t" "4" "-l" "1" "-v" "--")
set_tests_properties(tests.examples.jacobi_smp.jacobi_hpx PROPERTIES  _BACKTRACE_TRIPLES "/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;126;add_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;269;add_hpx_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;292;add_test_and_deps_test;/Users/harith/Desktop/Open Source/hpx/examples/jacobi_smp/CMakeLists.txt;61;add_hpx_example_test;/Users/harith/Desktop/Open Source/hpx/examples/jacobi_smp/CMakeLists.txt;0;")
