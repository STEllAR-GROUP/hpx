# CMake generated Testfile for 
# Source directory: /Users/harith/Desktop/Open Source/hpx/components/parcel_plugins/coalescing/tests/unit
# Build directory: /Users/harith/Desktop/Open Source/hpx/build2/components/parcel_plugins/coalescing/tests/unit
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(tests.unit.components.parcel_plugins.coalescing.distributed.tcp.put_parcels_with_coalescing "/Users/harith/Desktop/Open Source/hpx/build2/bin/hpxrun.py" "/Users/harith/Desktop/Open Source/hpx/build2/bin/put_parcels_with_coalescing_test" "-e" "0" "-t" "1" "-l" "2" "-p" "tcp" "-v" "--")
set_tests_properties(tests.unit.components.parcel_plugins.coalescing.distributed.tcp.put_parcels_with_coalescing PROPERTIES  RUN_SERIAL "TRUE" _BACKTRACE_TRIPLES "/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;227;add_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;269;add_hpx_test;/Users/harith/Desktop/Open Source/hpx/cmake/HPX_AddTest.cmake;277;add_test_and_deps_test;/Users/harith/Desktop/Open Source/hpx/components/parcel_plugins/coalescing/tests/unit/CMakeLists.txt;28;add_hpx_unit_test;/Users/harith/Desktop/Open Source/hpx/components/parcel_plugins/coalescing/tests/unit/CMakeLists.txt;0;")
