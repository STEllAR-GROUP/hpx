#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "HPX::component_storage_component" for configuration "Release"
set_property(TARGET HPX::component_storage_component APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(HPX::component_storage_component PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhpx_component_storage.1.11.0.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libhpx_component_storage.1.dylib"
  )

list(APPEND _cmake_import_check_targets HPX::component_storage_component )
list(APPEND _cmake_import_check_files_for_HPX::component_storage_component "${_IMPORT_PREFIX}/lib/libhpx_component_storage.1.11.0.dylib" )

# Import target "HPX::unordered_component" for configuration "Release"
set_property(TARGET HPX::unordered_component APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(HPX::unordered_component PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhpx_unordered.1.11.0.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libhpx_unordered.1.dylib"
  )

list(APPEND _cmake_import_check_targets HPX::unordered_component )
list(APPEND _cmake_import_check_files_for_HPX::unordered_component "${_IMPORT_PREFIX}/lib/libhpx_unordered.1.11.0.dylib" )

# Import target "HPX::partitioned_vector_component" for configuration "Release"
set_property(TARGET HPX::partitioned_vector_component APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(HPX::partitioned_vector_component PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhpx_partitioned_vector.1.11.0.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libhpx_partitioned_vector.1.dylib"
  )

list(APPEND _cmake_import_check_targets HPX::partitioned_vector_component )
list(APPEND _cmake_import_check_files_for_HPX::partitioned_vector_component "${_IMPORT_PREFIX}/lib/libhpx_partitioned_vector.1.11.0.dylib" )

# Import target "HPX::iostreams_component" for configuration "Release"
set_property(TARGET HPX::iostreams_component APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(HPX::iostreams_component PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhpx_iostreams.1.11.0.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libhpx_iostreams.1.dylib"
  )

list(APPEND _cmake_import_check_targets HPX::iostreams_component )
list(APPEND _cmake_import_check_files_for_HPX::iostreams_component "${_IMPORT_PREFIX}/lib/libhpx_iostreams.1.11.0.dylib" )

# Import target "HPX::parcel_coalescing" for configuration "Release"
set_property(TARGET HPX::parcel_coalescing APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(HPX::parcel_coalescing PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/hpx/libhpx_parcel_coalescing.1.11.0.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libhpx_parcel_coalescing.1.dylib"
  )

list(APPEND _cmake_import_check_targets HPX::parcel_coalescing )
list(APPEND _cmake_import_check_files_for_HPX::parcel_coalescing "${_IMPORT_PREFIX}/lib/hpx/libhpx_parcel_coalescing.1.11.0.dylib" )

# Import target "HPX::memory_counters_component" for configuration "Release"
set_property(TARGET HPX::memory_counters_component APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(HPX::memory_counters_component PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/hpx/libhpx_memory_counters.1.11.0.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libhpx_memory_counters.1.dylib"
  )

list(APPEND _cmake_import_check_targets HPX::memory_counters_component )
list(APPEND _cmake_import_check_files_for_HPX::memory_counters_component "${_IMPORT_PREFIX}/lib/hpx/libhpx_memory_counters.1.11.0.dylib" )

# Import target "HPX::process_component" for configuration "Release"
set_property(TARGET HPX::process_component APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(HPX::process_component PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhpx_process.1.11.0.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libhpx_process.1.dylib"
  )

list(APPEND _cmake_import_check_targets HPX::process_component )
list(APPEND _cmake_import_check_files_for_HPX::process_component "${_IMPORT_PREFIX}/lib/libhpx_process.1.11.0.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
