# This is a dummy target that we add compile flags to all tests will depend on
# this target and inherit the flags but user code linking against hpx will not
add_library(hpx_internal_flags INTERFACE)

# Default unnamed config (not Debug/Release/etc) are in this var
get_property(_temp_flags GLOBAL PROPERTY HPX_CMAKE_FLAGS_CXX_)
target_compile_options(hpx_internal_flags INTERFACE ${_temp_flags})

# Could potentially use CMAKE_CONFIGURATION_TYPES in case a user defined config exists
foreach(_config "DEBUG" "RELEASE" "RELWITHDEBINFO" "MINSIZEREL")
  get_property(_temp_flags GLOBAL PROPERTY HPX_CMAKE_FLAGS_CXX_${_config})
  target_compile_options(hpx_internal_flags INTERFACE $<$<CONFIG:${_config}>:${_temp_flags}>)
endforeach()

foreach(_keyword PUBLIC;PRIVATE)
  get_property(HPX_TARGET_COMPILE_OPTIONS_VAR
    GLOBAL PROPERTY HPX_TARGET_COMPILE_OPTIONS_${_keyword})
  foreach(_flag ${HPX_TARGET_COMPILE_OPTIONS_VAR})
    target_compile_options(hpx_internal_flags INTERFACE ${_flag})
  endforeach()
endforeach()
