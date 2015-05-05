#-------------------------------------------------------------------------------
# adds hpx_ prefix to give hpx_${name} to libraries and components
#-------------------------------------------------------------------------------
MACRO (hpx_set_lib_name target name)
  # there is no need to change debug/release names explicitly
  # as we use CMAKE_DEBUG_POSTFIX to alter debug names
  
  set_target_properties (${target}
      PROPERTIES
      DEBUG_OUTPUT_NAME          hpx_${name}
      RELEASE_OUTPUT_NAME        hpx_${name}
      MINSIZEREL_OUTPUT_NAME     hpx_${name}
      RELWITHDEBINFO_OUTPUT_NAME hpx_${name}
  )
ENDMACRO()