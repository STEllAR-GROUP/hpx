# Install script for directory: /Users/harith/Desktop/Open Source/hpx/libs/core/program_options/examples

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/Library/Developer/CommandLineTools/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/config_file_types")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/config_file_types" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/config_file_types")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/config_file_types")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/config_file_types")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/libs/core/program_options/examples/CMakeFiles/config_file_types.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/custom_syntax")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/custom_syntax" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/custom_syntax")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/custom_syntax")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/custom_syntax")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/libs/core/program_options/examples/CMakeFiles/custom_syntax.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/env_options")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/env_options" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/env_options")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/env_options")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/env_options")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/libs/core/program_options/examples/CMakeFiles/env_options.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/first")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/first" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/first")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/first")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/first")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/libs/core/program_options/examples/CMakeFiles/first.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/multiple_sources")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/multiple_sources" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/multiple_sources")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/multiple_sources")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/multiple_sources")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/libs/core/program_options/examples/CMakeFiles/multiple_sources.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/option_groups")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/option_groups" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/option_groups")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/option_groups")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/option_groups")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/libs/core/program_options/examples/CMakeFiles/option_groups.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/options_description")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/options_description" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/options_description")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/options_description")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/options_description")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/libs/core/program_options/examples/CMakeFiles/options_description.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/options_hierarchy")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/options_hierarchy" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/options_hierarchy")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/options_hierarchy")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/options_hierarchy")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/libs/core/program_options/examples/CMakeFiles/options_hierarchy.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/real")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/real" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/real")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/real")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/real")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/libs/core/program_options/examples/CMakeFiles/real.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/regex")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/regex" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/regex")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/regex")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/regex")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/libs/core/program_options/examples/CMakeFiles/regex.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/response_file")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/response_file" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/response_file")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/response_file")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/response_file")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/libs/core/program_options/examples/CMakeFiles/response_file.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/harith/Desktop/Open Source/hpx/build2/libs/core/program_options/examples/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
