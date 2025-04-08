# Install script for directory: /Users/harith/Desktop/Open Source/hpx/libs/core/resiliency/examples

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/async_replay")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_replay" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_replay")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_replay")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_replay")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/libs/core/resiliency/examples/CMakeFiles/async_replay.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/async_replicate")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_replicate" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_replicate")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_replicate")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_replicate")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/libs/core/resiliency/examples/CMakeFiles/async_replicate.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/async_replicate_vote")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_replicate_vote" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_replicate_vote")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_replicate_vote")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_replicate_vote")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/libs/core/resiliency/examples/CMakeFiles/async_replicate_vote.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/version")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/version" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/version")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/version")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/version")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/libs/core/resiliency/examples/CMakeFiles/version.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/1d_stencil_replay_exception")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/1d_stencil_replay_exception" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/1d_stencil_replay_exception")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/1d_stencil_replay_exception")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/1d_stencil_replay_exception")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/libs/core/resiliency/examples/CMakeFiles/1d_stencil_replay_exception.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/1d_stencil_replay_validate")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/1d_stencil_replay_validate" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/1d_stencil_replay_validate")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/1d_stencil_replay_validate")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/1d_stencil_replay_validate")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/libs/core/resiliency/examples/CMakeFiles/1d_stencil_replay_validate.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/dataflow_replicate")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/dataflow_replicate" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/dataflow_replicate")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/dataflow_replicate")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/dataflow_replicate")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/libs/core/resiliency/examples/CMakeFiles/dataflow_replicate.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/harith/Desktop/Open Source/hpx/build2/libs/core/resiliency/examples/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
