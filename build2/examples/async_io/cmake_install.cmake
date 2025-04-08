# Install script for directory: /Users/harith/Desktop/Open Source/hpx/examples/async_io

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/async_io_simple")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_io_simple" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_io_simple")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_io_simple")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_io_simple")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/examples/async_io/CMakeFiles/async_io_simple.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/async_io_action")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_io_action" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_io_action")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_io_action")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_io_action")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/examples/async_io/CMakeFiles/async_io_action.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/async_io_external")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_io_external" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_io_external")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_io_external")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_io_external")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/examples/async_io/CMakeFiles/async_io_external.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/async_io_low_level")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_io_low_level" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_io_low_level")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_io_low_level")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/async_io_low_level")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/examples/async_io/CMakeFiles/async_io_low_level.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/harith/Desktop/Open Source/hpx/build2/examples/async_io/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
