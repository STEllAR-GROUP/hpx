# Install script for directory: /Users/harith/Desktop/Open Source/hpx/libs/core/executors/examples

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/disable_thread_stealing_executor")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/disable_thread_stealing_executor" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/disable_thread_stealing_executor")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/disable_thread_stealing_executor")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/disable_thread_stealing_executor")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/libs/core/executors/examples/CMakeFiles/disable_thread_stealing_executor.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/harith/Desktop/Open Source/hpx/build2/bin/executor_with_thread_hooks")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/executor_with_thread_hooks" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/executor_with_thread_hooks")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/harith/Desktop/Open Source/hpx/build2/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/executor_with_thread_hooks")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/executor_with_thread_hooks")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "executables" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/Users/harith/Desktop/Open Source/hpx/build2/libs/core/executors/examples/CMakeFiles/executor_with_thread_hooks.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/harith/Desktop/Open Source/hpx/build2/libs/core/executors/examples/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
