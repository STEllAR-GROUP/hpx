# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Abusive hacks that allow us to have installable targets that are not built
# by default.

set(HPX_INSTALL_LOADED TRUE)

hpx_include(ParseArguments
            TargetPaths)

macro(hpx_symlink source destination)
  if(NOT HPX_NO_INSTALL)
    install(CODE
      "set(symlink_root \"${CMAKE_INSTALL_PREFIX}\")
       execute_process(
         COMMAND \"\${CMAKE_COMMAND}\" -E create_symlink
                 \"${source}\" \"${destination}\"
         WORKING_DIRECTORY \"\${symlink_root}\")")
  endif()
endmacro()

macro(hpx_executable_install name suffix)
  if(NOT HPX_NO_INSTALL)
    hpx_get_target_location(location ${name})

    set(target_directory "${CMAKE_INSTALL_PREFIX}/${suffix}")

    set(install_code
        "file(INSTALL FILES \"${location}\"
              DESTINATION \"${target_directory}\"
              TYPE EXECUTABLE
              OPTIONAL
              PERMISSIONS OWNER_READ OWNER_EXECUTE OWNER_WRITE
                          GROUP_READ GROUP_EXECUTE
                          WORLD_READ WORLD_EXECUTE)")

    hpx_debug("hpx_executable_install.${name}"
      "installing: ${location} to ${target_directory}")
    hpx_debug("hpx_executable_install.${name}"
      "install code: ${install_code}")

    install(CODE "${install_code}")
  endif()
endmacro()

macro(hpx_library_install name suffix)
  if(NOT HPX_NO_INSTALL)
    hpx_get_target_file(lib ${name})
    hpx_get_target_path(output_dir ${name})

    if(UNIX)
      set(targets ${lib} ${lib}.${HPX_SOVERSION} ${lib}.${HPX_VERSION})
    else()
      set(targets ${lib})
    endif()

    set(target_directory "${CMAKE_INSTALL_PREFIX}/${suffix}")
	
	if(MSVC OR MINGW)
		install(TARGETS ${name} ARCHIVE DESTINATION "${target_directory}" OPTIONAL)
	endif()

    foreach(target ${targets})
      set(install_code
        "file(INSTALL FILES \"${output_dir}/${target}\"
              DESTINATION \"${target_directory}\"
              TYPE SHARED_LIBRARY
              OPTIONAL
              PERMISSIONS OWNER_READ OWNER_EXECUTE OWNER_WRITE
                          GROUP_READ GROUP_EXECUTE
                          WORLD_READ WORLD_EXECUTE)")

      hpx_debug("hpx_library_install.${name}"
        "installing: ${output_dir}/${target} to ${target_directory}")
      hpx_debug("hpx_library_install.${name}"
        "install code: ${install_code}")

      install(CODE "${install_code}")
    endforeach()
  endif()
endmacro()

macro(hpx_archive_install name suffix)
  if(NOT HPX_NO_INSTALL)
    hpx_get_target_location(location ${name})

    set(target_directory "${CMAKE_INSTALL_PREFIX}/${suffix}")

    set(install_code
      "file(INSTALL FILES \"${location}\"
            DESTINATION \"${target_directory}\"
            OPTIONAL
            PERMISSIONS OWNER_READ OWNER_READ OWNER_READ
                        GROUP_READ GROUP_READ
                        WORLD_READ WORLD_READ)")

      hpx_debug("hpx_archive_install.${name}"
        "installing: ${location} to ${target_directory}")
      hpx_debug("hpx_archive_install.${name}"
        "install code: ${install_code}")

    install(CODE "${install_code}")
  endif()
endmacro()

macro(hpx_ini_install ini)
  if(NOT HPX_NO_INSTALL)
    set(target_directory "${CMAKE_INSTALL_PREFIX}/share/hpx-${HPX_VERSION}/ini")

    set(install_code
        "file(INSTALL FILES \"${CMAKE_CURRENT_SOURCE_DIR}/${ini}\"
              DESTINATION \"${target_directory}\"
              OPTIONAL
              PERMISSIONS OWNER_READ OWNER_WRITE
                          GROUP_READ
                          WORLD_READ)")

      hpx_debug("hpx_ini_install.${name}"
        "installing: ${CMAKE_CURRENT_SOURCE_DIR}/${ini} to ${target_directory}")
      hpx_debug("hpx_ini_install.${name}"
        "install code: ${install_code}")

    install(CODE "${install_code}")
  endif()
endmacro()

