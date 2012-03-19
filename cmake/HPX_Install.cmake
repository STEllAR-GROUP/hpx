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
      "set(symlink_root \"${HPX_PREFIX}\")
       execute_process(
         COMMAND \"\${CMAKE_COMMAND}\" -E create_symlink
                 \"${source}\" \"${destination}\"
         WORKING_DIRECTORY \"\${symlink_root}\")")
  endif()
endmacro()

macro(hpx_executable_install name)
  if(NOT HPX_NO_INSTALL)
    hpx_get_target_location(location ${name})

    set(install_code
        "file(INSTALL FILES ${location}
              DESTINATION ${HPX_PREFIX}/bin
              TYPE EXECUTABLE
              OPTIONAL
              PERMISSIONS OWNER_READ OWNER_EXECUTE OWNER_WRITE
                          GROUP_READ GROUP_EXECUTE
                          WORLD_READ WORLD_EXECUTE)")
    install(CODE "${install_code}")
  endif()
endmacro()

macro(hpx_library_install name)
  if(NOT HPX_NO_INSTALL)
    hpx_get_target_file(lib ${name})
    hpx_get_target_path(output_dir ${name})

    if(UNIX)
      set(targets ${lib} ${lib}.${HPX_SOVERSION} ${lib}.${HPX_VERSION})
    else()
      set(targets ${lib})
    endif()

    foreach(target ${targets})
      set(install_code
        "file(INSTALL FILES ${output_dir}/${target}
              DESTINATION ${HPX_PREFIX}/lib/hpx
              TYPE SHARED_LIBRARY
              OPTIONAL
              PERMISSIONS OWNER_READ OWNER_EXECUTE OWNER_WRITE
                          GROUP_READ GROUP_EXECUTE
                          WORLD_READ WORLD_EXECUTE)")
      install(CODE "${install_code}")
    endforeach()
  endif()
endmacro()

macro(hpx_archive_install name)
  if(NOT HPX_NO_INSTALL)
    hpx_get_target_location(location ${name})

    set(install_code
      "file(INSTALL FILES ${location}
            DESTINATION ${HPX_PREFIX}/lib/hpx
            OPTIONAL
            PERMISSIONS OWNER_READ OWNER_READ OWNER_READ
                        GROUP_READ GROUP_READ
                        WORLD_READ WORLD_READ)")
    install(CODE "${install_code}")
  endif()
endmacro()

macro(hpx_ini_install name ini)
  if(NOT HPX_NO_INSTALL)
    set(install_code
        "if(EXISTS \"${name}\")
            file(INSTALL FILES ${CMAKE_CURRENT_SOURCE_DIR}/${ini}
                 DESTINATION ${HPX_PREFIX}/share/hpx-${HPX_VERSION}/ini
                 OPTIONAL
                 PERMISSIONS OWNER_READ OWNER_WRITE
                             GROUP_READ
                             WORLD_READ)
         endif()")
    install(CODE "${install_code}")
  endif()
endmacro()

