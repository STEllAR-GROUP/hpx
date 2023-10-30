# Copyright (c) 2019-2023 Ste||ar Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_Message)

macro(hpx_setup_openshmem)

  if(NOT TARGET PkgConfig::OPENSHMEM)

    set(OPENSHMEM_PC "")
    if(HPX_WITH_PARCELPORT_MPI
       AND HPX_WITH_PARCELPORT_OPENSHMEM
       AND (NOT ("${HPX_WITH_PARCELPORT_OPENSHMEM_CONDUIT}" STREQUAL "mpi"))
    )
      message(
        FATAL
        "HPX_WITH_PARCELPORT_MPI & HPX_WITH_PARCELPORT_OPENSHMEM both set to ON and HPX_WITH_PARCELPORT_OPENSHMEM_CONDUIT != 'mpi'"
      )

    elseif(
      HPX_WITH_PARCELPORT_MPI
      AND HPX_WITH_PARCELPORT_OPENSHMEM
      AND HPX_WITH_FETCH_OPENSHMEM
      AND ("${HPX_WITH_PARCELPORT_OPENSHMEM_CONDUIT}" STREQUAL "mpi")
    )
      message(
        FATAL_ERROR
          "HPX_WITH_FETCH_OPENSHMEM for HPX_WITH_PARCELPORT_OPENSHMEM and HPX_WITH_PARCELPORT_OPENSHMEM_CONDUIT='mpi' not supported"
      )

    elseif(
      HPX_WITH_PARCELPORT_OPENSHMEM
      AND (NOT HPX_WITH_FETCH_OPENSHMEM)
      AND ("${HPX_WITH_PARCELPORT_OPENSHMEM_CONDUIT}" STREQUAL "mpi")
    )

      set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${MPI_LIBDIR}/pkgconfig")

      set(OPENSHMEM_PC "oshmem")
      pkg_search_module(OPENSHMEM IMPORTED_TARGET GLOBAL ${OPENSHMEM_PC})

      if(NOT OPENSHMEM_FOUND)
        find_program(OSHMEM_INFO NAMES oshmem_info ompi_info REQUIRED)

        if(NOT OSHMEM_INFO)
          message(
            FATAL_ERROR
              "oshmem_info and/or ompi_info not found! pkg-config cannot find OpenMPI's `${OPENSHMEM_PC}.pc`"
          )
        endif()

        set(OSHMEM_INFO_OUTPUT
            "${CMAKE_CURRENT_SOURCE_DIR}/oshmem_info_stdout.log"
        )
        set(OSHMEM_INFO_ERROR
            "${CMAKE_CURRENT_SOURCE_DIR}/oshmem_info_error.log"
        )

        execute_process(
          COMMAND bash -c "${OSHMEM_INFO} --path libdir"
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          RESULT_VARIABLE OSHMEM_INFO_STATUS
          OUTPUT_FILE ${OSHMEM_INFO_OUTPUT}
          ERROR_FILE ${OSHMEM_INFO_ERROR}
        )

        if(NOT OSHMEM_INFO_STATUS)
          message(
            FATAL_ERROR
              "${OSHMEM_INFO} Failed! Program status code: ${OSHMEM_INFO_STATUS}"
          )
        endif()

        file(READ ${OSHMEM_INFO_OUTPUT} OSHMEM_INFO_OUTPUT_CONTENT)

        if(NOT OSHMEM_INFO_OUTPUT_CONTENT)
          message(
            FATAL_ERROR "${OSHMEM_INFO} Failed! Check: ${OSHMEM_INFO_ERROR}"
          )
        endif()

        string(REGEX MATCH "(\/.*)" OSHMEM_LIBDIR_PATH
                     ${OSHMEM_INFO_OUTPUT_CONTENT}
        )

        set(ENV{PKG_CONFIG_PATH} "${OSHMEM_INFO_OUTPUT_CONTENT}/pkgconfig")

        pkg_search_module(OPENSHMEM IMPORTED_TARGET GLOBAL oshmem)

        if(NOT OPENSHMEM_FOUND)
          message(
            FATAL_ERROR
              "oshmem not found for HPX_WITH_PARCELPORT_OPENSHMEM and HPX_WITH_PARCELPORT_OPENSHMEM_CONDUIT='mpi'"
          )
        endif()
      endif()

      set_target_properties(
        PkgConfig::OPENSHMEM PROPERTIES INTERFACE_COMPILE_OPTIONS
                                        "${OPENSHMEM_CFLAGS}"
      )
      set_target_properties(
        PkgConfig::OPENSHMEM PROPERTIES INTERFACE_LINK_OPTIONS
                                        "${OPENSHMEM_LDFLAGS}"
      )
      set_target_properties(
        PkgConfig::OPENSHMEM PROPERTIES INTERFACE_LINK_DIRECTORIES
                                        "${OPENSHMEM_LIBRARY_DIRS}"
      )

      return()

    elseif("${HPX_WITH_PARCELPORT_OPENSHMEM_CONDUIT}" STREQUAL "ucx")
      set(OPENSHMEM_PC "ucx")

      pkg_search_module(UCX IMPORTED_TARGET GLOBAL ${OPENSHMEM_PC})

      if(NOT UCX_FOUND)
        message(
          FATAL
          "HPX_WITH_PARCELPORT_OPENSHMEM=ucx selected but UCX is unavailable"
        )
      endif()

      pkg_search_module(OPENSHMEM IMPORTED_TARGET GLOBAL osss-ucx)
    elseif("${HPX_WITH_PARCELPORT_OPENSHMEM_CONDUIT}" STREQUAL "sos")
      set(OPENSHMEM_PC "sandia-openshmem")

      pkg_search_module(OPENSHMEM IMPORTED_TARGET GLOBAL ${OPENSHMEM_PC})
    endif()
  endif()

  if((NOT OPENSHMEM_FOUND)
     AND HPX_WITH_FETCH_OPENSHMEM
     AND NOT ("${HPX_WITH_PARCELPORT_OPENSHMEM_CONDUIT}" STREQUAL "mpi")
  )
    if(NOT CMAKE_C_COMPILER)
      message(
        FATAL_ERROR
          "HPX_WITH_FETCH_OPENSHMEM requires `-DCMAKE_C_COMPILER` to be set; CMAKE_C_COMPILER is currently unset."
      )
    endif()
    if(NOT CMAKE_CXX_COMPILER)
      message(
        FATAL_ERROR
          "HPX_WITH_FETCH_OPENSHMEM requires `-DCMAKE_CXX_COMPILER` to be set; CMAKE_CXX_COMPILER is currently unset."
      )
    endif()

    find_program(MAKE_EXECUTABLE NAMES gmake make mingw32-make REQUIRED)

    include(FindOpenShmemPmi)

    set(PMI_AUTOCONF_OPTS "")
    if(NOT PMI_LIBRARY OR NOT PMI_FOUND)
      set(PMI_AUTOCONF_OPTS "--enable-pmi-simple")
    else()
      string(REGEX MATCH "(^\/[^\/]+)" PMI_INCLUDE_DIR_ROOT_PATH
                   ${PMI_INCLUDE_DIR}
      )
      string(REGEX MATCH "(^\/[^\/]+)" PMI_LIBRARY_ROOT_PATH ${PMI_LIBRARY})
      set(PMI_AUTOCONF_OPTS
          "--with-pmi=${PMI_INCLUDE_DIR_ROOT_PATH} --with-pmi-libdir=${PMI_LIBRARY_ROOT_PATH}"
      )
    endif()

    include(FetchContent)

    if("${HPX_WITH_PARCELPORT_OPENSHMEM_CONDUIT}" STREQUAL "ucx")

      message(STATUS "Fetching OSSS-UCX-OpenSHMEM")

      fetchcontent_declare(
        openshmem
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        URL https://github.com/openshmem-org/osss-ucx/archive/refs/tags/v1.0.2.tar.gz
      )

      message(
        STATUS
          "Building OSSS-UCX (OpenSHMEM on UCX) and installing into ${CMAKE_INSTALL_PREFIX}"
      )

    elseif("${HPX_WITH_PARCELPORT_OPENSHMEM_CONDUIT}" STREQUAL "sos")

      message(STATUS "Fetching Sandia-OpenSHMEM")

      fetchcontent_declare(
        openshmem
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        URL https://github.com/Sandia-OpenSHMEM/SOS/archive/refs/tags/v1.5.2.tar.gz
      )

      message(
        STATUS
          "Building and installing Sandia OpenSHMEM into ${CMAKE_INSTALL_PREFIX}"
      )

    else()
      message(
        FATAL_ERROR "HPX_WITH_PARCELPORT_OPENSHMEM is not set to `ucx` or `sos`"
      )
    endif()

    fetchcontent_getproperties(openshmem)
    if(NOT openshmem)
      fetchcontent_populate(openshmem)
    endif()

    set(OPENSHMEM_DIR "${openshmem_SOURCE_DIR}")
    set(OPENSHMEM_BUILD_OUTPUT "${OPENSHMEM_DIR}/build.log")
    set(OPENSHMEM_ERROR_FILE "${OPENSHMEM_DIR}/error.log")

    set(ENV{PKG_CONFIG_PATH} "${OPENSHMEM_DIR}/install/lib/pkgconfig")

    execute_process(
      COMMAND
        bash -c
        "./autogen.sh && CC=${CMAKE_C_COMPILER} ./configure --prefix=${OPENSHMEM_DIR}/install --enable-shared --disable-fortran ${PMI_AUTOCONF_OPTS} && make && make install"
      WORKING_DIRECTORY ${OPENSHMEM_DIR}
      RESULT_VARIABLE OPENSHMEM_BUILD_STATUS
      OUTPUT_FILE ${OPENSHMEM_BUILD_OUTPUT}
      ERROR_FILE ${OPENSHMEM_ERROR_FILE}
    )

    if(OPENSHMEM_BUILD_STATUS)
      file(READ ${OPENSHMEM_BUILD_OUTPUT} OPENSHMEM_BUILD_OUTPUT_CONTENT)

      message(
        FATAL_ERROR
          "OpenSHMEM build result = ${OPENSHMEM_BUILD_STATUS} - see ${OPENSHMEM_BUILD_OUTPUT} for more details\nOpenSHMEM build command\t:./autogen.sh && CC=${CMAKE_C_COMPILER} ./configure --prefix=${OPENSHMEM_DIR}/install --enable-shared --disable-fortran ${PMI_AUTOCONF_OPTS} && make && make install\n${OPENSHMEM_BUILD_OUTPUT_CONTENT}"
      )
    else()

      find_file(OPENSHMEM_PKGCONFIG_FILE_FOUND ${OPENSHMEM_PC}.pc
                ${OPENSHMEM_DIR}/install/lib/pkgconfig
      )

      if(NOT OPENSHMEM_PKGCONFIG_FILE_FOUND)
        message(
          FATAL_ERROR
            "PKG-CONFIG ERROR (${OPENSHMEM_PKGCONFIG_FILE_FOUND}; ${OPENSHMEM_PC}) -> CANNOT FIND COMPILED OpenSHMEM: ${OPENSHMEM_DIR}/install/lib/pkgconfig"
        )
      endif()

      install(CODE "set(OPENSHMEM_PKGCONF \"${OPENSHMEM_PC}\")")
      install(CODE "set(OPENSHMEM_PATH \"${OPENSHMEM_DIR}\")")

      install(
        CODE [[
          file(READ
            ${OPENSHMEM_PATH}/install/lib/pkgconfig/${OPENSHMEM_PKGCONF}.pc
	    OPENSHMEM_PKGCONFIG_FILE_CONTENT)

          if(NOT OPENSHMEM_PKGCONFIG_FILE_CONTENT)
            message(FATAL_ERROR "ERROR INSTALLING OPENSHMEM")
          endif()

	  string(REPLACE "${OPENSHMEM_PATH}/install" "${CMAKE_INSTALL_PREFIX}"
                         OPENSHMEM_PKGCONFIG_FILE_CONTENT
                         ${OPENSHMEM_PKGCONFIG_FILE_CONTENT})

          file(WRITE
            ${OPENSHMEM_PATH}/install/lib/pkgconfig/${OPENSHMEM_PKGCONF}.pc
            ${OPENSHMEM_PKGCONFIG_FILE_CONTENT}
          )

          file(GLOB_RECURSE OPENSHMEM_FILES ${OPENSHMEM_PATH}/install/*)

	  if(NOT OPENSHMEM_FILES)
           message(STATUS "ERROR INSTALLING OPENSHMEM")
          endif()

          foreach(OPENSHMEM_FILE ${OPENSHMEM_FILES})
            set(OPENSHMEM_FILE_CACHED "${OPENSHMEM_FILE}")

            string(REGEX MATCH "(^\/.*\/)" OPENSHMEM_FILE_PATH ${OPENSHMEM_FILE})

            string(REPLACE "${OPENSHMEM_PATH}/install" "${CMAKE_INSTALL_PREFIX}"
                           OPENSHMEM_FILE ${OPENSHMEM_FILE}
            )

            string(REPLACE "${OPENSHMEM_PATH}/install" "${CMAKE_INSTALL_PREFIX}"
                           OPENSHMEM_FILE_PATH ${OPENSHMEM_FILE_PATH}
            )

            file(MAKE_DIRECTORY ${OPENSHMEM_FILE_PATH})

            string(LENGTH ${OPENSHMEM_FILE_PATH} OPENSHMEM_FILE_PATH_SIZE)
            math(EXPR OPENSHMEM_FILE_PATH_SIZE "${OPENSHMEM_FILE_PATH_SIZE}-1")

            string(SUBSTRING ${OPENSHMEM_FILE_PATH} 0 ${OPENSHMEM_FILE_PATH_SIZE}
                             OPENSHMEM_FILE_PATH)

            file(COPY ${OPENSHMEM_FILE_CACHED} DESTINATION ${OPENSHMEM_FILE_PATH})
          endforeach()
        ]]
      )
    endif()

    set(CMAKE_PREFIX_PATH "${OPENSHMEM_DIR}/install/lib/pkgconfig")
    set(ENV{PKG_CONFIG_PATH} "${OPENSHMEM_DIR}/install/lib/pkgconfig")

    pkg_search_module(OPENSHMEM REQUIRED IMPORTED_TARGET GLOBAL ${OPENSHMEM_PC})

    if(NOT OPENSHMEM_FOUND)
      message(
        FATAL_ERROR
          "OpenSHMEM downloaded, compiled, but cannot be found in ${CMAKE_INSTALL_PREFIX}"
      )
    endif()
  elseif((NOT OPENSHMEM_FOUND) AND (NOT HPX_WITH_FETCH_OPENSHMEM))
    message(
      FATAL_ERROR "OpenSHMEM not found and HPX_WITH_FETCH_OPENSHMEM not set!"
    )
  endif()

  if(OPENSHMEM_CFLAGS)
    set(IS_PARAM "0")
    set(PARAM_FOUND "0")
    set(NEWPARAM "")
    set(IDX 0)
    set(FLAG_LIST "")

    foreach(X IN ITEMS ${OPENSHMEM_CFLAGS})
      string(FIND "${X}" "--param" PARAM_FOUND)
      if(NOT "${PARAM_FOUND}" EQUAL "-1")
        set(IS_PARAM "1")
        set(NEWPARAM "SHELL:${X}")
      endif()
      if("${PARAM_FOUND}" EQUAL "-1"
         AND "${IS_PARAM}" EQUAL "0"
         OR "${IS_PARAM}" EQUAL "-1"
      )
        list(APPEND FLAG_LIST "${X}")
        set(IS_PARAM "0")
      elseif("${PARAM_FOUND}" EQUAL "-1" AND "${IS_PARAM}" EQUAL "1")
        list(APPEND FLAG_LIST "${NEWPARAM}
          ${X}"
        )
        set(NEWPARAM "")
        set(IS_PARAM "0")
      endif()
    endforeach()

    list(LENGTH OPENSHMEM_CFLAGS IDX)
    foreach(X RANGE ${IDX})
      list(POP_FRONT OPENSHMEM_CFLAGS NEWPARAM)
    endforeach()

    foreach(X IN ITEMS ${FLAG_LIST})
      list(APPEND OPENSHMEM_CFLAGS "${X}")
    endforeach()
  endif()

  if(OPENSHMEM_CFLAGS_OTHER)
    set(IS_PARAM "0")
    set(PARAM_FOUND "0")
    set(NEWPARAM "")
    set(IDX 0)
    set(FLAG_LIST "")

    foreach(X IN ITEMS ${OPENSHMEM_CFLAGS_OTHER})
      string(FIND "${X}" "--param" PARAM_FOUND)
      if(NOT "${PARAM_FOUND}" EQUAL "-1")
        set(IS_PARAM "1")
        set(NEWPARAM "SHELL:${X}")
      endif()
      if("${PARAM_FOUND}" EQUAL "-1"
         AND "${IS_PARAM}" EQUAL "0"
         OR "${IS_PARAM}" EQUAL "-1"
      )
        list(APPEND FLAG_LIST "${X}")
        set(IS_PARAM "0")
      elseif("${PARAM_FOUND}" EQUAL "-1" AND "${IS_PARAM}" EQUAL "1")
        list(APPEND FLAG_LIST "${NEWPARAM}
          ${X}"
        )
        set(NEWPARAM "")
        set(IS_PARAM "0")
      endif()
    endforeach()

    list(LENGTH OPENSHMEM_CFLAGS_OTHER IDX)
    foreach(X RANGE ${IDX})
      list(POP_FRONT OPENSHMEM_CFLAGS_OTHER NEWPARAM)
    endforeach()

    foreach(X IN ITEMS ${FLAG_LIST})
      list(APPEND OPENSHMEM_CFLAGS_OTHER "${X}")
    endforeach()
  endif()

  if(OPENSHMEM_LDFLAGS)
    set(IS_PARAM "0")
    set(PARAM_FOUND "0")
    set(NEWPARAM "")
    set(IDX 0)
    set(DIRIDX 0)
    set(SKIP 0)
    set(FLAG_LIST "")
    set(DIR_LIST "")
    set(LIB_LIST "")

    foreach(X IN ITEMS ${OPENSHMEM_LDFLAGS})
      string(FIND "${X}" "--param" PARAM_FOUND)
      string(FIND "${X}" "-lsma" IDX)
      string(FIND "${X}" "-l" LIDX)
      string(FIND "${X}" "-L" DIRIDX)
      string(FIND "${X}" "-Wl" SKIP)

      if("${SKIP}" EQUAL "-1")
        if(NOT "${PARAM_FOUND}" EQUAL "-1")
          set(IS_PARAM "1")
          set(NEWPARAM "SHELL:${X}")
        endif()
        if("${PARAM_FOUND}" EQUAL "-1"
           AND "${IDX}" EQUAL "-1"
           AND "${IS_PARAM}" EQUAL "0"
           OR "${IS_PARAM}" EQUAL "-1"
        )
          list(APPEND FLAG_LIST "${X}")
          set(IS_PARAM "0")
        elseif("${PARAM_FOUND}" EQUAL "-1" AND "${IS_PARAM}" EQUAL "1")
          list(APPEND FLAG_LIST "${NEWPARAM}
          ${X}"
          )
          set(NEWPARAM "")
          set(IS_PARAM "0")
        elseif(NOT "${IDX}" EQUAL "-1" AND NOT "${LIDX}" EQUAL "-1")
          set(TMPSTR "")
          string(REPLACE "-l" "" TMPSTR "${X}")
          list(APPEND LIB_LIST "${TMPSTR}")
          set(IDX 0)
        elseif("${IDX}" EQUAL "-1" AND NOT "${LIDX}" EQUAL "-1")
          list(APPEND FLAG_LIST "${X}")
        endif()
        if(NOT "${DIRIDX}" EQUAL "-1")
          set(TMPSTR "")
          string(REPLACE "-L" "" TMPSTR "${X}")
          list(APPEND DIR_LIST "${TMPSTR}")
        endif()
      endif()
    endforeach()

    set(IDX 0)
    list(LENGTH OPENSHMEM_LDFLAGS IDX)
    foreach(X RANGE ${IDX})
      list(POP_FRONT OPENSHMEM_LDFLAGS NEWPARAM)
    endforeach()

    foreach(X IN ITEMS ${FLAG_LIST})
      list(APPEND OPENSHMEM_LDFLAGS "${X}")
    endforeach()

    set(IDX 0)
    list(LENGTH LIB_LIST IDX)
    if(NOT "${IDX}" EQUAL "0")
      set(IDX 0)

      if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(NEWLINK "SHELL:-Wl,--whole-archive
          "
        )
        foreach(X IN ITEMS ${LIB_LIST})
          set(DIRSTR "")
          string(REPLACE ";" "
          " DIRSTR "${DIR_LIST}"
          )
          foreach(Y IN ITEMS ${DIR_LIST})
            find_library(
              FOUND_LIB
              NAMES ${X} "lib${X}" "lib${X}.a"
              PATHS ${Y}
              HINTS ${Y} NO_CACHE
              NO_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH
            )

            list(LENGTH FOUND_LIB IDX)
            if(NOT "${IDX}" EQUAL "0")
              string(APPEND NEWLINK "${FOUND_LIB}")
              set(FOUND_LIB "")
            endif()
          endforeach()
        endforeach()
        string(APPEND NEWLINK "
          -Wl,--no-whole-archive"
        )
        string(FIND "SHELL:-Wl,--whole-archive
          -Wl,--no-whole-archive" "${NEWLINK}" IDX
        )
        if("${IDX}" EQUAL "-1")
          list(APPEND OPENSHMEM_LDFLAGS "${NEWLINK}")
        endif()
      elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(APPLE)
          set(NEWLINK "SHELL:-Wl,-force_load,")
        else()
          set(NEWLINK "SHELL:
          "
          )
        endif()
        foreach(X IN ITEMS ${LIB_LIST})
          set(DIRSTR "")
          string(REPLACE ";" "
          " DIRSTR "${DIR_LIST}"
          )
          foreach(Y IN ITEMS ${DIR_LIST})
            find_library(
              FOUND_LIB
              NAMES ${X} "lib${X}" "lib${X}.a"
              PATHS ${Y}
              HINTS ${Y} NO_CACHE
              NO_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH
            )

            list(LENGTH FOUND_LIB IDX)
            if(NOT "${IDX}" EQUAL "0")
              string(APPEND NEWLINK "${FOUND_LIB}")
              set(FOUND_LIB "")
            endif()
          endforeach()
        endforeach()
        string(FIND "SHELL:" "${NEWLINK}" IDX)
        if("${IDX}" EQUAL "-1")
          list(APPEND OPENSHMEM_LDFLAGS "${NEWLINK}")
        endif()
      endif()
    endif()
  endif()

  if(OPENSHMEM_LDFLAGS_OTHER)
    unset(FOUND_LIB)
    set(IS_PARAM "0")
    set(PARAM_FOUND "0")
    set(NEWPARAM "")
    set(SKIP 0)
    set(IDX 0)
    set(DIRIDX 0)
    set(FLAG_LIST "")
    set(DIR_LIST "")
    set(LIB_LIST "")

    foreach(X IN ITEMS ${OPENSHMEM_LDFLAGS_OTHER})
      string(FIND "${X}" "--param" PARAM_FOUND)
      string(FIND "${X}" "-lsma" IDX)
      string(FIND "${X}" "-L" DIRIDX)
      string(FIND "${X}" "-Wl" SKIP)

      if("${SKIP}" EQUAL "-1")
        if(NOT "${PARAM_FOUND}" EQUAL "-1")
          set(IS_PARAM "1")
          set(NEWPARAM "SHELL:${X}")
        endif()
        if("${PARAM_FOUND}" EQUAL "-1"
           AND "${IDX}" EQUAL "-1"
           AND "${IS_PARAM}" EQUAL "0"
           OR "${IS_PARAM}" EQUAL "-1"
        )
          list(APPEND FLAG_LIST "${X}")
          set(IS_PARAM "0")
        elseif("${PARAM_FOUND}" EQUAL "-1" AND "${IS_PARAM}" EQUAL "1")
          list(APPEND FLAG_LIST "${NEWPARAM}
          ${X}"
          )
          set(NEWPARAM "")
          set(IS_PARAM "0")
        elseif(NOT "${IDX}" EQUAL "-1" AND NOT "${LIDX}" EQUAL "-1")
          set(TMPSTR "")
          string(REPLACE "-l" "" TMPSTR "${X}")
          list(APPEND LIB_LIST "${TMPSTR}")
          set(IDX 0)
        elseif("${IDX}" EQUAL "-1" AND NOT "${LIDX}" EQUAL "-1")
          list(APPEND FLAG_LIST "${X}")
        endif()
        if(NOT "${DIRIDX}" EQUAL "-1")
          set(TMPSTR "")
          string(REPLACE "-L" "" TMPSTR "${X}")
          list(APPEND DIR_LIST "${TMPSTR}")
        endif()
      endif()
    endforeach()

    set(IDX 0)
    list(LENGTH OPENSHMEM_LDFLAGS_OTHER IDX)
    foreach(X RANGE ${IDX})
      list(POP_FRONT OPENSHMEM_LDFLAGS_OTHER NEWPARAM)
    endforeach()

    foreach(X IN ITEMS ${FLAG_LIST})
      list(APPEND OPENSHMEM_LDFLAGS_OTHER "${X}")
    endforeach()

    set(IDX 0)
    list(LENGTH LIB_LIST IDX)
    if(NOT "${IDX}" EQUAL "0")
      set(IDX 0)
      if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(NEWLINK "SHELL:-Wl,--whole-archive
          "
        )
        foreach(X IN ITEMS ${LIB_LIST})
          set(DIRSTR "")
          string(REPLACE ";" "
          " DIRSTR "${DIR_LIST}"
          )
          foreach(Y IN ITEMS ${DIR_LIST})
            find_library(
              FOUND_LIB
              NAMES ${X} "lib${X}" "lib${X}.a"
              PATHS ${Y}
              HINTS ${Y} NO_CACHE
              NO_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH
            )

            list(LENGTH FOUND_LIB IDX)
            if(NOT "${IDX}" EQUAL "0")
              string(APPEND NEWLINK "${FOUND_LIB}")
              set(FOUND_LIB "")
            endif()
          endforeach()
        endforeach()
        string(APPEND NEWLINK "
          -Wl,--no-whole-archive"
        )

        string(FIND "SHELL:-Wl,--whole-archive
          -Wl,--no-whole-archive" "${NEWLINK}" IDX
        )
        if("${IDX}" EQUAL "-1")
          list(APPEND OPENSHMEM_LDFLAGS_OTHER "${NEWLINK}")
        endif()
      elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(APPLE)
          set(NEWLINK "SHELL:-Wl,-force_load,")
        else()
          set(NEWLINK "SHELL:
          "
          )
        endif()
        foreach(X IN ITEMS ${LIB_LIST})
          set(DIRSTR "")
          string(REPLACE ";" "
          " DIRSTR "${DIR_LIST}"
          )
          foreach(Y IN ITEMS ${DIR_LIST})
            find_library(
              FOUND_LIB
              NAMES ${X} "lib${X}" "lib${X}.a"
              PATHS ${Y}
              HINTS ${Y} NO_CACHE
              NO_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH
            )

            list(LENGTH FOUND_LIB IDX)
            if(NOT "${IDX}" EQUAL "0")
              string(APPEND NEWLINK "${FOUND_LIB}")
              set(FOUND_LIB "")
            endif()
          endforeach()
        endforeach()
        string(FIND "SHELL:" "${NEWLINK}" IDX)
        if("${IDX}" EQUAL "-1")
          list(APPEND OPENSHMEM_LDFLAGS "${NEWLINK}")
        endif()
      endif()
    endif()

  endif()

  if(OPENSHMEM_STATIC_CFLAGS)
    set(IS_PARAM "0")
    set(PARAM_FOUND "0")
    set(NEWPARAM "")
    set(IDX 0)
    set(FLAG_LIST "")

    foreach(X IN ITEMS ${OPENSHMEM_STATIC_CFLAGS})
      string(FIND "${X}" "--param" PARAM_FOUND)
      if(NOT "${PARAM_FOUND}" EQUAL "-1")
        set(IS_PARAM "1")
        set(NEWPARAM "SHELL:${X}")
      endif()
      if("${PARAM_FOUND}" EQUAL "-1"
         AND "${IS_PARAM}" EQUAL "0"
         OR "${IS_PARAM}" EQUAL "-1"
      )
        list(APPEND FLAG_LIST "${X}")
        set(IS_PARAM "0")
      elseif("${PARAM_FOUND}" EQUAL "-1" AND "${IS_PARAM}" EQUAL "1")
        list(APPEND FLAG_LIST "${NEWPARAM}
          ${X}"
        )
        set(NEWPARAM "")
        set(IS_PARAM "0")
      endif()
    endforeach()

    list(LENGTH OPENSHMEM_STATIC_CFLAGS IDX)
    foreach(X RANGE ${IDX})
      list(POP_FRONT OPENSHMEM_STATIC_CFLAGS NEWPARAM)
    endforeach()

    foreach(X IN ITEMS ${FLAG_LIST})
      list(APPEND OPENSHMEM_STATIC_CFLAGS "${X}")
    endforeach()
  endif()

  if(OPENSHMEM_STATIC_CFLAGS_OTHER)
    set(IS_PARAM "0")
    set(PARAM_FOUND "0")
    set(NEWPARAM "")
    set(IDX 0)
    set(FLAG_LIST "")

    foreach(X IN ITEMS ${OPENSHMEM_STATIC_CFLAGS_OTHER})
      string(FIND "${X}" "--param" PARAM_FOUND)
      if(NOT "${PARAM_FOUND}" EQUAL "-1")
        set(IS_PARAM "1")
        set(NEWPARAM "SHELL:${X}")
      endif()
      if("${PARAM_FOUND}" EQUAL "-1"
         AND "${IS_PARAM}" EQUAL "0"
         OR "${IS_PARAM}" EQUAL "-1"
      )
        list(APPEND FLAG_LIST "${X}")
        set(IS_PARAM "0")
      elseif("${PARAM_FOUND}" EQUAL "-1" AND "${IS_PARAM}" EQUAL "1")
        list(APPEND FLAG_LIST "${NEWPARAM}
          ${X}"
        )
        set(NEWPARAM "")
        set(IS_PARAM "0")
      endif()
    endforeach()

    list(LENGTH OPENSHMEM_STATIC_CFLAGS_OTHER IDX)
    foreach(X RANGE ${IDX})
      list(POP_FRONT OPENSHMEM_STATIC_CFLAGS_OTHER NEWPARAM)
    endforeach()

    foreach(X IN ITEMS ${FLAG_LIST})
      list(APPEND OPENSHMEM_STATIC_CFLAGS_OTHER "${X}")
    endforeach()
  endif()

  if(OPENSHMEM_STATIC_LDFLAGS)
    unset(FOUND_LIB)
    set(IS_PARAM "0")
    set(PARAM_FOUND "0")
    set(NEWPARAM "")
    set(SKIP 0)
    set(IDX 0)
    set(DIRIDX 0)
    set(FLAG_LIST "")
    set(DIR_LIST "")
    set(LIB_LIST "")

    foreach(X IN ITEMS ${OPENSHMEM_STATIC_LDFLAGS})
      string(FIND "${X}" "--param" PARAM_FOUND)
      string(FIND "${X}" "-lsma" IDX)
      string(FIND "${X}" "-L" DIRIDX)
      string(FIND "${X}" "-Wl" SKIP)

      if("${SKIP}" EQUAL "-1")
        if(NOT "${PARAM_FOUND}" EQUAL "-1")
          set(IS_PARAM "1")
          set(NEWPARAM "SHELL:${X}")
        endif()
        if("${PARAM_FOUND}" EQUAL "-1"
           AND "${IDX}" EQUAL "-1"
           AND "${IS_PARAM}" EQUAL "0"
           OR "${IS_PARAM}" EQUAL "-1"
        )
          list(APPEND FLAG_LIST "${X}")
          set(IS_PARAM "0")
        elseif("${PARAM_FOUND}" EQUAL "-1" AND "${IS_PARAM}" EQUAL "1")
          list(APPEND FLAG_LIST "${NEWPARAM}
          ${X}"
          )
          set(NEWPARAM "")
          set(IS_PARAM "0")
        elseif(NOT "${IDX}" EQUAL "-1" AND NOT "${LIDX}" EQUAL "-1")
          set(TMPSTR "")
          string(REPLACE "-l" "" TMPSTR "${X}")
          list(APPEND LIB_LIST "${TMPSTR}")
          set(IDX 0)
        elseif("${IDX}" EQUAL "-1" AND NOT "${LIDX}" EQUAL "-1")
          list(APPEND FLAG_LIST "${X}")
        endif()
        if(NOT "${DIRIDX}" EQUAL "-1")
          set(TMPSTR "")
          string(REPLACE "-L" "" TMPSTR "${X}")
          list(APPEND DIR_LIST "${TMPSTR}")
        endif()
      endif()
    endforeach()

    set(IDX 0)
    list(LENGTH OPENSHMEM_STATIC_LDFLAGS IDX)
    foreach(X RANGE ${IDX})
      list(POP_FRONT OPENSHMEM_STATIC_LDFLAGS NEWPARAM)
    endforeach()

    foreach(X IN ITEMS ${FLAG_LIST})
      list(APPEND OPENSHMEM_STATIC_LDFLAGS "${X}")
    endforeach()

    set(IDX 0)
    list(LENGTH LIB_LIST IDX)
    if(NOT "${IDX}" EQUAL "0")
      set(IDX 0)
      if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(NEWLINK "SHELL:-Wl,--whole-archive
          "
        )
        foreach(X IN ITEMS ${LIB_LIST})
          set(DIRSTR "")
          string(REPLACE ";" "
          " DIRSTR "${DIR_LIST}"
          )
          foreach(Y IN ITEMS ${DIR_LIST})
            find_library(
              FOUND_LIB
              NAMES ${X} "lib${X}" "lib${X}.a"
              PATHS ${Y}
              HINTS ${Y} NO_CACHE
              NO_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH
            )

            list(LENGTH FOUND_LIB IDX)

            if(NOT "${IDX}" EQUAL "0")
              string(APPEND NEWLINK "${FOUND_LIB}")
              set(FOUND_LIB "")
            endif()
          endforeach()
        endforeach()
        string(APPEND NEWLINK "
          -Wl,--no-whole-archive"
        )

        string(FIND "SHELL:-Wl,--whole-archive
          -Wl,--no-whole-archive" "${NEWLINK}" IDX
        )
        if("${IDX}" EQUAL "-1")
          list(APPEND OPENSHMEM_STATIC_LDFLAGS "${NEWLINK}")
        endif()
      elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(APPLE)
          set(NEWLINK "SHELL:-Wl,-force_load,")
        else()
          set(NEWLINK "SHELL:
          "
          )
        endif()
        foreach(X IN ITEMS ${LIB_LIST})
          set(DIRSTR "")
          string(REPLACE ";" "
          " DIRSTR "${DIR_LIST}"
          )
          foreach(Y IN ITEMS ${DIR_LIST})
            find_library(
              FOUND_LIB
              NAMES ${X} "lib${X}" "lib${X}.a"
              PATHS ${Y}
              HINTS ${Y} NO_CACHE
              NO_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH
            )

            list(LENGTH FOUND_LIB IDX)
            if(NOT "${IDX}" EQUAL "0")
              string(APPEND NEWLINK "${FOUND_LIB}")
              set(FOUND_LIB "")
            endif()
          endforeach()
        endforeach()
        string(FIND "SHELL:" "${NEWLINK}" IDX)
        if("${IDX}" EQUAL "-1")
          list(APPEND OPENSHMEM_LDFLAGS "${NEWLINK}")
        endif()
      endif()
    endif()
  endif()

  if(OPENSHMEM_STATIC_LDFLAGS_OTHER)
    unset(FOUND_LIB)
    set(IS_PARAM "0")
    set(PARAM_FOUND "0")
    set(NEWPARAM "")
    set(SKIP 0)
    set(IDX 0)
    set(DIRIDX 0)
    set(FLAG_LIST "")
    set(DIR_LIST "")
    set(LIB_LIST "")

    foreach(X IN ITEMS ${OPENSHMEM_STATIC_LDFLAGS_OTHER})
      string(FIND "${X}" "--param" PARAM_FOUND)
      string(FIND "${X}" "-lsma" IDX)
      string(FIND "${X}" "-L" DIRIDX)
      string(FIND "${X}" "-Wl" SKIP)

      if("${SKIP}" EQUAL "-1")
        if(NOT "${PARAM_FOUND}" EQUAL "-1")
          set(IS_PARAM "1")
          set(NEWPARAM "SHELL:${X}")
        endif()
        if("${PARAM_FOUND}" EQUAL "-1"
           AND "${IDX}" EQUAL "-1"
           AND "${IS_PARAM}" EQUAL "0"
           OR "${IS_PARAM}" EQUAL "-1"
        )
          list(APPEND FLAG_LIST "${X}")
          set(IS_PARAM "0")
        elseif("${PARAM_FOUND}" EQUAL "-1" AND "${IS_PARAM}" EQUAL "1")
          list(APPEND FLAG_LIST "${NEWPARAM}
          ${X}"
          )
          set(NEWPARAM "")
          set(IS_PARAM "0")
        elseif(NOT "${IDX}" EQUAL "-1" AND NOT "${LIDX}" EQUAL "-1")
          set(TMPSTR "")
          string(REPLACE "-l" "" TMPSTR "${X}")
          list(APPEND LIB_LIST "${TMPSTR}")
          set(IDX 0)
        elseif("${IDX}" EQUAL "-1" AND NOT "${LIDX}" EQUAL "-1")
          list(APPEND FLAG_LIST "${X}")
        endif()
        if(NOT "${DIRIDX}" EQUAL "-1")
          set(TMPSTR "")
          string(REPLACE "-L" "" TMPSTR "${X}")
          list(APPEND DIR_LIST "${TMPSTR}")
        endif()
      endif()
    endforeach()

    set(IDX 0)
    list(LENGTH OPENSHMEM_STATIC_LDFLAGS_OTHER IDX)
    foreach(X RANGE ${IDX})
      list(POP_FRONT OPENSHMEM_STATIC_LDFLAGS_OTHER NEWPARAM)
    endforeach()

    foreach(X IN ITEMS ${FLAG_LIST})
      list(APPEND OPENSHMEM_STATIC_LDFLAGS_OTHER "${X}")
    endforeach()

    set(IDX 0)
    list(LENGTH LIB_LIST IDX)
    if(NOT "${IDX}" EQUAL "0")
      set(IDX 0)
      if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(NEWLINK "SHELL:-Wl,--whole-archive
          "
        )
        foreach(X IN ITEMS ${LIB_LIST})
          set(DIRSTR "")
          string(REPLACE ";" "
          " DIRSTR "${DIR_LIST}"
          )
          foreach(Y IN ITEMS ${DIR_LIST})
            find_library(
              FOUND_LIB
              NAMES ${X} "lib${X}" "lib${X}.a"
              PATHS ${Y}
              HINTS ${Y} NO_CACHE
              NO_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH
            )

            list(LENGTH FOUND_LIB IDX)

            message(STATUS "${FOUND_LIB}
          ${X}"
            )
            if(NOT "${IDX}" EQUAL "0")
              string(APPEND NEWLINK "${FOUND_LIB}")
              set(FOUND_LIB "")
            endif()
          endforeach()
        endforeach()
        string(APPEND NEWLINK "
          -Wl,--no-whole-archive"
        )
        string(FIND "SHELL:-Wl,--whole-archive
          -Wl,--no-whole-archive" "${NEWLINK}" IDX
        )
        if("${IDX}" EQUAL "-1")
          list(APPEND OPENSHMEM_STATIC_LDFLAGS_OTHER "${NEWLINK}")
        endif()
      elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(APPLE)
          set(NEWLINK "SHELL:-Wl,-force_load,")
        else()
          set(NEWLINK "SHELL:
          "
          )
        endif()
        foreach(X IN ITEMS ${LIB_LIST})
          set(DIRSTR "")
          string(REPLACE ";" "
          " DIRSTR "${DIR_LIST}"
          )
          foreach(Y IN ITEMS ${DIR_LIST})
            find_library(
              FOUND_LIB
              NAMES ${X} "lib${X}" "lib${X}.a"
              PATHS ${Y}
              HINTS ${Y} NO_CACHE
              NO_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH
            )

            list(LENGTH FOUND_LIB IDX)
            if(NOT "${IDX}" EQUAL "0")
              string(APPEND NEWLINK "${FOUND_LIB}")
              set(FOUND_LIB "")
            endif()
          endforeach()
        endforeach()
        string(FIND "SHELL:" "${NEWLINK}" IDX)
        if("${IDX}" EQUAL "-1")
          list(APPEND OPENSHMEM_LDFLAGS "${NEWLINK}")
        endif()
      endif()
    endif()
  endif()

  if(OPENSHMEM_DIR)
    list(TRANSFORM OPENSHMEM_CFLAGS
         REPLACE "${OPENSHMEM_DIR}/install"
                 "$<BUILD_INTERFACE:${OPENSHMEM_DIR}/install>"
    )
    list(TRANSFORM OPENSHMEM_LDFLAGS
         REPLACE "${OPENSHMEM_DIR}/install"
                 "$<BUILD_INTERFACE:${OPENSHMEM_DIR}/install>"
    )
    list(TRANSFORM OPENSHMEM_LIBRARY_DIRS
         REPLACE "${OPENSHMEM_DIR}/install"
                 "$<BUILD_INTERFACE:${OPENSHMEM_DIR}/install>"
    )

    set_target_properties(
      PkgConfig::OPENSHMEM PROPERTIES INTERFACE_COMPILE_OPTIONS
                                      "${OPENSHMEM_CFLAGS}"
    )
    set_target_properties(
      PkgConfig::OPENSHMEM PROPERTIES INTERFACE_LINK_OPTIONS
                                      "${OPENSHMEM_LDFLAGS}"
    )
    set_target_properties(
      PkgConfig::OPENSHMEM PROPERTIES INTERFACE_LINK_DIRECTORIES
                                      "${OPENSHMEM_LIBRARY_DIRS}"
    )
  else()
    set_target_properties(
      PkgConfig::OPENSHMEM PROPERTIES INTERFACE_COMPILE_OPTIONS
                                      "${OPENSHMEM_CFLAGS}"
    )
    set_target_properties(
      PkgConfig::OPENSHMEM PROPERTIES INTERFACE_LINK_OPTIONS
                                      "${OPENSHMEM_LDFLAGS}"
    )
    set_target_properties(
      PkgConfig::OPENSHMEM PROPERTIES INTERFACE_LINK_DIRECTORIES
                                      "${OPENSHMEM_LIBRARY_DIRS}"
    )
  endif()

endmacro()
