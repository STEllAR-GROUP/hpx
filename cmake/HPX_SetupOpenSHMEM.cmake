# Copyright (c)      2017 Thomas Heller
# Copyright (c)      2023 Christopher Taylor
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#

macro(hpx_setup_openshmem)

  if(NOT TARGET PkgConfig::OPENSHMEM)

    find_package(PkgConfig REQUIRED QUIET COMPONENTS)

    hpx_info(
      "OpenSHMEM needs to be compiled with the following environment variables set during autoconf: `CFLAGS=-fPIC CXXFLAGS=-fPIC ./configure ...`"
    )

    if("${HPX_WITH_PARCELPORT_OPENSHMEM_CONDUIT}" STREQUAL "ucx")
      pkg_search_module(
        OPENSHMEM REQUIRED IMPORTED_TARGET GLOBAL
        osss-ucx
      )
    elseif("${HPX_WITH_PARCELPORT_OPENSHMEM_CONDUIT}" STREQUAL "sos")
      pkg_search_module(
        OPENSHMEM REQUIRED IMPORTED_TARGET GLOBAL
        sandia-openshmem
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
          list(APPEND FLAG_LIST "${NEWPARAM} ${X}")
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
          list(APPEND FLAG_LIST "${NEWPARAM} ${X}")
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
          list(APPEND FLAG_LIST "${NEWPARAM} ${X}")
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
          set(NEWLINK "SHELL:-Wl,--whole-archive ")
          foreach(X IN ITEMS ${LIB_LIST})
            set(DIRSTR "")
            string(REPLACE ";" " " DIRSTR "${DIR_LIST}")
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
          string(APPEND NEWLINK " -Wl,--no-whole-archive")
          string(FIND "SHELL:-Wl,--whole-archive  -Wl,--no-whole-archive"
                      "${NEWLINK}" IDX
          )
          if("${IDX}" EQUAL "-1")
            list(APPEND OPENSHMEM_LDFLAGS "${NEWLINK}")
          endif()
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
          if(APPLE)
            set(NEWLINK "SHELL:-Wl,-force_load,")
	  else()
            set(NEWLINK "SHELL: ")
          endif()
          foreach(X IN ITEMS ${LIB_LIST})
            set(DIRSTR "")
            string(REPLACE ";" " " DIRSTR "${DIR_LIST}")
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
          string(FIND "SHELL:"
                      "${NEWLINK}" IDX
          )
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
          list(APPEND FLAG_LIST "${NEWPARAM} ${X}")
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
          set(NEWLINK "SHELL:-Wl,--whole-archive ")
          foreach(X IN ITEMS ${LIB_LIST})
            set(DIRSTR "")
            string(REPLACE ";" " " DIRSTR "${DIR_LIST}")
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
          string(APPEND NEWLINK " -Wl,--no-whole-archive")

          string(FIND "SHELL:-Wl,--whole-archive  -Wl,--no-whole-archive"
                      "${NEWLINK}" IDX
          )
          if("${IDX}" EQUAL "-1")
            list(APPEND OPENSHMEM_LDFLAGS_OTHER "${NEWLINK}")
          endif()
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
          if(APPLE)
            set(NEWLINK "SHELL:-Wl,-force_load,")
	  else()
            set(NEWLINK "SHELL: ")
          endif()
          foreach(X IN ITEMS ${LIB_LIST})
            set(DIRSTR "")
            string(REPLACE ";" " " DIRSTR "${DIR_LIST}")
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
          string(FIND "SHELL:"
                      "${NEWLINK}" IDX
          )
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
          list(APPEND FLAG_LIST "${NEWPARAM} ${X}")
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
          list(APPEND FLAG_LIST "${NEWPARAM} ${X}")
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
          list(APPEND FLAG_LIST "${NEWPARAM} ${X}")
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
          set(NEWLINK "SHELL:-Wl,--whole-archive ")
          foreach(X IN ITEMS ${LIB_LIST})
            set(DIRSTR "")
            string(REPLACE ";" " " DIRSTR "${DIR_LIST}")
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
          string(APPEND NEWLINK " -Wl,--no-whole-archive")

          string(FIND "SHELL:-Wl,--whole-archive  -Wl,--no-whole-archive"
                      "${NEWLINK}" IDX
          )
          if("${IDX}" EQUAL "-1")
            list(APPEND OPENSHMEM_STATIC_LDFLAGS "${NEWLINK}")
          endif()
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
          if(APPLE)
            set(NEWLINK "SHELL:-Wl,-force_load,")
	  else()
            set(NEWLINK "SHELL: ")
          endif()
          foreach(X IN ITEMS ${LIB_LIST})
            set(DIRSTR "")
            string(REPLACE ";" " " DIRSTR "${DIR_LIST}")
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
          string(FIND "SHELL:"
                      "${NEWLINK}" IDX
          )
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
          list(APPEND FLAG_LIST "${NEWPARAM} ${X}")
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
          set(NEWLINK "SHELL:-Wl,--whole-archive ")
          foreach(X IN ITEMS ${LIB_LIST})
            set(DIRSTR "")
            string(REPLACE ";" " " DIRSTR "${DIR_LIST}")
            foreach(Y IN ITEMS ${DIR_LIST})
              find_library(
                FOUND_LIB
                NAMES ${X} "lib${X}" "lib${X}.a"
                PATHS ${Y}
                HINTS ${Y} NO_CACHE
                NO_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH
              )

              list(LENGTH FOUND_LIB IDX)

              message(STATUS "${FOUND_LIB} ${X}")
              if(NOT "${IDX}" EQUAL "0")
                string(APPEND NEWLINK "${FOUND_LIB}")
                set(FOUND_LIB "")
              endif()
            endforeach()
          endforeach()
          string(APPEND NEWLINK " -Wl,--no-whole-archive")
          string(FIND "SHELL:-Wl,--whole-archive  -Wl,--no-whole-archive"
                      "${NEWLINK}" IDX
          )
          if("${IDX}" EQUAL "-1")
            list(APPEND OPENSHMEM_STATIC_LDFLAGS_OTHER "${NEWLINK}")
          endif()
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
          if(APPLE)
            set(NEWLINK "SHELL:-Wl,-force_load,")
	  else()
            set(NEWLINK "SHELL: ")
          endif()
          foreach(X IN ITEMS ${LIB_LIST})
            set(DIRSTR "")
            string(REPLACE ";" " " DIRSTR "${DIR_LIST}")
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
          string(FIND "SHELL:"
                      "${NEWLINK}" IDX
          )
          if("${IDX}" EQUAL "-1")
            list(APPEND OPENSHMEM_LDFLAGS "${NEWLINK}")
          endif()
        endif()
      endif()
    endif()

    set_target_properties(
      PkgConfig::OPENSHMEM PROPERTIES INTERFACE_COMPILE_OPTIONS "${OPENSHMEM_CFLAGS}"
    )
    set_target_properties(
      PkgConfig::OPENSHMEM PROPERTIES INTERFACE_LINK_OPTIONS "${OPENSHMEM_LDFLAGS}"
    )
    set_target_properties(
      PkgConfig::OPENSHMEM PROPERTIES INTERFACE_LINK_DIRECTORIES
                                   "${OPENSHMEM_LIBRARY_DIRS}"
    )

  endif()

endmacro()
