# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_COMPILERFLAGS_LOADED TRUE)

include(HPX_Include)

hpx_include(Message
            GCCVersion
            AddConfigTest
            ParseArguments)

macro(hpx_language_suffix var language)
  if(${language} STREQUAL "CXX")
    set(${var} .cpp)
  elseif(${language} STREQUAL "C")
    set(${var} .c)
  elseif(${language} STREQUAL "Fortran")
    set(${var} .fpp)
  else()
    hpx_error("language_suffix" "${language} is unsupported")
  endif()
endmacro()

macro(hpx_append_flag flag)
  hpx_parse_arguments(APPEND "LANGUAGES" "" ${ARGN})

  set(languages "CXX")

  if(APPEND_LANGUAGES)
    set(languages ${APPEND_LANGUAGES})
  endif()

  foreach(language ${languages})
    set(CMAKE_${language}_FLAGS "${CMAKE_${language}_FLAGS} ${flag}")
  endforeach()
endmacro()

macro(hpx_remove_flag flag)
  hpx_parse_arguments(REMOVE "LANGUAGES" "" ${ARGN})

  set(languages "CXX")

  if(REMOVE_LANGUAGES)
    set(languages ${REMOVE_LANGUAGES})
  endif()

  foreach(language ${languages})
    string(REPLACE "${flag}" ""
           CMAKE_${language}_FLAGS "${CMAKE_${language}_FLAGS}")
  endforeach()
endmacro()

macro(hpx_use_flag_if_available flag)
  hpx_parse_arguments(FLAG "NAME;LANGUAGES" "" ${ARGN})

  set(uppercase_name "")

  if(FLAG_NAME)
    string(TOUPPER ${FLAG_NAME} uppercase_name)
  else()
    string(TOUPPER ${flag} uppercase_name)
  endif()

  string(REGEX REPLACE "^-+" "" uppercase_name ${uppercase_name})
  string(REGEX REPLACE "[=\\-]" "_" uppercase_name ${uppercase_name})
  string(REGEX REPLACE "\\+" "X" uppercase_name ${uppercase_name})

  string(TOLOWER ${uppercase_name} lowercase_name)

  # C++ is the only language tested by default
  set(languages "CXX")

  if(FLAG_LANGUAGES)
    set(languages ${FLAG_LANGUAGES})
  endif()

  foreach(language ${languages})
    if(CMAKE_${language}_COMPILER)
      hpx_language_suffix(language_suffix ${language})
      string(TOUPPER ${language} uppercase_language)

      if(HPX_ROOT)
        set(source_dir "${HPX_ROOT}")
        add_hpx_config_test(${lowercase_name}
          HPX_${language}_FLAG_${uppercase_name}
          DEFINITIONS HPX_HAVE_${uppercase_language}_FLAG_${uppercase_name}
          LANGUAGE ${language}
          ROOT ${source_dir}
          SOURCE cmake/tests/flag${language_suffix}
          FLAGS "${flag}" FILE)
      elseif($ENV{HPX_ROOT})
        set(source_dir "$ENV{HPX_ROOT}")
        add_hpx_config_test(${lowercase_name}
          HPX_${language}_FLAG_${uppercase_name}
          DEFINITIONS HPX_HAVE_${uppercase_language}_FLAG_${uppercase_name}
          LANGUAGE ${language}
          ROOT ${source_dir}
          SOURCE cmake/tests/flag${language_suffix}
          FLAGS "${flag}" FILE)
      else()
        add_hpx_config_test(${lowercase_name}
          HPX_${language}_FLAG_${uppercase_name}
          DEFINITIONS HPX_HAVE_${uppercase_language}_FLAG_${uppercase_name}
          LANGUAGE ${language}
          SOURCE cmake/tests/flag${language_suffix}
          FLAGS "${flag}" FILE)
      endif()

      if(HPX_${language}_FLAG_${uppercase_name})
        hpx_append_flag("${flag}" LANGUAGES ${language})
      #else()
      #  hpx_warn("use_flag_if_available" "${flag} is unavailable for ${language}")
      endif()
    endif()
  endforeach()
endmacro()

