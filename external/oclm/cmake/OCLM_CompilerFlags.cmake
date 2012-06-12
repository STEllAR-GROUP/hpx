# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(OCLM_COMPILERFLAGS_LOADED TRUE)

include(OCLM_Include)

oclm_include(Message
            GCCVersion
            AddConfigTest
            ParseArguments)

macro(oclm_language_suffix var language)
  if(${language} STREQUAL "CXX")
    set(${var} .cpp)
  elseif(${language} STREQUAL "Fortran")
    set(${var} .fpp)
  else()
    oclm_error("language_suffix" "${language} is unsupported")
  endif()
endmacro()

macro(oclm_append_flag flag)
  oclm_parse_arguments(APPEND "LANGUAGES" "" ${ARGN})

  set(languages "CXX")

  if(APPEND_LANGUAGES)
    set(languages ${APPEND_LANGUAGES})
  endif()

  foreach(language ${languages})
    set(CMAKE_${language}_FLAGS "${CMAKE_${language}_FLAGS} ${flag}")
  endforeach()
endmacro()

macro(oclm_remove_flag flag)
  oclm_parse_arguments(REMOVE "LANGUAGES" "" ${ARGN})

  set(languages "CXX")

  if(REMOVE_LANGUAGES)
    set(languages ${REMOVE_LANGUAGES})
  endif()

  foreach(language ${languages})
    string(REPLACE "${flag}" ""
           CMAKE_${language}_FLAGS "${CMAKE_${language}_FLAGS}")
  endforeach()
endmacro()

macro(oclm_use_flag_if_available flag)
    string(TOUPPER ${flag} uppercase_name)
    
    string(REGEX REPLACE "^-+" "" uppercase_name ${uppercase_name})
    string(REGEX REPLACE "[=\\-]" "_" uppercase_name ${uppercase_name})
    string(REGEX REPLACE "\\+" "X" uppercase_name ${uppercase_name})
    
    string(TOLOWER ${uppercase_name} lowercase_name)
    
    # C++ is the only language tested by default
    set(language "CXX")
    
    if(CMAKE_${language}_COMPILER)
        set(language_suffix ".cpp")
        string(TOUPPER ${language} uppercase_language)
        
        if(OCLM_ROOT)
            set(source_dir "${OCLM_ROOT}")
            add_oclm_config_test(${lowercase_name}
                OCLM_${uppercase_language}_FLAG_${uppercase_name}
                DEFINITIONS OCLM_HAVE_${uppercase_language}_FLAG_${uppercase_name}
                LANGUAGE ${language}
                ROOT ${source_dir}
                SOURCE cmake/tests/flag${language_suffix}
                FLAGS "${flag}" FILE)
        elseif($ENV{OCLM_ROOT})
            set(source_dir "$ENV{OCLM_ROOT}")
            add_oclm_config_test(${lowercase_name}
                OCLM_${uppercase_language}_FLAG_${uppercase_name}
                DEFINITIONS OCLM_HAVE_${uppercase_language}_FLAG_${uppercase_name}
                LANGUAGE ${language}
                ROOT ${source_dir}
                SOURCE cmake/tests/flag${language_suffix}
                FLAGS "${flag}" FILE)
        else()
            add_oclm_config_test(${lowercase_name}
                OCLM_${uppercase_language}_FLAG_${uppercase_name}
                DEFINITIONS OCLM_HAVE_${uppercase_language}_FLAG_${uppercase_name}
                LANGUAGE ${language}
                SOURCE cmake/tests/flag${language_suffix}
                FLAGS "${flag}" FILE)
        endif()
        
        if(OCLM_${uppercase_language}_FLAG_${uppercase_name})
            oclm_append_flag("${flag}" LANGUAGES ${language})
        else()
            oclm_warn("use_flag_if_available" "${flag} is unavailable for ${language}")
        endif()
    endif()
endmacro()

