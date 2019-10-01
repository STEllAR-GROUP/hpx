# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c) 2011-2017 Thomas Heller
# Copyright (c) 2017 Anton Bikineev
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(hpx_detect_cpp_dialect_non_msvc)

  if(HPX_WITH_CUDA AND NOT HPX_WITH_CUDA_CLANG)
    set(CXX_FLAG -std=c++11)
    set(HPX_CXX_STANDARD 11)
    hpx_info("C++ mode used: C++11")
  else()

    # Try -std=c++17 first
    if(HPX_WITH_CXX17 OR NOT (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
                              AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17)))
      check_cxx_compiler_flag(-std=c++17 HPX_WITH_CXX17)
    endif()

    if(HPX_WITH_CXX17)
      set(CXX_FLAG -std=c++17)
      hpx_add_config_cond_define(BOOST_NO_AUTO_PTR)
      set(HPX_CXX_STANDARD 17)
    else()
      # ... otherwise try -std=c++1z
      if(HPX_WITH_CXX1Z OR NOT (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
                                AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17)))
        check_cxx_compiler_flag(-std=c++1z HPX_WITH_CXX1Z)
      endif()

      if(HPX_WITH_CXX1Z)
        set(CXX_FLAG -std=c++1z)
        hpx_add_config_cond_define(BOOST_NO_AUTO_PTR)
        set(HPX_CXX_STANDARD 1z)
      else()
        # ... otherwise try -std=c++14
        if(HPX_WITH_CXX14 OR NOT (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
                                  AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17)))
          check_cxx_compiler_flag(-std=c++14 HPX_WITH_CXX14)
        endif()

        if(HPX_WITH_CXX14)
          set(CXX_FLAG -std=c++14)
          # The Intel compiler doesn't appear to have a fully functional
          # implementation of C++14 constexpr. It's fine with our C++14 constexpr
          # usage in HPX but chokes on Boost.
          # FIXME: This should be replaced with a version-based check in the future
          # when the Intel compiler is able to build Boost with -std=c++14.
          if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            hpx_add_config_cond_define(BOOST_NO_CXX14_CONSTEXPR)
          endif()
          set(HPX_CXX_STANDARD 14)
        else()
          # ... otherwise try -std=c++1y
          if(HPX_WITH_CXX1Y OR NOT (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
                                    AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17)))
            check_cxx_compiler_flag(-std=c++1y HPX_WITH_CXX1Y)
          endif()

          if(HPX_WITH_CXX1Y)
            set(CXX_FLAG -std=c++1y)
            set(HPX_CXX_STANDARD 1y)
          else()
            # ... otherwise try -std=c++11
            check_cxx_compiler_flag(-std=c++11 HPX_WITH_CXX11)
            if(HPX_WITH_CXX11)
              set(CXX_FLAG -std=c++11)
              set(HPX_CXX_STANDARD 11)
              hpx_warn("Compiling in C++11 mode is deprecated. HPX will require C++14 support in future releases. Set HPX_WITH_CXX14 (or newer) to ON during CMake configuration to enable C++14 support.")
            else()
              # ... otherwise try -std=c++0x
              check_cxx_compiler_flag(-std=c++0x HPX_WITH_CXX0X)
              if(HPX_WITH_CXX0X)
                hpx_error("HPX requires at least C++11 while C+0x was enforced")
              endif()
            endif()
          endif()
        endif()
      endif()
    endif()
  endif()
  set(HPX_CXX_STANDARD ${HPX_CXX_STANDARD} PARENT_SCOPE)
  set(CXX_FLAG ${CXX_FLAG} PARENT_SCOPE)
endfunction()

function(hpx_detect_cpp_dialect)

  # the default should be C++14
  set(HPX_CXX_STANDARD 14)

  if(MSVC)
    set(CXX_FLAG)

    # enable enforcing a particular C++ mode
    if(HPX_WITH_CXX2A)
      set(CXX_FLAG -std:c++latest)
      set(HPX_CXX_STANDARD 2a)
      hpx_add_config_cond_define(BOOST_NO_AUTO_PTR)
      hpx_add_config_cond_define(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
    elseif(HPX_WITH_CXX17)
      set(CXX_FLAG -std:c++17)
      set(HPX_CXX_STANDARD 17)
      hpx_add_config_cond_define(BOOST_NO_AUTO_PTR)
      hpx_add_config_cond_define(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
    elseif(HPX_WITH_CXX1Z)
      set(HPX_CXX_STANDARD 1z)
      set(CXX_FLAG -std:c++latest)
      hpx_add_config_cond_define(BOOST_NO_AUTO_PTR)
      hpx_add_config_cond_define(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
    elseif(HPX_WITH_CXX14)
      set(CXX_FLAG -std:c++14)
      if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        hpx_add_config_cond_define(BOOST_NO_CXX14_CONSTEXPR)
      endif()
      set(HPX_CXX_STANDARD 14)
    elseif(HPX_WITH_CXX1Y)
      set(CXX_FLAG -std:c++14)
      set(HPX_CXX_STANDARD 1y)
    elseif(HPX_WITH_CXX11)
      set(HPX_CXX_STANDARD 11)
      hpx_warn("Compiling in C++11 mode is dprecated. HPX will require C++14 support in future releases. Set HPX_WITH_CXX14 (or newer) to ON during CMake configuration to enable C++14 support.")
    elseif(HPX_WITH_CXX0X)
      hpx_error("HPX requires at least C++11 while C+0x was enforced")
    endif()

    hpx_add_config_cond_define(_HAS_AUTO_PTR_ETC 1)

  else(MSVC)

    # enable enforcing a particular C++ mode
    if(HPX_WITH_CXX2A)
      set(CXX_FLAG -std=c++2a)
      set(HPX_CXX_STANDARD 2a)
      hpx_add_config_cond_define(BOOST_NO_AUTO_PTR)
    elseif(HPX_WITH_CXX17)
      set(CXX_FLAG -std=c++17)
      set(HPX_CXX_STANDARD 17)
      hpx_add_config_cond_define(BOOST_NO_AUTO_PTR)
    elseif(HPX_WITH_CXX1Z)
      set(CXX_FLAG -std=c++1z)
      set(HPX_CXX_STANDARD 1z)
      hpx_add_config_cond_define(BOOST_NO_AUTO_PTR)
    elseif(HPX_WITH_CXX14)
      set(CXX_FLAG -std=c++14)
      set(HPX_CXX_STANDARD 14)
      if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        hpx_add_config_cond_define(BOOST_NO_CXX14_CONSTEXPR)
      endif()
    elseif(HPX_WITH_CXX1Y)
      set(CXX_FLAG -std=c++1y)
      set(HPX_CXX_STANDARD 1y)
    elseif(HPX_WITH_CXX11)
      set(CXX_FLAG -std=c++11)
      set(HPX_CXX_STANDARD 11)
      hpx_warn("Compiling in C++11 mode is deprecated. HPX will require C++14 support in future releases. Set HPX_WITH_CXX14 (or newer) to ON during CMake configuration to enable C++14 support.")
    elseif(HPX_WITH_CXX0X)
      hpx_error("HPX requires at least C++11 while C+0x was enforced")
    else()
      # if no C++ mode is enforced, try to detect which one to use
      hpx_detect_cpp_dialect_non_msvc()
    endif()

  endif(MSVC)

  set(HPX_CXX_STANDARD ${HPX_CXX_STANDARD} PARENT_SCOPE)
  hpx_info("C++ mode used: C++${HPX_CXX_STANDARD}")

  if(CXX_FLAG)
    hpx_add_target_compile_option(${CXX_FLAG} PUBLIC)
  endif()

  # Re-export the local CXX_FLAG varaible.
  set(CXX_FLAG ${CXX_FLAG} PARENT_SCOPE)

endfunction()
