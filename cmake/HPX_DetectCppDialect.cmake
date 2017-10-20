# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c) 2011-2017 Thomas Heller
# Copyright (c) 2017 Anton Bikineev
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

macro(hpx_detect_cpp_dialect_non_msvc)

  if(HPX_WITH_CUDA AND NOT HPX_WITH_CUDA_CLANG)
    set(CXX_FLAG -std=c++11)
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
      hpx_info("C++ mode used: C++17")
    else()
      # ... otherwise try -std=c++1z
      if(HPX_WITH_CXX1Z OR NOT (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
                                AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17)))
        check_cxx_compiler_flag(-std=c++1z HPX_WITH_CXX1Z)
      endif()

      if(HPX_WITH_CXX1Z)
        set(CXX_FLAG -std=c++1z)
        hpx_add_config_cond_define(BOOST_NO_AUTO_PTR)
        hpx_info("C++ mode used: C++1z")
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
          hpx_info("C++ mode used: C++14")
        else()
          # ... otherwise try -std=c++1y
          if(HPX_WITH_CXX1Y OR NOT (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
                                    AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17)))
            check_cxx_compiler_flag(-std=c++1y HPX_WITH_CXX1Y)
          endif()

          if(HPX_WITH_CXX1Y)
            set(CXX_FLAG -std=c++1y)
            hpx_info("C++ mode used: C++1y")
          else()
            # ... otherwise try -std=c++11
            check_cxx_compiler_flag(-std=c++11 HPX_WITH_CXX11)
            if(HPX_WITH_CXX11)
              set(CXX_FLAG -std=c++11)
              hpx_info("C++ mode used: C++11")
            else()
              # ... otherwise try -std=c++0x
              check_cxx_compiler_flag(-std=c++0x HPX_WITH_CXX0X)
              if(HPX_WITH_CXX0X)
                set(CXX_FLAG -std=c++0x)
                hpx_info("C++ mode used: C++0x")
              endif()
            endif()
          endif()
        endif()
      endif()
    endif()
  endif()
endmacro()

macro(hpx_detect_cpp_dialect)

  if(MSVC)
    set(CXX_FLAG)

    # enable enforcing a particular C++ mode
    if(HPX_WITH_CXX17)
      set(CXX_FLAG -std:c++latest)
      hpx_add_config_cond_define(_HAS_AUTO_PTR_ETC 1)
      hpx_info("C++ mode enforced: C++17")
    elseif(HPX_WITH_CXX1Z)
      set(CXX_FLAG -std:c++latest)
      hpx_add_config_cond_define(_HAS_AUTO_PTR_ETC 1)
      hpx_add_config_cond_define(BOOST_NO_AUTO_PTR)
      hpx_info("C++ mode enforced: C++1z")
    elseif(HPX_WITH_CXX14)
      set(CXX_FLAG -std:c++14)
      hpx_add_config_cond_define(_HAS_AUTO_PTR_ETC 1)
      if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        hpx_add_config_cond_define(BOOST_NO_CXX14_CONSTEXPR)
      endif()
      hpx_info("C++ mode enforced: C++14")
    elseif(HPX_WITH_CXX1Y)
      set(CXX_FLAG -std:c++14)
      hpx_add_config_cond_define(_HAS_AUTO_PTR_ETC 1)
      hpx_info("C++ mode enforced: C++1y")
    elseif(HPX_WITH_CXX11)
      hpx_info("C++ mode enforced: C++11")
    elseif(HPX_WITH_CXX0X)
      hpx_info("C++ mode enforced: C++0x")
    else()
      hpx_info("C++ mode assumed: C++11")
    endif()

  else(MSVC)

    # enable enforcing a particular C++ mode
    if(HPX_WITH_CXX17)
      set(CXX_FLAG -std=c++17)
      hpx_add_config_cond_define(BOOST_NO_AUTO_PTR)
      hpx_info("C++ mode enforced: C++17")
    elseif(HPX_WITH_CXX1Z)
      set(CXX_FLAG -std=c++1z)
      hpx_add_config_cond_define(BOOST_NO_AUTO_PTR)
      hpx_info("C++ mode enforced: C++1z")
    elseif(HPX_WITH_CXX14)
      set(CXX_FLAG -std=c++14)
      if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        hpx_add_config_cond_define(BOOST_NO_CXX14_CONSTEXPR)
      endif()
      hpx_info("C++ mode enforced: C++14")
    elseif(HPX_WITH_CXX1Y)
      set(CXX_FLAG -std=c++1y)
      hpx_info("C++ mode enforced: C++1y")
    elseif(HPX_WITH_CXX11)
      set(CXX_FLAG -std=c++11)
      hpx_info("C++ mode enforced: C++11")
    elseif(HPX_WITH_CXX0X)
      set(CXX_FLAG -std=c++0x)
      hpx_info("C++ mode enforced: C++0x")
    else()
      # if no C++ mode is enforced, try to detect which one to use
      hpx_detect_cpp_dialect_non_msvc()
    endif()

  endif(MSVC)

  if(CXX_FLAG)
    hpx_add_target_compile_option(${CXX_FLAG})
  endif()

endmacro()
