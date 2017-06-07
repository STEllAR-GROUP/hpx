# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c) 2011-2017 Thomas Heller
# Copyright (c) 2017 Anton Bikineev
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

macro(hpx_detect_cpp_dialect)
  if(NOT MSVC)
    if(HPX_WITH_CUDA AND NOT HPX_WITH_CUDA_CLANG)
      set(CXX_FLAG -std=c++11)
    else()

      # Try -std=c++17 first
      check_cxx_compiler_flag(-std=c++17 HPX_WITH_CXX17)

      if(HPX_WITH_CXX17)
        set(CXX_FLAG -std=c++17)
        add_definitions(-DBOOST_NO_AUTO_PTR)
      else()
        # ... otherwise try -std=c++1z
        check_cxx_compiler_flag(-std=c++1z HPX_WITH_CXX1Z)

        if(HPX_WITH_CXX1Z)
          set(CXX_FLAG -std=c++1z)
          add_definitions(-DBOOST_NO_AUTO_PTR)
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
              add_definitions(-DBOOST_NO_CXX14_CONSTEXPR)
            endif()
          else()
            # ... otherwise try -std=c++1y
            if(HPX_WITH_CXX1Y OR NOT (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
                                      AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17)))
              check_cxx_compiler_flag(-std=c++1y HPX_WITH_CXX1Y)
            endif()

            if(HPX_WITH_CXX1Y)
              set(CXX_FLAG -std=c++1y)
            else()
              # ... otherwise try -std=c++11
              check_cxx_compiler_flag(-std=c++11 HPX_WITH_CXX11)
              if(HPX_WITH_CXX11)
                set(CXX_FLAG -std=c++11)
              else()
                # ... otherwise try -std=c++0x
                check_cxx_compiler_flag(-std=c++0x HPX_WITH_CXX0X)
                if(HPX_WITH_CXX0X)
                  set(CXX_FLAG -std=c++0x)
                endif()
              endif()
            endif()
          endif()
        endif()
      endif()
    endif()
    hpx_add_target_compile_option(${CXX_FLAG})
  endif()
endmacro()
