# Copyright (c) 2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_PREPROCESSING_LOADED TRUE)

include(HPX_Include)

hpx_include(Message)
hpx_include(ListContains)

find_package(HPX_BoostWave)

if(BOOSTWAVE_FOUND)
    set(HPX_WAVE_ADDITIONAL_INCLUDE_DIRS ${HPX_WAVE_ADDITIONAL_INCLUDE_DIRS} CACHE PATH "Additional (compiler specific) include directories for the wave preprocessing tool.")
    mark_as_advanced(FORCE HPX_WAVE_ADDITIONAL_INCLUDE_DIRS)
endif()
hpx_option(HPX_AUTOMATIC_PREPROCESSING BOOL "True if the automatic header preprocessing target should be created (default: OFF)." OFF ADVANCED)

set(HPX_PREPROCESS_HEADERS CACHE INTERNAL "" FORCE)
set(HPX_PREPROCESS_INCLUDE_DIRS CACHE INTERNAL "" FORCE)

if(NOT (HPX_AUTOMATIC_PREPROCESSING AND ${BOOST_MINOR_VERSION} GREATER 50))
    macro(hpx_partial_preprocess_header file)
    endmacro()
    macro(hpx_setup_partial_preprocess_headers)
    endmacro()
else()
    if(NOT BOOSTWAVE_FOUND)
        hpx_warn("preprocessing" "Boost.Wave is unavailable, automatic preprocessing of header files is disabled. This feature is only needed if you change headers that need to be preprocessed. Set BOOST_ROOT or BOOSTWAVE_ROOT pointing to your Boost.Wave Installation")
    endif()

    macro(hpx_partial_preprocess_header file)
        hpx_list_contains(FILE_ADDED_ALREADY ${file} ${HPX_PREPROCESS_HEADERS})

        get_property(include_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
        set(include_dirs ${include_dirs} ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
        foreach(dir ${include_dirs})
            hpx_list_contains(INCLUDE_DIR_ADDED_ALREADY ${dir} ${HPX_PREPROCESS_INCLUDE_DIRS})
            if(NOT INCLUDE_DIR_ADDED_ALREADY)
                hpx_debug("preprocessing" "Adding ${dir} to list of additional include directories")
                set(HPX_PREPROCESS_INCLUDE_DIRS ${HPX_PREPROCESS_INCLUDE_DIRS} ${dir} CACHE INTERNAL "")
            endif()
        endforeach()

        if(NOT FILE_ADDED_ALREADY)
            hpx_debug("preprocessing" "Adding ${file} to list of targets to preprocess")
            set(HPX_PREPROCESS_HEADERS ${HPX_PREPROCESS_HEADERS} ${file} CACHE INTERNAL "")
            hpx_parse_arguments(HPX_PREPROCESS_HEADERS_${file} "GUARD;LIMIT" "" ${ARGN})
            if(HPX_PREPROCESS_HEADERS_${file}_LIMIT)
                set(HPX_PREPROCESS_HEADERS_${file}_LIMIT ${HPX_PREPROCESS_HEADERS_${file}_LIMIT} CACHE INTERNAL "" FORCE)
            else()
                set(HPX_PREPROCESS_HEADERS_${file}_LIMIT "HPX_LIMIT" CACHE INTERNAL "" FORCE)
            endif()

            if(HPX_PREPROCESS_HEADERS_${file}_GUARD)
                set(HPX_PREPROCESS_HEADERS_${file}_GUARD ${HPX_PREPROCESS_HEADERS_${file}_GUARD} CACHE INTERNAL "" FORCE)
            else()
                string(TOUPPER "${file}" THIS_GUARD)
                string(REGEX REPLACE "/" "_" THIS_GUARD "${THIS_GUARD}")
                string(REGEX REPLACE "\\." "_" THIS_GUARD "${THIS_GUARD}")
                string(REGEX REPLACE "HPX_" "HPX_PREPROCESSED_" THIS_GUARD "${THIS_GUARD}")
                set(HPX_PREPROCESS_HEADERS_${file}_GUARD ${THIS_GUARD} CACHE INTERNAL "" FORCE)
            endif()
        endif()
    endmacro()

    macro(hpx_setup_partial_preprocess_headers)
        set(HPX_PREPROCESSING_LIMITS)

        set(HPX_PREPROCESS_INCLUDE_HEADERS)
        foreach(file ${HPX_PREPROCESS_HEADERS})
            hpx_debug("preprocessing" "Generating preprocessing wrapper ${file}")
            set(HPX_PREPROCESS_INCLUDE_HEADERS "${HPX_PREPROCESS_INCLUDE_HEADERS}#include <${file}>\n")

            set(HPX_PREPROCESS_LIMIT ${HPX_PREPROCESS_HEADERS_${file}_LIMIT})
            set(HPX_PREPROCESS_GUARD ${HPX_PREPROCESS_HEADERS_${file}_GUARD})
            get_filename_component(PREPROCESS_DIR "${file}" PATH)
            get_filename_component(PREPROCESS_HEADER "${file}" NAME_WE)
            set(HPX_PREPROCESS_HEADER ${PREPROCESS_DIR}/preprocessed/${PREPROCESS_HEADER})
            configure_file(
                ${hpx_SOURCE_DIR}/cmake/templates/preprocess_include.hpp.in
                ${hpx_SOURCE_DIR}/${HPX_PREPROCESS_HEADER}.hpp
                ESCAPE_QUOTES
                @ONLY
            )
        endforeach()

        set(HPX_WAVE_ARGUMENTS)

        if(HPX_WAVE_ADDITIONAL_INCLUDE_DIRS)
            foreach(dir ${HPX_WAVE_ADDITIONAL_INCLUDE_DIRS})
                set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-S${dir}\n")
            endforeach()
        endif()

        foreach(dir ${HPX_PREPROCESS_INCLUDE_DIRS})
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-S${dir}\n")
        endforeach()

        foreach(def ${HPX_PREPROCESS_DEFINITIONS})
            set(HPX_WAVE_DEFINITIONS "${HPX_WAVE_DEFINITIONS}${def}\n")
        endforeach()

        set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}#\n")
        set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}# Specify compiler specific macro names (adapt for your compiler)\n")
        set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}# \n")
        if(MSVC)
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D_WIN32\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-DWIN32\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D_MT\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D_DLL\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D_MSC_VER=1600\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D__forceinline=__inline\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D__uuidof(x)=IID()\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D__w64=\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D__int8=char\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D__int16=short\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D__int32=int\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D__int64=long long\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D__ptr64=\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D_MSC_EXTENSIONS\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D_M_IX86\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D_WCHAR_T_DEFINED\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D_INTEGRAL_MAX_BITS=64\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-DPASCAL=__stdcall\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-DRPC_ENTRY=__stdcall\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-DSHSTDAPI=HRESULT\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-DSHSTDAPI_(x)=x\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D_WIN32_WINNT=0x0501\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D_CPPUNWIND\n")
        else()
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D__GNUC__=${GCC_VERSION}\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D__USE_ISOC99\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D_GCC_LIMITS_H_\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D__USE_POSIX\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D__x86_64__\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D__EXCEPTIONS\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-DCHAR_BIT=8\n")
            set(HPX_WAVE_ARGUMENTS "${HPX_WAVE_ARGUMENTS}-D__GXX_EXPERIMENTAL_CXX0X__\n")
        endif()

        set(output_dir ${CMAKE_BINARY_DIR})
        hpx_info("preprocessing" "output_dir: ${output_dir}")

        configure_file(
            ${hpx_SOURCE_DIR}/cmake/templates/wave.cfg.in
            ${output_dir}/preprocess/wave.cfg
            ESCAPE_QUOTES
            @ONLY
        )

        configure_file(
            ${hpx_SOURCE_DIR}/cmake/templates/preprocess_hpx.cpp.in
            ${output_dir}/preprocess/preprocess_hpx.cpp
            ESCAPE_QUOTES
            @ONLY
        )

        foreach(limit RANGE 5 20 5)
            set(HPX_PREPROCESSING_LIMITS ${HPX_PREPROCESSING_LIMITS} "hpx_partial_preprocess_headers_${limit}")

            add_custom_command(
                OUTPUT ${output_dir}/preprocess/hpx_preprocessed_${limit}.touch
                COMMAND ${BOOSTWAVE_PROGRAM}
                ARGS -o- -DHPX_LIMIT=${limit} ${output_dir}/preprocess/preprocess_hpx.cpp --license=${hpx_SOURCE_DIR}/preprocess/preprocess_license.hpp --config-file ${output_dir}/preprocess/wave.cfg
            )
            add_custom_target(
                hpx_partial_preprocess_headers_${limit}
                DEPENDS ${output_dir}/preprocess/hpx_preprocessed_${limit}.touch
            )
            set_target_properties(hpx_partial_preprocess_headers_${limit}
                PROPERTIES FOLDER "Preprocessing/Dependencies")
        endforeach()

        add_custom_target(
            hpx_partial_preprocess_headers
            DEPENDS ${HPX_PREPROCESSING_LIMITS}
            WORKING_DIRECTORY ${output_dir}/preprocess
        )
        set_target_properties(hpx_partial_preprocess_headers
            PROPERTIES FOLDER "Preprocessing")

    endmacro()
endif()
