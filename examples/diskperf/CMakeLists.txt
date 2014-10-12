# Copyright (c) 2013 Shuangyang Yang
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(OrangeFS)

if(HPX_WITH_CXX11_LAMBDAS)
    set(example_programs
            diskperf_hpx_async_local_file
            diskperf_hpx_sync_local_file)
    set(diskperf_hpx_async_local_file_FLAGS DEPENDENCIES
            iostreams_component local_file_component)
    set(diskperf_hpx_sync_local_file_FLAGS DEPENDENCIES
            iostreams_component local_file_component)

    if(ORANGEFS_FOUND)
        # add OrangeFS header files directory
        include_directories(${ORANGEFS_INCLUDE_DIR})
        set(example_programs
                ${example_programs}
                diskperf_thread_sync_local_orangefs
                diskperf_hpx_sync_orangefs_file
                diskperf_hpx_async_orangefs_pxfs_file)
        set(diskperf_thread_sync_local_orangefs_FLAGS DEPENDENCIES
                ${ORANGEFS_LIBRARY} pthread dl rt crypto)
        set(diskperf_hpx_sync_orangefs_file_FLAGS DEPENDENCIES
                iostreams_component orangefs_file_component)
        set(diskperf_hpx_async_orangefs_pxfs_file_FLAGS DEPENDENCIES
                iostreams_component orangefs_file_component
                ${ORANGEFS_LIBRARY} pthread dl rt crypto)
    endif()
endif()

foreach(example_program ${example_programs})

    set(sources ${example_program}.cpp)

    source_group("Source Files" FILES ${sources})

    # add example executable
    add_hpx_executable(${example_program}
            SOURCES ${sources}
            ${${example_program}_FLAGS}
            FOLDER "Examples/Diskperf/${example_program}")

    # add a custom target for this example
    add_hpx_pseudo_target(examples.diskperf.${example_program})

    # make pseudo-targets depend on master pseudo-target
    add_hpx_pseudo_dependencies(examples.diskperf
            examples.diskperf.${example_program})

    # add dependencies to pseudo-target
    add_hpx_pseudo_dependencies(examples.diskperf.${example_program}
            ${example_program}_exe)
endforeach()

