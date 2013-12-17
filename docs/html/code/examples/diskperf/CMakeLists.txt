# Copyright (c) 2013 Shuangyang Yang
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(HPX_OrangeFS)

if(ORANGEFS_FOUND)

    set(example_programs
    )

    if(HPX_HAVE_CXX11_LAMBDAS)
        set(example_programs
             ${example_programs}
             diskperf_ofs_pxfs_action)
        set(diskperf_ofs_pxfs_action_FLAGS DEPENDENCIES iostreams_component)
    endif()


    foreach(example_program ${example_programs})

        set(${example_program}_FLAGS DEPENDENCIES iostreams_component)

        set(sources ${example_program}.cpp)

        source_group("Source Files" FILES ${sources})

        # add OrangeFS header files directory
        include_directories(${ORANGEFS_INCLUDE_DIR})
        # add OrangeFS lib directories
        link_directories(${ORANGEFS_LIBRARY_DIR})

        # add example executable
        add_hpx_executable(${example_program}
                            SOURCES ${sources}
                             ${${example_program}_FLAGS}
                          FOLDER "Examples/Diskperf/${example_program}")

          # link libraries
          target_link_libraries(${example_program}_exe
                                ${ORANGEFS_LIBRARIES} pthread dl rt crypto)

          # add a custom target for this example
          add_hpx_pseudo_target(examples.diskperf.${example_program})

          # make pseudo-targets depend on master pseudo-target
          add_hpx_pseudo_dependencies(examples.diskperf
                                   examples.diskperf.${example_program})

          # add dependencies to pseudo-target
          add_hpx_pseudo_dependencies(examples.diskperf.${example_program}
                                      ${example_program}_exe)


    endforeach()

endif()

