# Copyright (c) 2011 Bryce Adelstein-Lelbach
# Copyright (c) 2012 Daniel Kogler
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks
    #apply_scheduling_analysis
    check_address_analysis
    general_apply_analysis
    general_pa_analysis
    get_gid_analysis
    parcel_sr_smp_analysis
   )

#set(apply_scheduling_analysis_FLAGS DEPENDENCIES iostreams_component)
set(check_address_analysis_FLAGS DEPENDENCIES iostreams_component)
set(general_apply_analysis_FLAGS DEPENDENCIES iostreams_component)
set(general_pa_analysis_FLAGS DEPENDENCIES iostreams_component)
set(get_gid_analysis_FLAGS DEPENDENCIES iostreams_component)
set(parcel_sr_smp_analysis_FLAGS DEPENDENCIES iostreams_component)


foreach(benchmark ${benchmarks})
  set(sources
      ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${benchmark}
                     SOURCES ${sources}
                     ${${benchmark}_FLAGS}
                     FOLDER "Benchmarks/${benchmark}")

  # add a custom target for this example
  add_hpx_pseudo_target(tests.performance.statistical.${benchmark})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.statistical
                              tests.performance.statistical.${benchmark})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.statistical.${benchmark}
                              ${benchmark}_exe)
endforeach()

add_hpx_component(sender
    SOURCES "sender.cpp"
    FOLDER "Benchmarks/parcel_sr_dis")
   # SOURCE_GLOB "sender.cpp")

add_hpx_component(receiver
    SOURCES "receiver.cpp"
    HEADERS "receiver.hpp"
    #SOURCE_GLOB "receiver.cpp"
    #HEADER_GLOB "receiver.hpp"
    FOLDER "Benchmarks/parcel_sr_dis")
    #AUTOGLOB)

add_hpx_executable(parcel_sr_dis_analysis
    SOURCES parcel_sr_dis_analysis.cpp
    DEPENDENCIES iostreams_component sender_component receiver_component
    FOLDER "Benchmarks/parcel_sr_dis")

add_hpx_pseudo_target(tests.performance.statistical.parcel_sr_dis_analysis)
add_hpx_pseudo_dependencies(tests.performance.statistical 
                            tests.performance.statistical.parcel_sr_dis_analysis)
add_hpx_pseudo_dependencies(tests.performance.statistical.parcel_sr_dis_analysis
                            parcel_sr_dis_analysis_exe)
