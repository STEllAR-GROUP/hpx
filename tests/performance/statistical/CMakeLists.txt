# Copyright (c) 2011 Bryce Adelstein-Lelbach
# Copyright (c) 2012 Daniel Kogler
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_hpx_component(sender
    SOURCES "parcelsender/sender.cpp"
    FOLDER "Benchmarks/parcel_sr_analysis"
    DEPENDENCIES iostreams_component
    )

add_hpx_component(receiver
    SOURCES "parcelreceiver/receiver.cpp"
    FOLDER "Benchmarks/parcel_sr_analysis"
    DEPENDENCIES iostreams_component
    )

add_hpx_component(empty
    SOURCES "emptycomponent/emptycomponent.cpp"
    FOLDER "Benchmarks/distrib_factory_cdcomp_analysis.cpp"
    DEPENDENCIES iostreams_component
    )

set(benchmarks
#    apply_scheduling_analysis
    create_free_comps_analysis
#    distrib_factory_cdcomp_analysis
    general_apply_analysis
    general_dataflow_analysis
    general_pa_analysis
#    get_credit_analysis
    get_gid_analysis
    parcel_sr_analysis
    spawn_threads_analysis
    thread_cs_analysis
   )

set(apply_scheduling_analysis_FLAGS DEPENDENCIES iostreams_component)
set(create_free_comps_analysis_FLAGS DEPENDENCIES iostreams_component
                                                      empty_component)
set(distrib_factory_cdcomp_analysis_FLAGS DEPENDENCIES iostreams_component
                            empty_component distributing_factory_component)
set(check_address_analysis_FLAGS DEPENDENCIES iostreams_component)
set(general_apply_analysis_FLAGS DEPENDENCIES iostreams_component)
set(general_dataflow_analysis_FLAGS DEPENDENCIES iostreams_component
                                                  dataflow_component)
set(general_pa_analysis_FLAGS DEPENDENCIES iostreams_component)
set(get_credit_analysis_FLAGS DEPENDENCIES iostreams_component
 simple_refcnt_checker_component managed_refcnt_checker_component)
set(get_gid_analysis_FLAGS DEPENDENCIES iostreams_component)
set(parcel_sr_analysis_FLAGS DEPENDENCIES iostreams_component 
                          sender_component receiver_component)
set(spawn_threads_analysis_FLAGS DEPENDENCIES iostreams_component)
set(thread_cs_analysis_FLAGS DEPENDENCIES iostreams_component)

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

set_target_properties(general_dataflow_analysis_exe
    PROPERTIES COMPILE_FLAGS
    "-DHPX_ACTION_ARGUMENT_LIMIT=10 -DHPX_COMPONENT_CREATE_ARGUMENT_LIMIT=10 -DHPX_FUNCTION_ARGUMENT_LIMIT=13")

