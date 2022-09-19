set(tests
    transform_binary2_unseq
    transform_binary_unseq
    transform_unseq)

foreach(test ${tests})
  set(sources ${test}.cpp)

  set(${test}_PARAMETERS THREADS_PER_LOCALITY 4)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Modules/Core/Algorithms/Unseq"
  )

  add_hpx_unit_test(
    "modules.algorithms.unseq_algorithms" ${test} ${${test}_PARAMETERS}
  )
endforeach()