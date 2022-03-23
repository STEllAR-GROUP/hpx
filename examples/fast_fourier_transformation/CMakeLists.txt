# Copyright (c) 2022 Deepak Suresh

# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (example_programs fft)

foreach(example_program ${example_programs})
  set(${example_program}_PARAMETERS THREADS_PER_LOCALITY 4)

  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

   # add example executable
   add_hpx_executable(
    ${example_program} INTERNAL_FLAGS
    SOURCES ${sources} ${${example_program}_FLAGS}
    FOLDER "Examples/fast_fourier_transformation/${example_program}"
  )


add_hpx_example_target_dependencies("fft" ${example_program})

endforeach()
