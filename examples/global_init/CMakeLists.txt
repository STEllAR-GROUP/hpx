# Copyright (c) 2018      Nikunj Gupta
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################

set(example_programs
    factorial
		sierpinski
		)

foreach(example ${example_programs})

    add_hpx_executable(global_${example}
        SOURCES ${example}.cpp
        FOLDER "Examples/Global_init/")

    add_hpx_pseudo_target(examples.global_init.global_${example})
    add_hpx_pseudo_dependencies(examples.global_init examples.global_init.global_${example})
    add_hpx_pseudo_dependencies(examples.global_init.global_${example} global_${example}_exe)

endforeach()
