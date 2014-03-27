# Copyright (c) 2007-2014 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################

add_hpx_executable(
    network_storage
    AUTOGLOB
    FOLDER "Examples/NetworkStorage"
    DEPENDENCIES iostreams_component)

# add pseudo-target dependencies
add_hpx_pseudo_dependencies(examples.network_storage network_storage_exe)

