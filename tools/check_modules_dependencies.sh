#!/bin/bash

# Copyright (c)      2020 STE||AR GROUP
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Enable globbing
shopt -s globstar

source_dir=/hpx/source
pushd $source_dir/libs > /dev/null

# Extract the list of the modules
modules_list=($(find . -maxdepth 1 -type d | sort | tail --lines=+2))

# Find non module headers under the main hpx/ dir to exclude them later
non_module_files_list=($(ls ../hpx | grep .hpp))

# Iterate on all modules of the libs/ dir
for module in "${modules_list[@]}"
do
    # Scan the headers of the module
    module=$(basename $module)
    pushd ${module} > /dev/null

        # Find the dependencies through the includes and remove hpx/hpx_* like
        includes=($(grep -Erho 'hpx/[_a-z]*\.hpp\>' --include=*.{hpp,cpp}\
            include src 2> /dev/null | sort | uniq | grep -v hpx/hpx))

        # Check if the dependency is inside the CMakeLists.txt
        for include in "${includes[@]}"
        do
            # Exclude the headers from the main hpx/ dir
            if [[ ! "${non_module_files_list[@]}" =~ "$(basename $include)" ]]; then
                # Isolate the name of the module from the include
                module_deps=$(basename $include | cut -d'.' -f1)
                # We exclude the main headers and the ones owned by the module
                if [[ ! "$module_deps" == "$module" ]]; then
                    grep -Fq "hpx_${module_deps}" CMakeLists.txt
                    if [[ $? -eq 1 ]]; then
                        echo "Missing hpx_${module_deps} dependency in libs/${module}/CMakeLists.txt"
                    fi
                fi
            fi
        done

    popd > /dev/null

done
