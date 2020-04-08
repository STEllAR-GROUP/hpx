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

for module in "${modules_list[@]}"
do
    module=$(basename $module)

    # Check if the include directory exists
    if [ -d "${module}/include" ]; then
        # Scan the headers of the module
        pushd ${module}/include > /dev/null

            module_headers=($(ls **/*.hpp))

            # Check the presence of the header in the CMakeLists.txt of the module
            for header in "${module_headers[@]}"
            do
                grep -Fq "$header" ../CMakeLists.txt
                if [[ $? -eq 1 ]]; then
                    echo "Missing ${header} in libs/${module}/CMakeLists.txt"
                fi
            done

        popd > /dev/null
    fi

    # Check if the src directory exists
    if [ -d "${module}/src" ]; then
        # Scan the sources of the module
        pushd ${module}/src > /dev/null

            module_sources=($(ls **/*.cpp))

            # Check the presence of the source in the CMakeLists.txt of the module
            for sourcefile in "${module_sources[@]}"
            do
                grep -Fq "$sourcefile" ../CMakeLists.txt
                if [[ $? -eq 1 ]]; then
                    echo "Missing ${sourcefile} in libs/${module}/CMakeLists.txt"
                fi
            done

        popd > /dev/null
    fi

done

popd > /dev/null
