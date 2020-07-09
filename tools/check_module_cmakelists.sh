#!/bin/bash

# Copyright (c)      2020 STE||AR GROUP
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function check_failure {
    string_to_grep=$1
    tmp_module=$2
    cmakelist=$3
    grep -Fq "$string_to_grep" $cmakelist
    if [[ $? -eq 1 ]]; then
        echo "Missing ${string_to_grep} in libs/${tmp_module}/CMakeLists.txt"
    fi
}

function check_module_dependencies() {
    tmp_module=$1
    non_module_files_list=$2

    # Find the dependencies through the includes and remove hpx/hpx_* like
    includes=($(grep -Erho 'hpx/modules/[_a-z]*\.hpp\>' --include=*.{hpp,cpp}\
        include src 2> /dev/null | sort | uniq | grep -v hpx/hpx))

    # Check if the dependency is inside the CMakeLists.txt
    for include in "${includes[@]}"
    do
        # Exclude the headers from the main hpx/ dir
        if [[ ! "${non_module_files_list[@]}" =~ "$(basename $include)" ]]; then
            # Isolate the name of the module from the include
            module_deps=$(basename $include | cut -d'.' -f1)
            # Check if the name is not the current module
            if [[ ! "$module_deps" == "$tmp_module" ]]; then
                check_failure "hpx_${module_deps}" $tmp_module CMakeLists.txt
            fi
        fi
    done
}

function check_cmakelists_files() {
    tmp_module=$1
    tmp_dir=$2
    if [ -d $tmp_dir ]; then
        pushd $tmp_dir > /dev/null

            # Silence the .cpp not found errors in include dir and inversely
            module_files=($(ls **/*.{hpp,cpp} 2> /dev/null))
            # Check the presence of the header in the CMakeLists.txt of the module
            for header in "${module_files[@]}"
            do
                if [[ ! "$header" =~ "detail" ]]; then
                    check_failure $header $tmp_module ../CMakeLists.txt
                fi
            done

        popd > /dev/null
    fi
}

########
# MAIN #
########

# Enable globbing
shopt -s globstar

source_dir=/hpx/source
pushd $source_dir/libs > /dev/null

# Extract the list of the modules
modules_list=($(find . -maxdepth 1 -type d | sort | tail --lines=+2))

# Find non module headers under the main hpx/ dir to exclude them later
non_module_files_list=($(ls ../hpx | grep .hpp))

echo "" > /tmp/missing_files.txt
echo "" > /tmp/missing_deps.txt

# Iterate on all modules of the libs/ dir
for module in "${modules_list[@]}"
do
    module=$(basename $module)
    pushd ${module} > /dev/null

        check_module_dependencies $module $non_module_files_list >> /tmp/missing_deps.txt
        check_cmakelists_files $module include >> /tmp/missing_files.txt
        check_cmakelists_files $module src >> /tmp/missing_files.txt

    popd > /dev/null

done
