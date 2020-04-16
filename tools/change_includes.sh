#!/bin/bash

# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This script aims at replacing the old headers by the new one

# /!\ This file is globbing through the include_compatibility folder, so basic 
# files should already be here

# /!\ The sed command will replace all deprecated headers, including the one
# specified in the deprecation message in include_compatibility (so it is better
# to execute add_compat_headers.sh after this script)

# Not used in default mode (globbing), used if --files option specified
old_filenames=(
  hpx/util/cache/local_cache.hpp
  hpx/util/cache/lru_cache.hpp
)
new_filenames=(
  hpx/cache/local_cache.hpp
  hpx/cache/lru_cache.hpp
)

function extra_usage_message() {
    echo
    echo "Can specify the --project_path if different from the \$PWD variable"
    echo
    echo "In case you want to specify the files to replace manually, please"
    echo "specify them at the beginning of this script source file $0 and use"
    echo "the --files option"
}

if [[ $# -lt 1 ]]; then
    arg=${BASH_SOURCE[0]}
    echo "Usage : "$arg" -m <module_name> -p <project_path>"
    echo "Example: "$arg" -m cache"
    extra_usage_message
    exit
fi

function parse_arguments() {

    # store arguments list
    POSITIONAL=()

    while [[ $# -gt 0 ]]
    do
        local key="$1"
        case $key in
            -f|--files)
                all_files=0
                echo "Replacement based on manually specified files"
                echo "(change directly those in the script $0)"
                shift # pass option
                ;;
            -m|--module)
                module=$2
                echo "module : ${module}"
                shift # pass option
                shift # pass value
                ;;
            -p|--project_path)
                project_path=$2
                shift # pass option
                shift # pass value
                ;;
            --help|*)
                echo $"Usage: $0 [-m, --module <value>] [-p, --project_path <value>]"
                echo "[-f, --files \"<value1> <value2>\"]"
                echo "Example: "$0" -m cache -p \$PWD"
                echo
                echo "- Can specify the --project_path if different from the environmental"
                echo "variable \$PWD"
                exit
                return
        esac
    done

    # restore positional parameters
    set -- "${POSITIONAL[@]}"

}

# Retrieve the corresponding new header
function find_matching() {
    new_file=""
    notfound=false
    for file in "${new_filenames[@]}"; do
        basefile=$(basename $file)
        if [[ "$basefile" = "$1" ]]; then
            new_file=$file
            return
        fi
    done
    # In case no matching file is found in the list specified
    if [[ "$new_file" = "" ]] && [[ $all_files -eq 0 ]]; then
        notfound=true
    fi
}

########
# MAIN #
########

# Defaults arguments
module=
project_path=$PWD
all_files=1 # default is globbing

echo
# Parsing arguments
parse_arguments "$@"

echo "project_path: ${project_path}"
echo

# Activate the ** globing
shopt -s globstar

if [[ $all_files -eq 1 ]]; then
    echo
    echo "Globbing has been specified, we will glob in include/ and"
    echo "include_compatibility/"
    echo
    pushd ${project_path} > /dev/null

    # Get all the old headers names
    pushd libs/${module}/include_compatibility > /dev/null
    old_filenames=($(ls **/*.hpp))

    # Get all the new headers names
    cd ../include
    new_filenames=($(ls **/*.hpp))

    popd > /dev/null # go back at the top level project_path
fi

name_it=0
# Introduce backslash in front of a . or a /
for file in "${old_filenames[@]}"; do

    old_file=$file
    basefilename=$(basename $old_file)
    find_matching "$basefilename"
    if [[ "$notfound" = "true" ]]; then
        new_file=${new_filenames[$name_it]}
        echo "new file !!!" $new_file
        echo "(not found in the list specified)"
    fi
    echo "old header : $old_file"
    echo "new header : $new_file"
    echo
    # Add backslash in front of the special chars
    old_file=${old_file////\\/}
    old_file=${old_file//./\\.}
    new_file=${new_file////\\/}
    new_file=${new_file//./\\.}

    # Replace by the new header in all hpp and cpp files
    sed -i "s/$old_file/$new_file/" **/*.{hpp,cpp}
    name_it=$((name_it+1))

done

popd > /dev/null
