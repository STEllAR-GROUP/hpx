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

# This file should be sourced for the sed to work

# Arguments to modify :
module=functional
project_root=~/projects/hpx_module_functional

# Need to specify this to allow globbing
shopt -s globstar

# Retrieve the corresponding new header
function find_matching() {
    for file in "${new_filenames[@]}"; do
        basefile=$(basename $file)
        if [[ "$basefile" = "$1" ]]; then
            new_file=$file
            return
        fi
    done
}

# Activate the ** globing
shopt -s globstar

pushd ${project_root} > /dev/null

# Get all the old headers names
pushd libs/${module}/include_compatibility > /dev/null
old_filenames=($(ls **/*.hpp))

# Get all the new headers names
cd ../include
new_filenames=($(ls **/*.hpp))

popd > /dev/null # go back at the top level project_root

# Introduce backslash in front of a . or a /
for file in "${old_filenames[@]}"; do

    old_file=$file
    basefilename=$(basename $old_file)
    find_matching "$basefilename"
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

done

popd > /dev/null
