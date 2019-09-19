#!/usr/bin/env bash

# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Script to introduce the deprecated include after factorazing in modules
# This file is globbing through the include_compatibility folder, so basic 
# files should already be here

script_sourced=0
# Important to be at the beginning
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    script_sourced=1
fi

function _exit() {
    if [[ $script_sourced -eq 0 ]]; then
        exit
    fi
}

if [[ $# -lt 1 ]]; then
    arg=${BASH_SOURCE[0]}
    echo "Usage : "$arg" -m <module_name> --old_path <include_path> --new_path <include_path>"
    echo "Example: "$arg" -m cache -o hpx/util/cache -n hpx/cache"
    echo "Can specify the --project_path if different from $HPX_ROOT"
    _exit
    return
fi

function parse_arguments() {

    # store arguments list
    POSITIONAL=()

    while [[ $# -gt 0 ]]
    do
        local key="$1"
        case $key in
            -m|--module)
                module=$2
                echo "module : ${module}"
                shift # pass option
                shift # padd value
                ;;
            -n|--new_path)
                new_path=$2
                echo "new_path : ${new_path}"
                shift # pass option
                shift # padd value
                ;;
            -o|--old_path)
                old_path=$2
                echo "old_path : ${old_path}"
                shift # pass option
                shift # padd value
                ;;
            -p|--project_path)
                project_path=$2
                echo "project_path: ${project_path}"
                shift # pass option
                shift # padd value
                ;;
            --help|*)
                echo $"Usage: $0 [-m, --module <value>] [-o, --old_path <value>] \
                [-n, --new_path <value>] [-p, --project_path <value>]"
                echo "Example: "$0" -m cache -o hpx/util/cache -n hpx/cache"
                echo "Can specify the --project_path if different from $HPX_ROOT"
                _exit
                return
        esac
    done

    # restore positional parameters
    set -- "${POSITIONAL[@]}"

}

# Default values which can be overwritten while parsing args
project_path=$HPX_ROOT
module=cache
old_path=hpx/util/${module}
new_path=hpx/${module}
# Default is globbing compatibility files
all_files=1
files=

# Parsing arguments
parse_arguments "$@"

# Usual vars (depend on the parsing step)
libs_path=$project_path/libs
module_path=$libs_path/${module}
module_caps=${module^^}

# Project path not set (full specified path to be sure which source is used)
if [[ -z $HPX_ROOT && -z $project_path ]]; then
    "HPX_ROOT env var doesn't exists and project_path option not specified !"
    _exit
    return
fi

pushd $module_path/include_compatibility/${old_root}
if [[ $? -eq 1 ]]; then
    echo "Please specify a correct project_path"
    _exit
    return
fi
# To enable **
shopt -s globstar
# Globbing step to get all the include_compatibility files, the files have to
# already be there, we are just rewriting them
if [[ all_files -eq 0 ]]; then
    # To set the files manually, don't forget to put the subdirs (of old_root) if there is
    files=(histogram rolling_max rolling_min)
    files=(${files[@]/%/.hpp})
else
    files=($(ls **/*.hpp))
fi

echo "Files overwritten :"
# Create the corresponding compatibility headers
for full_file in "${files[@]}"; do
    # basename not used otherwise we lose hierarchy if any
    f=${full_file#"$old_path/"}
    echo $full_file

# No indentation otherwise it appears in the cat
cat >${full_file} <<EOL
//  Copyright (c) 2019 Ste||ar Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/${module}/config/defines.hpp>
#include <${new_path}/${f}>

#if defined(HPX_${module_caps}_HAVE_DEPRECATION_WARNINGS)
#if defined(HPX_MSVC)
#pragma message( \\
    "The header ${old_path}/${f} is deprecated, \\
    please include ${new_path}/${f} instead")
#else
#warning \\
    "The header ${old_path}/${f} is deprecated, \\
    please include ${new_path}/${f} instead"
#endif
#endif
EOL

done

popd
