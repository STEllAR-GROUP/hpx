#!/bin/bash
#  Copyright (c) 2016 Thomas Heller
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


FILES=$(grep '"file":' compile_commands.json | awk '{print $2}' | tr -d '"' | sort)

if [[ x"${1}" == x"-diff-master" ]]
then
    SRC_PATH=${2}
    if [[ x"${SRC_PATH}" == x"" ]]
    then
        if [[ -f CMakeCache.txt ]]
        then
            SRC_PATH=$(grep "CMAKE_HOME_DIRECTORY" CMakeCache.txt | awk -F '=' '{print $2}')
        else
            echo "Not executing in the build directory, can not determine source path!"
            exit 1
        fi
    fi
    cd ${SRC_PATH}
    CHANGED_FILES=$(git diff --name-only origin/master | grep ".cpp$")
    CHANGED_FILES=$(echo ${CHANGED_FILES} | awk -v var=${PWD} '{printf("%s/%s\n", var, $1)}')
    FILES=$(comm -12 <(echo "${FILES}" | sort) <(echo "${CHANGED_FILES}" | sort))
    cd -
fi

# Filter out header tests ...
FILES=$(echo "${FILES}" | grep -v "tests/headers")
if [[ x"${EXTERN_FILTER}" != x"" ]]
then
    FILES=$(echo "${FILES}" | grep ${EXTERN_FILTER})
fi

NUM_FILES=$(echo "${FILES}" | wc -l)
echo "Checking ${NUM_FILES} files"

RESULT=0
CHECKS="-*"
CHECKS="$CHECKS,modernize-use-nullptr"
CHECKS="$CHECKS,misc-use-after-move"
CHECKS="$CHECKS,misc-virtual-near-miss"
CHECKS="$CHECKS,misc-multiple-statement-macro"
CHECKS="$CHECKS,misc-move-constructor-init"
CHECKS="$CHECKS,misc-move-forwarding-reference"
CHECKS="$CHECKS,misc-assert-side-effect"
CHECKS="$CHECKS,misc-dangling-handle"
CHECKS="$CHECKS,misc-non-copyable-objects"
CHECKS="$CHECKS,misc-forwarding-reference-overload"
#CHECKS="$CHECKS,bugprone-integer-division"
#CHECKS="$CHECKS,bugprone-suspicious-memset-usage"
#CHECKS="$CHECKS,bugprone-undefined-memory-manipulation"
#CHECKS="$CHECKS,cert-err34-c"
#CHECKS="$CHECKS,cert-err52-cpp"
#CHECKS="$CHECKS,cert-err58-cpp"
#CHECKS="$CHECKS,cert-err60-cpp"
#CHECKS="$CHECKS,cppcoreguidelines-interfaces-global-init"
#CHECKS="$CHECKS,cppcoreguidelines-pro-bounds-constant-array-index"
#CHECKS="$CHECKS,cppcoreguidelines-pro-bounds-pointer-arithmetic"
#CHECKS="$CHECKS,cppcoreguidelines-pro-type-member-init"
#CHECKS="$CHECKS,cppcoreguidelines-pro-type-reinterpret-cast"
#CHECKS="$CHECKS,cppcoreguidelines-slicing"
#CHECKS="$CHECKS,hicpp-signed-bitwise"
#CHECKS="$CHECKS,misc-definitions-in-headers"
#CHECKS="$CHECKS,misc-fold-init-type"
#CHECKS="$CHECKS,misc-forward-declaration-namespace"
#CHECKS="$CHECKS,misc-inaccurate-erase"
#CHECKS="$CHECKS,misc-incorrect-roundings"
#CHECKS="$CHECKS,misc-inefficient-algorithm"
#CHECKS="$CHECKS,misc-misplaced-widening-cast"
#CHECKS="$CHECKS,misc-redundant-expression"
#CHECKS="$CHECKS,misc-sizeof-container"
#CHECKS="$CHECKS,misc-sizeof-expression"
#CHECKS="$CHECKS,misc-string-compare"
#CHECKS="$CHECKS,misc-string-constructor"
#CHECKS="$CHECKS,misc-string-integer-assignment"
#CHECKS="$CHECKS,misc-string-literal-with-embedded-nul"
#CHECKS="$CHECKS,misc-suspicious-enum-usage"
#CHECKS="$CHECKS,misc-suspicious-missing-comma"
#CHECKS="$CHECKS,misc-suspicious-semicolon"
#CHECKS="$CHECKS,misc-suspicious-string-compare"
#CHECKS="$CHECKS,misc-swapped-arguments"
#CHECKS="$CHECKS,misc-undelegated-constructor"
#CHECKS="$CHECKS,misc-unused-raii"
# CHECKS="$CHECKS,"

i=1
for file in $FILES
do
    which bc > /dev/null
    if [[ $? == 0 ]]
    then
        percentage=$(echo "scale=2; ${i}/${NUM_FILES} * 100" | bc | cut -d '.' -f 1)
        percentage=" ${percentage}%"
    else
        percentage=""
    fi
    echo -n "[${i}/${NUM_FILES}${percentage}] ${file}:"
    OUT=$(clang-tidy -header-filter=".*hpx.*" -p . -warnings-as-errors="*" -checks="${CHECKS}" ${file} 2>&1);
    if [[ $? != 0 ]]
    then
        echo ""
        echo "${OUT}"
        RESULT=1
    else
        echo " Nothing found"
    fi
    i=$((i + 1))
done

exit ${RESULT}
