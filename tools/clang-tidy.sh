#!/bin/bash
#  Copyright (c) 2016 Thomas Heller
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


FILES=$(grep '"file":' compile_commands.json | awk '{print $2}' | tr -d '"')

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
    FILES=$(comm -12 <(echo "${FILES}") <(echo "${CHANGED_FILES}"))
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
CHECKS="-*,modernize-use-nullptr"

i=1
for file in $FILES
do
    percentage=$(echo "scale=2; ${i}/${NUM_FILES} * 100" | bc | cut -d '.' -f 1)
    echo -n "[${i}/${NUM_FILES} ${percentage}%] ${file}:"
    OUT=$(clang-tidy -header-filter=".*hpx.*" -p . -checks="${CHECKS}" ${file} 2>&1);
    echo ${OUT} | grep -vq "warning:"
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
