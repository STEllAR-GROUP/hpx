#!/bin/bash

SCRIPTPATH=$( cd $(dirname $0) ; pwd -P )
DEPS_PATH=$SCRIPTPATH/../.deps
PROGRAM_NAME="tools"

while getopts ":" opt; do
  case $opt in
    a)
      echo "-a was triggered, Parameter: $OPTARG" >&2
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

function assert_single_entry {
    local directory=$1
    if ! [ -d $directory ]; then
        echo "$directory does not exist!"
        exit 1
    fi
    local num_entries=$(cd $directory; ls -l | grep -v "total " | wc -l)
    if [[ "$num_entries" != "1" ]]; then
        echo "$directory contains $num_entries entries, but only one expected!"
        exit 1
    fi
}

NUM_NONOPT_ARGS=$(expr $# - $OPTIND + 1)

if [[ "$NUM_NONOPT_ARGS" != "0" ]]; then
    echo "Needs exactly 0 arguments!"
    exit 1
fi


ROOT_DIR=${DEPS_PATH}/${PROGRAM_NAME}
TMP_DIR=/tmp/${PROGRAM_NAME}
EXISTS_FILE=${DEPS_PATH}/${PROGRAM_NAME}.exists
if [ -f $EXISTS_FILE ]; then
    echo "${PROGRAM_NAME} ${VERSION} already cached."
    exit 0
fi

echo "${PROGRAM_NAME} not cached, building ..."

rm -rf $ROOT_DIR
mkdir -p $ROOT_DIR || exit 1
rm -rf $TMP_DIR
mkdir -p $TMP_DIR || exit 1

mkdir -p $TMP_DIR/wget || exit 1


function download_and_extract {
    local URL=$1
    local TARGET_DIR=$2
    local PARENT_DIR=$(dirname $TARGET_DIR)
    if ! [ -d $PARENT_DIR ]; then
        echo "$PARENT_DIR should exist, but doesn't!"
        exit 1
    fi
    rm -rf $TARGET_DIR
    echo "Downloading $URL ..."
    wget -qO- ${URL} | tar -xz -C $TMP_DIR/wget --checkpoint=1000 || exit 1
    assert_single_entry $TMP_DIR/wget
    mv $TMP_DIR/wget/* $TARGET_DIR || exit 1
}


# cmake
echo "### CMAKE ###"

echo "Cloning ..."
git clone git://cmake.org/cmake.git --depth 1 --branch release $TMP_DIR/cmake || exit 1
cd $TMP_DIR/cmake || exit 1

echo "Configuring ..."
./configure --prefix=${ROOT_DIR} || exit 1

echo "Building ..."
make -j2 || exit 1

echo "Installing ..."
make install || exit 1


# hwloc
echo "### HWLOC ###"

download_and_extract "http://www.open-mpi.org/software/hwloc/v1.10/downloads/hwloc-1.10.1.tar.gz" $TMP_DIR/hwloc
cd $TMP_DIR/hwloc || exit 1

echo "Configuring ..."
./configure --prefix=${ROOT_DIR} || exit 1

echo "Building ..."
make -j2 || exit 1

echo "Installing ..."
make install || exit 1

touch $EXISTS_FILE

exit 0
