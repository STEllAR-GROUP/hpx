#!/bin/bash

SCRIPTPATH=$( cd $(dirname $0) ; pwd -P )
DEPS_PATH=$SCRIPTPATH/../.deps
PROGRAM_NAME="boost"

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

if [[ "$NUM_NONOPT_ARGS" != "1" ]]; then
    echo "Needs exactly one argument!"
    exit 1
fi


VERSION=${@:${OPTIND}:1}
UNDERSCORE_VERSION=${VERSION//./_}

ROOT_DIR=${DEPS_PATH}/${PROGRAM_NAME}-${VERSION}
TMP_DIR=/tmp/${PROGRAM_NAME}-${VERSION}
EXISTS_FILE=${DEPS_PATH}/${PROGRAM_NAME}-${VERSION}.exists
if [ -f $EXISTS_FILE ]; then
    echo "${PROGRAM_NAME} ${VERSION} already cached."
    exit 0
fi

echo "${PROGRAM_NAME} ${VERSION} not cached, building ..."

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

BOOST_URL="http://downloads.sourceforge.net/sourceforge/boost/boost/${VERSION}/boost_${UNDERSCORE_VERSION}.tar.gz"
download_and_extract $BOOST_URL $TMP_DIR/src
cd $TMP_DIR/src || exit 1

echo "Bootstrapping ..."
EXCLUDES="mpi,graph_parallel,python"
./bootstrap.sh --without-libraries=${EXCLUDES} --prefix=${ROOT_DIR} || exit 1

trap 'kill $(jobs -p)' EXIT
while true; do sleep 300; echo "Still building ..."; done &

echo "Building Debug ..."
./b2 variant=debug --prefix=${ROOT_DIR}/Debug -j2 >/dev/null || exit 1

echo "Building Release ..."
./b2 variant=release --prefix=${ROOT_DIR}/Release -j2 >/dev/null || exit 1


echo "Installing Debug to deps folder ..."
./b2 variant=debug --prefix=${ROOT_DIR}/Debug install >/dev/null || exit 1

echo "Installing Release to deps folder ..."
./b2 variant=release --prefix=${ROOT_DIR}/Release install >/dev/null || exit 1

touch $EXISTS_FILE

exit 2
