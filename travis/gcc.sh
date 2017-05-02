#!/bin/bash

SCRIPTPATH=$( cd $(dirname $0) ; pwd -P )
DEPS_PATH=$SCRIPTPATH/../.deps
PROGRAM_NAME="gcc"

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
    mv $TMP_DIR/wget/* ${TARGET_DIR} || exit 1
}

GCC_URL="http://ftpmirror.gnu.org/gcc/gcc-${VERSION}/gcc-${VERSION}.tar.gz"
download_and_extract $GCC_URL $TMP_DIR/src

echo "Downloading prerequisites ..."
cd $TMP_DIR/src
./contrib/download_prerequisites

echo "Creating build directory ..."
mkdir $TMP_DIR/build || exit 1
cd $TMP_DIR/build || exit 1



echo "Configuring ..."
../src/configure --prefix=${ROOT_DIR}                               \
                 --enable-languages=c,c++                           \
                 --disable-multilib                                 \
                 || exit 1

echo "Building ..."
# Prevent travis timeout
trap 'kill $(jobs -p)' EXIT
while true; do sleep 300; echo "Still building ..."; done &
make -j2 >/dev/null || exit 1

echo "Installing to deps folder ..."
make install || exit 1

touch $EXISTS_FILE

exit 2
