#!/bin/bash

if [ "$1" != "" ] ; then
	CXXFLAGS="-I $1"
fi


if ! [ -e /usr/include/boost/config.hpp ] && [ "$CXXFLAGS" == "" ] ; then
	echo usage /path/to/boost/root
else
  set -x
	g++ $CXXFLAGS -I . -rdynamic libs/backtrace/src/backtrace.cpp libs/backtrace/test/test_backtrace.cpp -ldl -o test_backtrace
	g++ $CXXFLAGS -I . -rdynamic libs/backtrace/src/backtrace.cpp libs/backtrace/test/test_throw_backtrace.cpp -ldl -o test_throw_backtrace
  set -x
fi

