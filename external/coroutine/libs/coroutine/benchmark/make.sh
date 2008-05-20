#!/bin/sh
BOOST_ROOT=~/src/boost bjam -sBUILD="<optimization>speed <inlining>on" -d2
