#!/usr/bin/perl -w
## Copyright (c) 2007--2012 Matthew Anderson
##
## Distributed under the Boost Software License, Version 1.0. (See accompanying
## file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

@cpp_files = qw(point.cpp);
@fortran_files = qw(setup.F90);
# Return the original files to their normal name
foreach $file (@cpp_files) {
  $nn = $file . ".orig";
  rename($nn,$file);
}
foreach $file (@fortran_files) {
  $nn = $file . ".orig";
  rename($nn,$file);
}

# get rid of the perl generated source
foreach $file (@fortran_files) {
  ($name,$suffix) = split(/\./,$file);
  $fileout = $name . "_pl." . $suffix;
  unlink($fileout);
}

foreach $file (@cpp_files) {
  ($name,$suffix) = split(/\./,$file);
  $fileout = $name . "_pl." . $suffix;
  unlink($fileout);
}
