#! /bin/bash
#
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file BOOST_LICENSE_1_0.rst or copy at http://www.boost.org/LICENSE_1_0.txt)

VERSION_MAJOR=`grep '#define HPX_VERSION_MAJOR' hpx/version.hpp | awk {' print $3 '}`
VERSION_MINOR=`grep '#define HPX_VERSION_MINOR' hpx/version.hpp | awk {' print $3 '}`
VERSION_SUBMINOR=`grep '#define HPX_VERSION_SUBMINOR' hpx/version.hpp | awk {' print $3 '}`

DOT_VERSION=$VERSION_MAJOR.$VERSION_MINOR.$VERSION_SUBMINOR
DASH_VERSION=$VERSION_MAJOR-$VERSION_MINOR-$VERSION_SUBMINOR

WEBSITE="http://stellar.cct.lsu.edu"

ZIP=hpx_$DOT_VERSION.zip
TARGZ=hpx_$DOT_VERSION.tar.gz
TARBZ2=hpx_$DOT_VERSION.tar.bz2
SEVENZ=hpx_$DOT_VERSION.7z

rm -rf packages
mkdir packages
mkdir packages/zip
mkdir packages/tar.gz
mkdir packages/tar.bz2
mkdir packages/7z

echo -n "Packaging $ZIP... "
zip -q -x .git\* -x packages -x packages/\* -r packages/$ZIP .
echo "DONE"

echo -n "Packaging $TARGZ... "
tar --exclude=.git\* --exclude=packages --exclude=packages/\* -czf packages/$TARGZ .
echo "DONE"

echo -n "Packaging $TARBZ2... "
tar --exclude=.git\* --exclude=packages --exclude=packages/\* -cjf packages/$TARBZ2 .
echo "DONE"

echo -n "Packaging $SEVENZ... "
7zr a -xr\!.git -xr\!packages packages/$SEVENZ . > /dev/null 
echo "DONE"

(cd packages/zip && unzip -qq ../$ZIP)
(cd packages/tar.gz && tar -xf ../$TARGZ)
(cd packages/tar.bz2 && tar -xf ../$TARBZ2)
(cd packages/7z && 7zr x ../$SEVENZ > /dev/null)

echo "<ul>"
echo "    <li>HPX V$DOT_VERSION: <a title=\"HPX V$DOT_VERSION Release Notes\" href=\"$WEBSITE/downloads/hpx-v$DASH_VERSION-release-notes/\">release notes</a>"
echo "    <table>"
echo "        <tr><th>File</th><th>MD5 Hash</th></tr>"
echo "        <tr><td><a title=\"HPX V$DOT_VERSION (zip)\" href=\"$WEBSITE/files/$ZIP\">zip</a></td><td><code>`md5sum packages/$ZIP | awk {'print $1'}`</code></td></tr>"
echo "        <tr><td><a title=\"HPX V$DOT_VERSION (gz)\" href=\"$WEBSITE/files/$TARGZ\">gz</a></td><td><code>`md5sum packages/$TARGZ | awk {'print $1'}`</code></td></tr>"
echo "        <tr><td><a title=\"HPX V$DOT_VERSION (bz2)\" href=\"$WEBSITE/files/$TARBZ2\">bz2</a></td><td><code>`md5sum packages/$TARBZ2 | awk {'print $1'}`</code></td></tr>"
echo "        <tr><td><a title=\"HPX V$DOT_VERSION (7z)\" href=\"$WEBSITE/files/$SEVENZ\">7z</a></td><td><code>`md5sum packages/$SEVENZ | awk {'print $1'}`</code></td></tr>"
echo "    </table>"
echo "    </li>"
echo "</ul>"

