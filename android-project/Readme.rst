.. Copyright (c) 2012 Thomas Heller

   Distributed under the Boost Software License, Version 1.0. (See accompanying
   file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

*****
 HPX on Android
*****

This folder contains Makefile definitions in order to get HPX build for an
Android device. Once hpx is included in your applications Android.mk, it takes
care of building the required boost dependencies.

**NOTE:** This has only been tested with boost trunk and a Samsung Galaxy Note
10.1 (Running Android 4.0.4) and the latest Android NDK (r8b)

*****
 Build Instructions
*****

1) Get the Android NDK
(`Download NDK <http://developer.android.com/tools/sdk/ndk/index.html>`_)
and follow the instruction on this site on how to install it.

2) Clone the master HPX git repository::
    $ git clone https://github.com/STEllAR-GROUP/hpx.git

3) Checkt out Boost trunk::
    $ svn co http://svn.boost.org/svn/boost/trunk boost-trunk

4) Patch boost::
    $ cd BOOST_SRC_ROOT
    $ patch -p0 HPX_ROOT/android-project/boost-android-ndk.patch

5) In Your Applications Android.mk add the following::
    MY_PATH := $(call my-dir)
    $(call import-module,hpx)
    LOCAL_PATH := $(MY_PATH)
    # ...
    LOCAL_C_INCLUDES:=$(HPX_INCLUDES)

    LOCAL_CPPFLAGS:=$(HPX_CPPFLAGS)
    LOCAL_CPPFLAGS+=-DHPX_APPLICATION_NAME=application_name
    LOCAL_CPPFLAGS+=-DHPX_APPLICATION_STRING=\"application_name\"
    LOCAL_CPPFLAGS+=-DHPX_APPLICATION_EXPORTS
    LOCAL_CPPFLAGS+=-DBOOST_ENABLE_ASSERT_HANDLER
    LOCAL_CPPFLAGS+=-DPAGE_SIZE="(1UL << 12)"

    LOCAL_CPP_FEATURES:= exceptions rtti

6) In order for the Android NDK to find hpx, you need to set the NDK_MODULE_PATH 
environment variable to point to HPX_ROOT/android-project and set the
BOOST_SRC_ROOT to point to your patched source code version of boost.

7) Run ndk-build in your applications root directory

