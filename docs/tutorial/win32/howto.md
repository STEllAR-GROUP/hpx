
//  Copyright (c) 2015 Harris Brakmic<br/>
//<br/>
//  Distributed under the Boost Software License, Version 1.0. (See accompanying<br/>
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)<br/>
<br/>
**How to build HPX under Windows 10 x64 with Visual Studio 2015**

* a) Download the *CMake 3.4.3* installer from <a href="https://blog.kitware.com/cmake-3-4-3-available-for-download/" target="_blank">here</a> <br/>
* b) Download *Portable Hardware Locality* from <a href="http://www.open-mpi.org/software/hwloc/v1.11/downloads/hwloc-win64-build-1.11.0.zip">here</a> and unpack it.<br/>
* c) Download *Boost libraries* from <a href="https://sourceforge.net/projects/boost/files/boost/1.60.0/">here</a> and unpack them.<br/>
* d) *Build* the boost DLLs and LIBs by using these commands from Command Line (or PowerShell):<br/>

Open CMD/PowerShell inside the Boost dir and type in:<br/>
```code 
bootstrap.bat
```
*This batch file will set up everything needed to create a successful build.*<br/>
```code
b2.exe link=shared variant=release,debug architecture=x86 address-model=64 threading=multi --build-type=complete install
```
*This command will start a (very long) build of all available Boost libraries. Please, be patient.*<br/>
Many thanks to @biddisco and @hkaiser for providing the needed info on how to combine all these flags. :thumbsup:

* e) Open CMake-GUI.exe and set up your *HPX source dir* to the **base dir** of the source code you downloaded from HPX' GitHub pages. <br/>
   Here's an example of my CMake path settings which point to my *Documents/GitHub/hpx* folder.<br/>
   
   <img src="http://fs2.directupload.net/images/150813/ldi6oedi.png" width="350" height="400"><br/>
   Inside the **Where is the source-code** put in the *ROOT path* of your HPX source directory (do not put the "src" subdir!)<br/>
   Inside **Where to build the binaries** you should put in the path where all the building process will happen. <br/>
   This is important because the building machinery will do an "out-of-tree" build. <br/><br/>
   CMake is not touching or changing in any way the original source files. Instead, it will generate Visual Studio Solution Files
   which will build HPX packages out of the HPX source tree.<br/>
d) Set four new environment variables (in CMake, not in Windows environment, by the way): **BOOST_ROOT, HWLOC_ROOT, CMAKE_INSTALL_PREFIX** and **HPX_WITH_BOOST_ALL_DYNAMIC_LINK**<br/>    

The meaning of these variables is as follows:<br/>
**BOOST_ROOT**: the root directory of the unpacked Boost headers/cpp files.<br/>
**HWLOC_ROOT**: the root directory of the unpacked Portable Hardware Locality files.<br/>
**CMAKE_INSTALL_PREFIX**: the "root directory" where the future builds of HPX should land.<br/><br/> 
*Hint*: *HPX is a BIG software collection and I really don't recommend using the default *C:\Program Files\hpx* <br/>
I prefer simpler paths *without* white space, like C:\bin\hpx or D:\bin\hpx etc. <br/>
To insert nev env-vars klick on "Add Library" and then insert the name inside "Name", select *PATH* as Type and put the path-name in "Path" text field.<br/>
Repeat this for the first three variables. <br/>
The last one: *HPX_WITH_BOOST_ALL_DYNAMIC_LINK* is a *BOOL* and must be checked (there will be a checkbox instead of a textfield)  <br/><br/>
*This is how variable insertion looks like*<br/>
<img src="http://fs2.directupload.net/images/150813/cf4kfips.png" width="350" height="400"><br/><br/>
Alternatively you could provide BOOST_LIBRARYDIR instead of BOOST_ROOT with a difference that LIBRARYDIR points to the sudbirectory inside 
Boost root where all the compiled DLLs/LIBs are. <br/>
I myself have used *BOOST_LIBRARYDIR* which pointed to the *bin.v2* subdirectory under the Boost rootdir.<br/>
Important is to keep the meanings of these two variables separated from each other: <br/>
BOOST_DIR points to the ROOT folder of the boost library. <br/>
BOOST_LIBRARYDIR points to the subdir inside Boost root folder where the compiled binaries are.<br/>
* f) Click the *Configure* button of CMake-GUI. You will be immediately presented a small window <br/>
where you can select the C++ compiler to be used within Visual Studio. <br/>In my case I have used the latest v14 (a.k.a C++ 2015) but older versions should be sufficient too.<br/>
* g) After the generate process has finished successfully click the *Generate* button.  Now, CMake will put new VS Solution files into the BUILD folder you selected at the beginning.<br/>
* h) Open Visual Studio and load the **HPX.sln** from your build folder.<br/>
* i) Go to *CMakePredefinedTargets* and build the **INSTALL** project<br/><br/>
<img src="http://fs2.directupload.net/images/150813/q9bcypwg.png" width="350" height="350"><br/>
It will take some time to compile everything and in the end you should see an output similar to this one:<br/>
<img src="http://fs2.directupload.net/images/150813/qa9pxark.png" width="450" height="300">
