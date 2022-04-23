@REM This is an example cmd.exe batch script
@REM   that uses hpxdep.exe to generate a
@REM   complete HPX dependency report.
@REM 
@REM It needs to be run from the HPX root.
@REM 
@REM Copyright 2022 Hartmut Kaiser
@REM Copyright 2014, 2015, 2017 Peter Dimov
@REM 
@REM SPDX-License-Identifier: BSL-1.0
@REM Distributed under the Boost Software License, Version 1.0.
@REM See accompanying file LICENSE_1_0.txt or copy at
@REM http://www.boost.org/LICENSE_1_0.txt

SET HPXDEP=hpxdep.exe
SET OPTIONS=--hpx-root %1 --hpx-build-root %2
SET OUTDIR=.\report

mkdir %OUTDIR%
mkdir %OUTDIR%\core
mkdir %OUTDIR%\full

%HPXDEP% --list-modules > %OUTDIR%\list-modules.txt

%HPXDEP% %OPTIONS% --html-title "HPX Module Overview" --html --module-overview > %OUTDIR%\module-overview.html
%HPXDEP% %OPTIONS% --html-title "HPX Module Levels" --html --module-levels > %OUTDIR%\module-levels.html
%HPXDEP% %OPTIONS% --html-title "HPX Module Weights" --html --module-weights > %OUTDIR%\module-weights.html

FOR /f %%i IN (%OUTDIR%\list-modules.txt) DO %HPXDEP% --html-title "HPX Dependency Report for %%i" %OPTIONS% --html --primary %%i --secondary %%i --reverse %%i > %OUTDIR%\%%i.html
