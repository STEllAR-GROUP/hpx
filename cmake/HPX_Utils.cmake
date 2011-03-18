# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_UTILS_LOADED TRUE)

include(HPX_Include)

hpx_include(Message
            ListContains
            ParseArguments
            Install
            AddComponent
            AddExecutable
            AddTest
            AddConfigTest
            AddLibrarySources
            CompileObject
            ForceOutOfTreeBuild)

