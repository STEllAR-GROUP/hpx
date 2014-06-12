# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_Include)

hpx_include(Message
            Option
            ListContains
            ParseArguments
            AppendProperty
            AppendLibraries
            Install
            GetIncludeDirectory
            Compile
            TargetPaths
            AddCompileFlag
            AddLinkFlag
            AddComponent
            AddDefinitions
            AddLibrary
            AddExecutable
            AddConfigTest
            AddTest
            AddPythonConfigTest
            AddPseudoDependencies
            AddPseudoTarget
            AddLibrarySources
            AddLibraryHeaders
            AddSourceGroup
            LinkDirectories
            IncludeDirectories
            ForceOutOfTreeBuild
            SetOutputPaths)

