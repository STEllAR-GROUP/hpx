# Copyright (c) 2007-2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This directory contains the sources for an HPX component. The following
# directives add this component to the build process.

set(component
    hello_world)          # name of the component

set(dependencies
    iostreams)

add_hpx_component(${component}
    AUTOGLOB
    HEADER_GLOB "${component}_component.h*"
    SOURCE_GLOB "${component}_component.c*"
    COMPONENT_DEPENDENCIES ${dependencies}
    FOLDER "Examples/HelloWorldExternal")
