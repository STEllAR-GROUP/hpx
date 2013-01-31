#  Copyright (c) 2011-2012 Vinay C Amatya

#  Distributed under the Boost Software License, Version 1.0. (Seec accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(dependencies
    distributing_factory_component dataflow_component)

add_hpx_component(fft
    DEPENDENCIES ${dependencies}
    FOLDER "Examples/Fft"
    AUTOGLOB)
