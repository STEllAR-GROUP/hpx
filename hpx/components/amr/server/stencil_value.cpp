//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>

#include <hpx/components/amr/stencil.hpp>

///////////////////////////////////////////////////////////////////////////////
typedef 
    hpx::components::server::detail::stencil_value<
        hpx::components::amr::detail::stencil, double, 3
    >
stencil_value_type;

HPX_DEFINE_GET_COMPONENT_TYPE(stencil_value_type);

