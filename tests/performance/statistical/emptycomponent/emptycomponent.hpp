//  Copyright (c)      2012 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/emptycomponent.hpp"

namespace hpx { namespace components 
{

    class emptycomponent :
        public client_base<emptycomponent, stubs::emptycomponent>
    {
    public:
        emptycomponent(){}
    };

}}


