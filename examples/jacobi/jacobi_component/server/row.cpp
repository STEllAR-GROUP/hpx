
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include "row.hpp"

namespace jacobi
{
    namespace server
    {
        void row::init(std::size_t nx, double init)
        {
            values.reset(new value_holder(nx, init));
        }
    }
}

typedef hpx::components::component<
    jacobi::server::row
> row_type;

HPX_REGISTER_COMPONENT(row_type, row);

HPX_REGISTER_ACTION(
    jacobi::server::row::init_action
  , jacobi_server_row_init_action
)

HPX_REGISTER_ACTION(
    jacobi::server::row::get_action
  , jacobi_server_row_get_action
)
