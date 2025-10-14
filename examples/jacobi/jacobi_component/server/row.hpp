//  Copyright (c) 2012 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include "../row_range.hpp"

#include <hpx/assert.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/modules/memory.hpp>

#include <cstddef>

namespace jacobi { namespace server {

    struct HPX_COMPONENT_EXPORT row : hpx::components::component_base<row>
    {
        typedef hpx::components::component_base<row> base_type;

        typedef hpx::components::component<row> component_type;

        void init(std::size_t nx, double value);

        typedef hpx::intrusive_ptr<value_holder> values_type;

        values_type values;

        row_range get(std::size_t begin, std::size_t end)
        {
            //std::cout << this->get_id() << "row::get ...\n";
            HPX_ASSERT(values);
            return row_range(values, static_cast<std::ptrdiff_t>(begin),
                static_cast<std::ptrdiff_t>(end));
        }

        HPX_DEFINE_COMPONENT_ACTION(row, get, get_action)
        HPX_DEFINE_COMPONENT_ACTION(row, init, init_action)
    };
}}    // namespace jacobi::server

HPX_REGISTER_ACTION_DECLARATION(
    jacobi::server::row::get_action, jacobi_server_row_get_action)

HPX_REGISTER_ACTION_DECLARATION(
    jacobi::server::row::init_action, jacobi_server_row_init_action)

#endif
