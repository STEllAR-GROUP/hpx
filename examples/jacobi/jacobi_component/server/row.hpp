
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef JACOBI_SERVER_ROW_HPP
#define JACOBI_SERVER_ROW_HPP

#include "../row_range.hpp"

#include <hpx/include/components.hpp>

#include <boost/smart_ptr/shared_array.hpp>

namespace jacobi {
    namespace server
    {

        struct HPX_COMPONENT_EXPORT row
            : hpx::components::managed_component_base<
                row
              , hpx::components::detail::this_type
              , hpx::traits::construct_with_back_ptr
            >
        {
            typedef
                hpx::components::managed_component_base<
                    row
                  , hpx::components::detail::this_type
                  , hpx::traits::construct_with_back_ptr
                >
                base_type;

            typedef hpx::components::managed_component<row> component_type;

            row(component_type * back_ptr)
                : base_type(back_ptr)
            {}

            void init(std::size_t nx, double value);

            typedef boost::intrusive_ptr<value_holder> values_type;

            values_type values;

            row_range get(std::size_t begin, std::size_t end)
            {
                //std::cout << this->get_id() << "row::get ...\n";
                HPX_ASSERT(values);
                return row_range(values, begin, end);
            }

            HPX_DEFINE_COMPONENT_ACTION(row, get, get_action);
            HPX_DEFINE_COMPONENT_ACTION(row, init, init_action);
        };
    }
}

HPX_REGISTER_ACTION_DECLARATION(
    jacobi::server::row::get_action
  , jacobi_server_row_get_action
)

HPX_REGISTER_ACTION_DECLARATION(
    jacobi::server::row::init_action
  , jacobi_server_row_init_action
)

#endif
