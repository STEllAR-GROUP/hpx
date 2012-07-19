
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef JACOBI_SERVER_STENCIL_ITERATOR_HPP
#define JACOBI_SERVER_STENCIL_ITERATOR_HPP

#include "../row.hpp"
#include "../stencil_iterator.hpp"

#include <hpx/include/components.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/lcos.hpp>

#include <vector>

namespace jacobi
{
    struct stencil_iterator;

    namespace server
    {
        struct HPX_COMPONENT_EXPORT stencil_iterator
            : hpx::components::managed_component_base<
                stencil_iterator
              , hpx::components::detail::this_type
              , hpx::traits::construct_with_back_ptr
            >
        {
            typedef 
                hpx::components::managed_component_base<
                    stencil_iterator
                  , hpx::components::detail::this_type
                  , hpx::traits::construct_with_back_ptr
                >
                base_type;
            typedef hpx::components::managed_component<stencil_iterator> component_type;

            stencil_iterator(component_type * back_ptr)
                : base_type(back_ptr)
            {
            }

            void init(
                jacobi::row const & r
              , std::size_t y_
              , std::size_t nx_
              , std::size_t ny_
              , std::size_t l
            )
            {
                y = y_;
                rows[0] = r;
                jacobi::row tmp;
                hpx::components::component_type
                    type = hpx::components::get_component_type<
                        server::row
                    >();
                tmp.id = hpx::async<hpx::components::server::runtime_support::create_component_action>(
                    hpx::naming::get_locality_from_id(r.id)
                  , type
                  , 1
                  ).get();
                tmp.init(nx_).get();
                rows[1] = tmp;
                nx = nx_;
                ny = ny_;
                line_block = l;
                src = 0;
                dst = 1;
            }


            void setup_boundary(
                jacobi::stencil_iterator const & t
              , jacobi::stencil_iterator const & b
            )
            {
                top = t;
                bottom = b;
            }

            void step();
            
            row_range get_range(std::size_t begin, std::size_t end);

            void update(
                hpx::lcos::future<row_range> dst
              , hpx::lcos::future<row_range> src
              , hpx::lcos::future<row_range> top
              , hpx::lcos::future<row_range> bottom
            );

            HPX_DEFINE_COMPONENT_ACTION(stencil_iterator, init, init_action);
            HPX_DEFINE_COMPONENT_ACTION(stencil_iterator, setup_boundary, setup_boundary_action);
            HPX_DEFINE_COMPONENT_ACTION(stencil_iterator, step, step_action);
            HPX_DEFINE_COMPONENT_ACTION(stencil_iterator, get_range, get_range_action);

            std::size_t y;
            std::size_t ny;
            std::size_t nx;
            std::size_t line_block;
            std::size_t src;
            std::size_t dst;
            jacobi::stencil_iterator top;
            jacobi::row rows[2];
            jacobi::stencil_iterator bottom;


        };
    }
}

HPX_REGISTER_ACTION_DECLARATION_EX(
    jacobi::server::stencil_iterator::init_action
  , jacobi_server_stencil_iterator_init_action
)

HPX_REGISTER_ACTION_DECLARATION_EX(
    jacobi::server::stencil_iterator::setup_boundary_action
  , jacobi_server_stencil_iterator_setup_boundary_action
)

HPX_REGISTER_ACTION_DECLARATION_EX(
    jacobi::server::stencil_iterator::step_action
  , jacobi_server_stencil_iterator_step_action
)

HPX_REGISTER_ACTION_DECLARATION_EX(
    jacobi::server::stencil_iterator::get_range_action
  , jacobi_server_stencil_iterator_get_range_action
)

#endif
