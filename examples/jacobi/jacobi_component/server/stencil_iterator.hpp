
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
            : hpx::components::component_base<stencil_iterator>
        {
            typedef
                hpx::components::component_base<stencil_iterator>
                base_type;
            typedef hpx::components::component<stencil_iterator> component_type;

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
                typedef hpx::components::server::
                    create_component_action<server::row> create_action;
                tmp.id = hpx::async<create_action>(
                    hpx::naming::get_locality_from_id(r.id)
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
                top_future[src]    = t.get(src);
                top_future[dst]    = t.get(dst);
                bottom_future[src] = b.get(src);
                bottom_future[dst] = b.get(dst);
            }

            void step();

            jacobi::row get(std::size_t idx);

            void update(
                hpx::lcos::future<row_range> dst
              , hpx::lcos::future<row_range> src
              , hpx::lcos::future<row_range> top
              , hpx::lcos::future<row_range> bottom
            );

            HPX_DEFINE_COMPONENT_ACTION(stencil_iterator, init, init_action);
            HPX_DEFINE_COMPONENT_ACTION(stencil_iterator,
                setup_boundary, setup_boundary_action);
            HPX_DEFINE_COMPONENT_ACTION(stencil_iterator, step, step_action);
            HPX_DEFINE_COMPONENT_ACTION(stencil_iterator, get, get_action);

            std::size_t y;
            std::size_t ny;
            std::size_t nx;
            std::size_t line_block;
            std::size_t src;
            std::size_t dst;
            hpx::lcos::shared_future<jacobi::row> top_future[2];
            jacobi::row rows[2];
            hpx::lcos::shared_future<jacobi::row> bottom_future[2];

        };
    }
}

HPX_REGISTER_ACTION_DECLARATION(
    jacobi::server::stencil_iterator::init_action
  , jacobi_server_stencil_iterator_init_action
)

HPX_REGISTER_ACTION_DECLARATION(
    jacobi::server::stencil_iterator::setup_boundary_action
  , jacobi_server_stencil_iterator_setup_boundary_action
)

HPX_REGISTER_ACTION_DECLARATION(
    jacobi::server::stencil_iterator::step_action
  , jacobi_server_stencil_iterator_step_action
)

HPX_REGISTER_ACTION_DECLARATION(
    jacobi::server::stencil_iterator::get_action
  , jacobi_server_stencil_iterator_get_action
)

#endif
