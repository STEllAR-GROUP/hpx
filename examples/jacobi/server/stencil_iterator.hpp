
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
#include <hpx/components/dataflow/dataflow.hpp>
#include <hpx/components/dataflow/dataflow_trigger.hpp>

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
                center = r;
                nx = nx_;
                ny = ny_;
                line_block = l;
            }


            void setup_boundary(
                jacobi::stencil_iterator const & t
              , jacobi::stencil_iterator const & b
            )
            {
                top = t;
                bottom = b;
            }

            void run(std::size_t max_iterations);
            
            void next(
                std::size_t iter
              , std::size_t max_iterations
            );

            hpx::lcos::dataflow_base<void> get_dep(std::size_t iter, std::size_t begin, std::size_t end);
            
            row_range get(std::size_t iter, std::size_t begin, std::size_t end);

            void update(row_range dst, row_range src, row_range top, row_range bottom)
            {
                double * dst_ptr = dst.begin();
                double * src_ptr = src.begin();
                double * top_ptr = top.begin();
                double * bottom_ptr = bottom.begin();

                while(dst_ptr != dst.end())
                {
                    *dst_ptr
                        =(
                            *(src_ptr + 1) + *(src_ptr - 1)
                          + *(top_ptr) + *(bottom_ptr - 1)
                        ) * 0.25;
                    ++dst_ptr;
                    ++src_ptr;
                    ++top_ptr;
                    ++bottom_ptr;
                }
            }


            HPX_DEFINE_COMPONENT_ACTION(stencil_iterator, init, init_action);
            HPX_DEFINE_COMPONENT_ACTION(stencil_iterator, setup_boundary, setup_boundary_action);
            HPX_DEFINE_COMPONENT_ACTION(stencil_iterator, run, run_action);
            HPX_DEFINE_COMPONENT_ACTION(stencil_iterator, update, update_action);
            HPX_DEFINE_COMPONENT_ACTION(stencil_iterator, next, next_action);
            HPX_DEFINE_COMPONENT_ACTION(stencil_iterator, get, get_action);

            std::size_t y;
            std::size_t ny;
            std::size_t nx;
            std::size_t line_block;
            jacobi::stencil_iterator top;
            jacobi::row center;
            jacobi::stencil_iterator bottom;
            hpx::util::spinlock mtx;
            typedef
                std::map<
                    std::size_t
                  , std::map<
                        std::pair<std::size_t, std::size_t>
                      , hpx::lcos::dataflow_base<void>
                    >
                >
                iteration_deps_type;
            iteration_deps_type iteration_deps;
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
    jacobi::server::stencil_iterator::run_action
  , jacobi_server_stencil_iterator_run_action
)

HPX_REGISTER_ACTION_DECLARATION_EX(
    jacobi::server::stencil_iterator::update_action
  , jacobi_server_stencil_iterator_update_action
)

HPX_REGISTER_ACTION_DECLARATION_EX(
    jacobi::server::stencil_iterator::next_action
  , jacobi_server_stencil_iterator_next_action
)

HPX_REGISTER_ACTION_DECLARATION_EX(
    jacobi::server::stencil_iterator::get_action
  , jacobi_server_stencil_iterator_get_action
)

#endif
