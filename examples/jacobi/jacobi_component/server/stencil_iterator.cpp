
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/local/packaged_continuation.hpp>

#include "stencil_iterator.hpp"

#include <boost/make_shared.hpp>

#include <vector>

namespace jacobi
{
    namespace server
    {
        void stencil_iterator::step()
        {

            jacobi::row top = top_future[src].get();
            jacobi::row bottom = bottom_future[src].get();

            HPX_ASSERT(top.id);
            HPX_ASSERT(bottom.id);
            HPX_ASSERT(top.id != bottom.id);
            HPX_ASSERT(this->get_unmanaged_id() != top.id);
            HPX_ASSERT(this->get_unmanaged_id() != bottom.id);

            std::vector<hpx::lcos::future<void> > fs;
            for(std::size_t x = 1; x < nx-1; x += line_block)
            {
                std::size_t x_end = (std::min)(nx-1, x + line_block);
                fs.push_back(
                    hpx::async(
                        &stencil_iterator::update
                      , this
                      , rows[dst].get(x, x_end)
                      , rows[src].get(x-1, x_end+1)
                      , top.get(x, x_end)
                      , bottom.get(x, x_end)
                    )
                );
            }
            hpx::wait_all(fs);
            std::swap(src, dst);
        }

        void stencil_iterator::update(
            hpx::lcos::future<row_range> dst
          , hpx::lcos::future<row_range> src
          , hpx::lcos::future<row_range> top
          , hpx::lcos::future<row_range> bottom
        )
        {
            row_range d = dst.get();

            std::vector<double>::iterator dst_ptr = d.begin();
            row_range s = src.get();
            std::vector<double>::iterator src_ptr = s.begin();
            row_range t = top.get();
            std::vector<double>::iterator top_ptr = t.begin();
            row_range b = bottom.get();
            std::vector<double>::iterator bottom_ptr = b.begin();

            HPX_ASSERT(
                d.end() - d.begin() + 2 == s.end() - s.begin()
            );
            HPX_ASSERT(
                d.end() - d.begin() == t.end() - t.begin()
            );
            HPX_ASSERT(
                d.end() - d.begin() == b.end() - b.begin()
            );

            ++src_ptr;
            while(dst_ptr < d.end())
            {
                HPX_ASSERT(dst_ptr < d.end());
                HPX_ASSERT(src_ptr < s.end());
                HPX_ASSERT(top_ptr < t.end());
                HPX_ASSERT(bottom_ptr < b.end());
                *dst_ptr
                    =(
                        *(src_ptr - 1) + *(src_ptr + 1)
                      + *top_ptr + *bottom_ptr
                    ) * 0.25;
                ++dst_ptr;
                ++src_ptr;
                ++top_ptr;
                ++bottom_ptr;
            }
        }

        jacobi::row stencil_iterator::get(std::size_t idx)
        {
            HPX_ASSERT(rows[idx].id);
            return rows[idx];
        }
    }
}

typedef hpx::components::component<
    jacobi::server::stencil_iterator
> stencil_iterator_type;

HPX_REGISTER_COMPONENT(stencil_iterator_type, stencil_iterator);


HPX_REGISTER_ACTION(
    jacobi::server::stencil_iterator::init_action
  , jacobi_server_stencil_iterator_init_action
)

HPX_REGISTER_ACTION(
    jacobi::server::stencil_iterator::setup_boundary_action
  , jacobi_server_stencil_iterator_setup_boundary_action
)

HPX_REGISTER_ACTION(
    jacobi::server::stencil_iterator::step_action
  , jacobi_server_stencil_iterator_step_action
)

HPX_REGISTER_ACTION(
    jacobi::server::stencil_iterator::get_action
  , jacobi_server_stencil_iterator_get_action
)
