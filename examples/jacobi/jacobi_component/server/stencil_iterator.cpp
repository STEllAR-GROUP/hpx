
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/local/packaged_continuation.hpp>
#include <hpx/util/detail/remove_reference.hpp>

#include "stencil_iterator.hpp"

#include <boost/make_shared.hpp>

namespace jacobi
{
    namespace server
    {
        void stencil_iterator::step()
        {
            BOOST_ASSERT(top.id);
            BOOST_ASSERT(bottom.id);
            BOOST_ASSERT(top.id != bottom.id);
            BOOST_ASSERT(this->get_gid() != top.id);
            BOOST_ASSERT(this->get_gid() != bottom.id);
            //std::cout << "beginning to run ...\n";

            std::vector<hpx::lcos::future<void> > fs;
            for(std::size_t x = 1; x < nx-1; x += line_block)
            {
                std::size_t x_end = (std::min)(nx-1, x + line_block);
                fs.push_back(
                    hpx::async(
                        HPX_STD_BIND(
                            &stencil_iterator::update
                          , this
                          , rows[dst].get(x, x_end)
                          , rows[src].get(x-1, x_end+1)
                          , top.get_range(x, x_end)
                          , bottom.get_range(x, x_end)
                        )
                    )
                );
            }
            hpx::lcos::wait(fs);
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
            row_range s = src.get();
            row_range t = top.get();
            row_range b = bottom.get();

            std::vector<double>::iterator dst_ptr = d.begin();
            std::vector<double>::iterator src_ptr = s.begin();
            std::vector<double>::iterator top_ptr = t.begin();
            std::vector<double>::iterator bottom_ptr = b.begin();

            BOOST_ASSERT(
                d.end() - d.begin() + 2 == s.end() - s.begin()
            );
            BOOST_ASSERT(
                d.end() - d.begin() == t.end() - t.begin()
            );
            BOOST_ASSERT(
                d.end() - d.begin() == b.end() - b.begin()
            );

            ++src_ptr;
            while(dst_ptr < d.end())
            {
                BOOST_ASSERT(dst_ptr < d.end());
                BOOST_ASSERT(src_ptr < s.end());
                BOOST_ASSERT(top_ptr < t.end());
                BOOST_ASSERT(bottom_ptr < b.end());
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

        row_range stencil_iterator::get_range(std::size_t begin, std::size_t end)
        {
            return rows[src].get(begin, end).get();
        }
    }
}

typedef hpx::components::managed_component<
    jacobi::server::stencil_iterator
> stencil_iterator_type;

HPX_REGISTER_MINIMAL_GENERIC_COMPONENT_FACTORY(stencil_iterator_type, stencil_iterator);


HPX_REGISTER_ACTION_EX(
    jacobi::server::stencil_iterator::init_action
  , jacobi_server_stencil_iterator_init_action
)

HPX_REGISTER_ACTION_EX(
    jacobi::server::stencil_iterator::setup_boundary_action
  , jacobi_server_stencil_iterator_setup_boundary_action
)

HPX_REGISTER_ACTION_EX(
    jacobi::server::stencil_iterator::step_action
  , jacobi_server_stencil_iterator_step_action
)

HPX_REGISTER_ACTION_EX(
    jacobi::server::stencil_iterator::get_range_action
  , jacobi_server_stencil_iterator_get_range_action
)
