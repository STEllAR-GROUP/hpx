//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_MINI_GHOST_SEND_BUFFER_HPP
#define HPX_EXAMPLES_MINI_GHOST_SEND_BUFFER_HPP

#include <examples/mini_ghost/grid.hpp>
#include <examples/mini_ghost/pack_buffer.hpp>

#include <hpx/lcos/async.hpp>
#include <hpx/util/serialize_buffer.hpp>

namespace mini_ghost {
    template <typename BufferType, std::size_t Zone, typename Action>
    struct send_buffer
    {
        typedef hpx::lcos::local::spinlock mutex_type;

        typedef typename BufferType::value_type value_type;

        typedef
            BufferType
            buffer_type;

        send_buffer()
          : dest_(hpx::invalid_id)
          , send_future_(hpx::make_ready_future())
        {}

        send_buffer(hpx::id_type dest)
          : dest_(dest)
          , send_future_(hpx::make_ready_future())
        {}

        void operator()(grid<value_type> const & g, std::size_t step, std::size_t var)
        {
            if(dest_)
            {
                send_future_.wait();
                buffer_type buffer;
                pack_buffer<Zone>::call(g, buffer);
                send_future_ =
                    hpx::async(Action(), dest_, buffer, step, var);
            }
        }

        hpx::id_type dest_;
        hpx::future<void> send_future_;
    };
}

#endif
