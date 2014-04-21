//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_MINI_GHOST_RECV_BUFFER_HPP
#define HPX_EXAMPLES_MINI_GHOST_RECV_BUFFER_HPP

#include <examples/mini_ghost/grid.hpp>
#include <examples/mini_ghost/unpack_buffer.hpp>

#include <hpx/lcos/async.hpp>
#include <hpx/util/serialize_buffer.hpp>

namespace mini_ghost {
    template <typename BufferType, std::size_t Zone>
    struct recv_buffer
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(recv_buffer);
    public:
        typedef hpx::lcos::local::spinlock mutex_type;

        typedef typename BufferType::value_type value_type;

        typedef
            BufferType
            buffer_type;

        typedef
            hpx::lcos::local::promise<buffer_type>
            buffer_promise_type;

        typedef
            std::map<std::size_t, boost::shared_ptr<buffer_promise_type> >
            buffer_map_type;

        typedef
            typename buffer_map_type::iterator
            iterator;

        recv_buffer()
          : valid_(false)
        {}

        recv_buffer(recv_buffer &&other)
          : mtx_(std::move(other.mtx_))
          , buffer_map_(std::move(other.buffer_map_))
          , valid_(other.valid_)
        {
        }

        recv_buffer& operator=(recv_buffer &&other)
        {
            if(this != &other)
            {
                mtx_        = std::move(other.mtx_);
                buffer_map_ = std::move(other.buffer_map_);
                valid_      = other.valid_;
            }
            return *this;
        }

        ~recv_buffer()
        {
            for(auto & v : buffer_map_)
            {
                if(v.second->valid())
                {
                    v.second->get_future();
                }
            }
        }

        void operator()(grid<value_type> & g, std::size_t step)
        {
            hpx::util::high_resolution_timer timer;
            buffer_type buffer = get_buffer(step).get();
            double elapsed = timer.elapsed();
            profiling::data().time_wait(elapsed);
            switch(Zone)
            {
                case EAST:
                    profiling::data().time_wait_x(elapsed);
                    break;
                case WEST:
                    profiling::data().time_wait_x(elapsed);
                    break;
                case NORTH:
                    profiling::data().time_wait_y(elapsed);
                    break;
                case SOUTH:
                    profiling::data().time_wait_y(elapsed);
                    break;
                case FRONT:
                    profiling::data().time_wait_z(elapsed);
                    break;
                case BACK:
                    profiling::data().time_wait_z(elapsed);
                    break;
            }
            unpack_buffer<Zone>::call(g, buffer);
        }

        void set_buffer(buffer_type buffer, std::size_t step)
        {
            mutex_type::scoped_lock l(mtx_);
            get_buffer_entry(step)->second->set_value(buffer);
        }

        hpx::future<buffer_type> get_buffer(std::size_t step)
        {
            hpx::future<buffer_type> res;
            {
                mutex_type::scoped_lock l(mtx_);
                iterator it = get_buffer_entry(step);
                res = it->second->get_future();
                HPX_ASSERT(buffer_map_.find(step) != buffer_map_.end());
                buffer_map_.erase(it);
            }
            return res;
        }

        iterator get_buffer_entry(std::size_t step)
        {
            iterator it = buffer_map_.find(step);
            if(it == buffer_map_.end())
            {
                auto res
                    = buffer_map_.insert(
                        std::make_pair(
                            step
                          , boost::make_shared<buffer_promise_type>()
                        )
                    );
                HPX_ASSERT(res.second);
                return res.first;
            }
            return it;
        }

        mutable mutex_type mtx_;

        buffer_map_type buffer_map_;
        bool valid_;
    };
}

#endif
