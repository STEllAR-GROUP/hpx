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

#ifndef _MSC_VER
        recv_buffer(recv_buffer &&) = default;
        recv_buffer& operator=(recv_buffer &&) = default;
#else
        recv_buffer(recv_buffer &&other)
        {
            this->buffer_map_ = std::move(other.buffer_map_);
            this->valid_      = other.valid_;
        }
        recv_buffer& operator=(recv_buffer &&other)
        {
            this->buffer_map_ = std::move(other.buffer_map_);
            this->valid_      = other.valid_;
            return *this;
        }
#endif

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

        hpx::future<void> operator()(grid<value_type> & g, std::size_t step)
        {
            if(valid_)
            {
                return get_buffer(step).then(
                    [&g](hpx::future<buffer_type> buffer_future)
                    {
                        unpack_buffer<Zone>::call(g, buffer_future.get());
                    }
                );
            }
            else
            {
                return hpx::make_ready_future();
            }
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
