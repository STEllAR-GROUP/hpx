//  Copyright (c) 2014 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_RECEIVE_BUFFER_MAY_08_2014_1102AM)
#define HPX_LCOS_LOCAL_RECEIVE_BUFFER_MAY_08_2014_1102AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/conditional_trigger.hpp>
#include <hpx/lcos/local/no_mutex.hpp>
#include <hpx/util/assert.hpp>

#include <boost/dynamic_bitset.hpp>
#include <utility>
#include <boost/foreach.hpp>

#include <list>

namespace hpx { namespace lcos { namespace local
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Mutex = lcos::local::spinlock>
    struct receive_buffer
    {
    protected:
        typedef Mutex mutex_type;

        typedef T buffer_type;
        typedef hpx::lcos::local::promise<buffer_type> buffer_promise_type;

        typedef std::map<std::size_t, buffer_promise_type> buffer_map_type;
        typedef typename buffer_map_type::iterator iterator;

    private:
        HPX_MOVABLE_BUT_NOT_COPYABLE(receive_buffer)

    public:
        receive_buffer() {}

        ~receive_buffer()
        {
            iterator end = buffer_map_.end();
            for (iterator it = buffer_map_.begin(); it != end; ++it)
            {
                if ((*it).second.valid())
                    (*it).second.get_future();
            }
        }

        hpx::future<T> receive(std::size_t step)
        {
            typename mutex_type::scoped_lock l(mtx_);
            return receive_locked(step);
        }

        void store_received(std::size_t step, T && val)
        {
            typename mutex_type::scoped_lock l(mtx_);
            return store_received_locked(step, std::move(val));
        }

    protected:
        struct erase_on_exit
        {
            erase_on_exit(buffer_map_type& buffer_map, iterator it)
              : buffer_map_(buffer_map), it_(it)
            {}

            ~erase_on_exit()
            {
                buffer_map_.erase(it_);
            }

            buffer_map_type& buffer_map_;
            iterator it_;
        };

        hpx::future<T> receive_locked(std::size_t step)
        {
            iterator it = get_buffer_entry(step);
            HPX_ASSERT(it != buffer_map_.end());

            erase_on_exit t(buffer_map_, it);
            return it->second.get_future();
        }

        void store_received_locked(std::size_t step, T && val)
        {
            iterator it = get_buffer_entry(step);
            HPX_ASSERT(it != buffer_map_.end());

            it->second.set_value(std::move(val));
        }

        iterator get_buffer_entry(std::size_t step)
        {
            iterator it = buffer_map_.find(step);
            if (it == buffer_map_.end())
            {
                std::pair<iterator, bool> res =
                    buffer_map_.insert(std::make_pair(step, buffer_promise_type()));
                if (!res.second)
                {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "base_receive_buffer::get_buffer_entry",
                        "couldn't insert a new entry into the receive buffer");
                }
                return res.first;
            }
            return it;
        }

    private:
        mutable mutex_type mtx_;
        buffer_map_type buffer_map_;
    };
}}}

#endif

