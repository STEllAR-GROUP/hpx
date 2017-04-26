//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_BUFFER_POOL_HPP)
#define HPX_UTIL_BUFFER_POOL_HPP

#include <list>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace hpx { namespace util {

    // This class holds shared_ptr of vector<T, Allocator> with a power of two
    template <typename T, typename Allocator = std::allocator<T> >
    struct buffer_pool
    {
        typedef std::vector<T, Allocator> buffer_type;
        typedef std::shared_ptr<buffer_type> shared_buffer_type;
        typedef typename buffer_type::size_type size_type;
        typedef std::map<size_type, std::list<shared_buffer_type> > buffer_map_type;

        shared_buffer_type get_buffer(size_type size)
        {
            size_type capacity = next_power_of_two(size);
            typename buffer_map_type::iterator it = buffers_.find(capacity);
            shared_buffer_type res;
            if(it == buffers_.end() || it->second.empty())
            {
                res.reset(new buffer_type());
                res->reserve(capacity);
            }
            else
            {
                res = it->second.front();
                it->second.pop_front();
            }
            return res;
        }

        void reclaim_buffer(shared_buffer_type buffer)
        {
            size_type capacity = next_power_of_two(buffer->capacity());
            buffer->clear();
            if(capacity != buffer->capacity())
            {
                buffer->reserve(capacity);
            }

            typename buffer_map_type::iterator it = buffers_.find(capacity);
            if(it == buffers_.end())
            {
                it = buffers_.insert(it, std::make_pair(capacity,
                    std::list<shared_buffer_type>()));
            }
            it->second.push_back(buffer);
        }

        void clear()
        {
            buffers_.clear();
        }

    private:
        buffer_map_type buffers_;

        static size_type next_power_of_two(size_type size)
        {
            // Check if we already have a power of two
            // http://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
            if(size && !(size & (size - 1)))
                return size;
            size--;
            size |= size >> 1;
            size |= size >> 2;
            size |= size >> 4;
            size |= size >> 8;
            size |= size >> 16;
            if(sizeof(size_type) == 8)
            {
                size |= size >> 32; //-V112
            }
            size++;
            return size;
        }
    };
}}

#endif
