//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_MEMORY_CHUNK_POOL_ALLOCATOR_HPP
#define HPX_UTIL_MEMORY_CHUNK_POOL_ALLOCATOR_HPP

#include <hpx/config.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <cstdlib>
#include <vector>

namespace hpx { namespace util { namespace detail
{
    template <typename T,
              typename Pool,
              typename Mutex = hpx::lcos::local::spinlock>
    struct memory_chunk_pool_allocator
    {
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef const T& const_reference;
        typedef T value_type;

        template <typename U>
        struct rebind
        {
            typedef memory_chunk_pool_allocator<U, Pool, Mutex> other;
        };

        memory_chunk_pool_allocator() throw()
          : memory_pool_(0)
        {}
        memory_chunk_pool_allocator(Pool & mp) throw()
          : memory_pool_(&mp)
        {}
        memory_chunk_pool_allocator(
                memory_chunk_pool_allocator const & other) throw()
          : memory_pool_(other.memory_pool_)
        {}
        template <typename U>
        memory_chunk_pool_allocator(
                memory_chunk_pool_allocator<U, Pool, Mutex> const & other) throw()
          : memory_pool_(other.memory_pool_)
        {}

        pointer address(reference x) const
        {
            return &x;
        }

        const_pointer address(const_reference x) const
        {
            return &x;
        }

        pointer allocate(size_type n, void* /*hint*/ = nullptr)
        {
            HPX_ASSERT(memory_pool_);
            return reinterpret_cast<T*>(memory_pool_->allocate(sizeof(T) * n));
        }

        void deallocate(pointer p, size_type n)
        {
            HPX_ASSERT(memory_pool_);
            memory_pool_->deallocate(reinterpret_cast<char*>(p), sizeof(T) * n);
        }

        size_type max_size() const throw()
        {
            return (std::numeric_limits<std::size_t>::max)() / sizeof(T);
        }

        void construct(pointer p)
        {
            new (p) T();
        }

        template <typename U>
        void construct(pointer p, U && val)
        {
            new (p) typename util::decay<T>::type(std::forward<U>(val));
        }

        /** Destroy the object referenced by @c p. */
        void destroy(pointer p)
        {
            p->~T();
        }

        bool operator==(memory_chunk_pool_allocator const & other)
        {
            return memory_pool_ == other.memory_pool_;
        }

        bool operator!=(memory_chunk_pool_allocator const & other)
        {
            return memory_pool_ != other.memory_pool_;
        }

        Pool * memory_pool_;
    };
}}}

#endif
