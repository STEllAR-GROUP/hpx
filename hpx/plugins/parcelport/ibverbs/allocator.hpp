//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_IBVERBS_ALLOCATOR_HPP
#define HPX_PARCELSET_POLICIES_IBVERBS_ALLOCATOR_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_IBVERBS)

#include <hpx/util/memory_chunk_pool.hpp>

namespace hpx { namespace parcelset { namespace policies { namespace ibverbs
{
    template <std::size_t SmallSize>
    struct allocator
    {
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;
        typedef char* pointer;
        typedef const char* const_pointer;
        typedef char& reference;
        typedef const char& const_reference;
        typedef char value_type;

        template <typename U>
        struct rebind
        {
            typedef allocator other;
        };

        allocator() throw() : memory_pool_(0) { HPX_ASSERT(false); };
        allocator(util::memory_chunk_pool & mp) throw() : memory_pool_(&mp) {};
        allocator(allocator const & other) throw() : memory_pool_(other.memory_pool_) {};

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
            if(n <= SmallSize) return new char[n];
            return memory_pool_->allocate(n);
        }

        void deallocate(pointer p, size_type n)
        {
            if(n <= SmallSize) delete[] p;
            else memory_pool_->deallocate(p, n);
        }

        size_type max_size() const throw()
        {
            return (std::numeric_limits<std::size_t>::max)() / sizeof(char);
        }

        void construct(pointer p)
        {
            *p = 0;
        }

        void construct(pointer p, const char& val)
        {
            *p = val;
        }

        /** Destroy the object referenced by @c p. */
        void destroy(pointer p)
        {
            *p = 0;
        }

        util::memory_chunk_pool * memory_pool_;
    };
}}}}

#endif

#endif
