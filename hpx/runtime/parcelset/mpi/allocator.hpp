//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//  Copyright (c)      2006 Douglas Gregor
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file is an adaption of boost/mpi/allocator.hpp.
// We don't want to depend on Boost.Mpi because it might not be available on all
// supported platforms.

#ifndef HPX_PARCELSET_MPI_ALLOCATOR_HPP
#define HPX_PARCELSET_MPI_ALLOCATOR_HPP

#include <hpx/config.hpp>
#include <cstddef>
#include <memory>
#include <boost/limits.hpp>

namespace hpx { namespace parcelset { namespace mpi {
    template <typename T> class allocator;

    template <>
    class allocator<void>
    {
        public:
            typedef void* pointer;
            typedef const void* const_pointer;
            typedef void value_type;

            template <typename U>
            struct rebind
            {
                typedef allocator<U> other;
            };
    };

    template <typename T>
    class allocator
    {
        public:
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
                typedef allocator<U> other;
            };

            allocator() throw() {}

            allocator(allocator const &) throw() {}

            template <typename U>
            allocator(allocator<U> const &) throw() {}

            pointer address(reference x) const
            {
                return &x;
            }

            const_pointer address(const_reference x) const
            {
                return &x;
            }

            pointer allocate(size_type n, allocator<void>::const_pointer /*hint*/ = 0)
            {
                pointer result;
                MPI_Alloc_mem(static_cast<MPI_Aint>(n * sizeof(T)), MPI_INFO_NULL, &result);
                return result;
            }

            void deallocate(pointer p, size_type /*n*/)
            {
                MPI_Free_mem(p);
            }

            size_type max_size() const throw()
            {
                return (std::numeric_limits<std::size_t>::max)() / sizeof(T);
            }
            
            void construct(pointer p, const T& val)
            {
                new ((void *)p) T(val);
            }
            
            /** Destroy the object referenced by @c p. */
            void destroy(pointer p)
            {
                ((T*)p)->~T();
            }
    };

    template<typename T1, typename T2>
    inline bool operator==(const allocator<T1>&, const allocator<T2>&) throw()
    {
        return true;
    }
    
    template<typename T1, typename T2>
    inline bool operator!=(const allocator<T1>&, const allocator<T2>&) throw()
    {
        return false;
    }
}}}

#endif
