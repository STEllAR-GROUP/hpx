//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_LOCKFREE_FIFO_AUG_31_2012_0935PM)
#define HPX_UTIL_LOCKFREE_FIFO_AUG_31_2012_0935PM

#include <boost/version.hpp>

// use released library Boost.Lockfree starting V1.53.0
#if BOOST_VERSION >= 105300
#include <boost/lockfree/policies.hpp>
#include <boost/lockfree/queue.hpp>

namespace boost { namespace lockfree
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Alloc = std::allocator<T> >
    class caching_freelist
        : public lockfree::detail::freelist_stack<T, Alloc>
    {
        typedef lockfree::detail::freelist_stack<T, Alloc> base_type;

    public:
        caching_freelist (std::size_t n = 0)
          : lockfree::detail::freelist_stack<T, Alloc>(Alloc(), n)
        {}

        T* allocate()
        {
            return this->base_type::template allocate<true, false>();
        }

        void deallocate(T* n)
        {
            this->base_type::template deallocate<true>(n);
        }
    };

    template <typename T, typename Alloc = std::allocator<T> >
    class static_freelist
        : public lockfree::detail::freelist_stack<T, Alloc>
    {
        typedef lockfree::detail::freelist_stack<T, Alloc> base_type;

    public:
        static_freelist (std::size_t n = 0)
          : lockfree::detail::freelist_stack<T, Alloc>(Alloc(), n)
        {}

        T* allocate()
        {
            return this->base_type::template allocate<true, true>();
        }

        void deallocate(T* n)
        {
            this->base_type::template deallocate<true>(n);
        }
    };

    struct caching_freelist_t {};
    struct static_freelist_t {};
}}

#else
#include <boost/lockfree/detail/freelist.hpp>
#endif

#endif
