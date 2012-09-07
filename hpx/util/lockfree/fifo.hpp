//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_LOCKFREE_FIFO_AUG_31_2012_0935PM)
#define HPX_UTIL_LOCKFREE_FIFO_AUG_31_2012_0935PM

#include <boost/config.hpp>
#include <boost/version.hpp>

#if BOOST_VERSION >= 105200
#include <boost/lockfree/policies.hpp>
#include <boost/lockfree/queue.hpp>

namespace boost { namespace lockfree
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Alloc = std::allocator<T> >
    class caching_freelist : public lockfree::detail::freelist_stack<T, Alloc>
    {
        typedef lockfree::detail::freelist_stack<T, Alloc> base_type;

    public:
        caching_freelist (std::size_t n = 0) 
          : lockfree::detail::freelist_stack<T, Alloc>(Alloc(), n)
        {}

        T* allocate()
        {
            return this->base_type::allocate<true, false>();
        }

        void deallocate(T* n)
        {
            this->base_type::deallocate<true>(n);
        }
    };

    template <typename T, typename Alloc = std::allocator<T> >
    class static_freelist : public lockfree::detail::freelist_stack<T, Alloc>
    {
    public:
        static_freelist (std::size_t n = 0) 
          : lockfree::detail::freelist_stack<T, Alloc>(Alloc(), n)
        {}

        T* allocate()
        {
            return this->base_type::allocate<true, true>();
        }

        void deallocate(T* n)
        {
            this->base_type::deallocate<true>(n);
        }
    };

    struct caching_freelist_t {};
    struct static_freelist_t {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, 
        typename Freelist = caching_freelist_t, typename Alloc = std::allocator<void> >
    class fifo;

    template <typename T, typename Alloc>
    class fifo<T, caching_freelist_t, Alloc>
      : public queue<T, lockfree::fixed_sized<false>, lockfree::allocator<Alloc> >
    {
        typedef queue<
            T, lockfree::fixed_sized<false>, lockfree::allocator<Alloc> 
        > base_type;

    public:
        fifo() {}

        explicit fifo(std::size_t n)
          : base_type(n)
        {}

        bool enqueue(T const& t)
        {
            return this->base_type::push(t);
        }

        bool dequeue(T& t)
        {
            return this->base_type::pop(t);
        }
    };

    template <typename T, typename Alloc>
    class fifo<T, static_freelist_t, Alloc>
      : public queue<T, lockfree::fixed_sized<true>, lockfree::capacity<1000>, 
            lockfree::allocator<Alloc> >
    {
        typedef queue<T, lockfree::fixed_sized<true>, lockfree::capacity<1000>, 
            lockfree::allocator<Alloc> > base_type;

    public:
        fifo() {}

        explicit fifo(std::size_t n)
          : base_type(n)
        {}

        bool enqueue(T const& t)
        {
            return this->base_type::bounded_push(t);
        }

        bool dequeue(T& t)
        {
            return this->base_type::pop(t);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    using detail::likely;
    using detail::unlikely;
}}

#else
#include <boost/lockfree/fifo.hpp>
#endif

#endif
