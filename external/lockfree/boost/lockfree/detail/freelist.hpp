//  lock-free freelist
//
//  Copyright (C) 2008, 2009 Tim Blechmann
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//  Disclaimer: Not a Boost library.

#ifndef BOOST_LOCKFREE_FREELIST_HPP_INCLUDED
#define BOOST_LOCKFREE_FREELIST_HPP_INCLUDED

#include <boost/lockfree/detail/tagged_ptr.hpp>

#include <boost/atomic.hpp>
#include <boost/noncopyable.hpp>

#include <boost/mpl/map.hpp>
#include <boost/mpl/apply.hpp>
#include <boost/mpl/at.hpp>
#include <boost/type_traits/is_pod.hpp>

#include <cstring>
#include <algorithm>            /* for std::min */

namespace boost
{
namespace lockfree
{

template <typename T, typename Alloc = std::allocator<T> >
class caching_freelist:
    private boost::noncopyable,
    private Alloc
{
    struct freelist_node
    {
        lockfree::tagged_ptr<freelist_node> next;
    };

    typedef lockfree::tagged_ptr<freelist_node> tagged_ptr;

public:
    caching_freelist(void):
        pool_(tagged_ptr(NULL, 0))
    {}

    explicit caching_freelist(std::size_t initial_nodes):
        pool_(tagged_ptr(NULL, 0))
    {
        for (std::size_t i = 0; i != initial_nodes; ++i)
        {
            T * node = Alloc::allocate(1);   // initialize once
            std::memset(node, '\0', sizeof(T));
            deallocate(node);
        }
    }

    ~caching_freelist(void)
    {
        free_memory_pool();
    }

    T * allocate (void)
    {
        for(;;)
        {
            tagged_ptr old_pool = pool_.load(memory_order_consume);

            if (!old_pool.get_ptr()) {
                T* node = Alloc::allocate(1);   // initialize once
                std::memset(node, '\0', sizeof(T));
                return node;
            }

            freelist_node * new_pool_ptr = old_pool->next.get_ptr();
            tagged_ptr new_pool (new_pool_ptr, old_pool.get_tag() + 1);

            if (pool_.compare_exchange_strong(old_pool, new_pool)) {
                void * ptr = old_pool.get_ptr();
                return reinterpret_cast<T*>(ptr);
            }
        }
    }

    T* get(void)
    {
        for(;;)
        {
            tagged_ptr old_pool = pool_.load(memory_order_consume);

            if (!old_pool.get_ptr()) 
                return NULL;

            freelist_node * new_pool_ptr = old_pool->next.get_ptr();
            tagged_ptr new_pool (new_pool_ptr, old_pool.get_tag() + 1);

            if (pool_.compare_exchange_strong(old_pool, new_pool)) {
                void * ptr = old_pool.get_ptr();
                return reinterpret_cast<T*>(ptr);
            }
        }
    }

    void deallocate (T * n)
    {
        void * node = n;
        for(;;)
        {
            tagged_ptr old_pool = pool_.load(memory_order_consume);

            freelist_node * new_pool_ptr = reinterpret_cast<freelist_node*>(node);
            tagged_ptr new_pool (new_pool_ptr, old_pool.get_tag() + 1);

            new_pool->next.set_ptr(old_pool.get_ptr());

            if (pool_.compare_exchange_strong(old_pool, new_pool))
                return;
        }
    }

private:
    void free_memory_pool(void)
    {
        tagged_ptr current (pool_);

        while (current)
        {
            void * n = current.get_ptr();
            current = current->next;
            Alloc::deallocate(reinterpret_cast<T*>(n), 1);
        }
    }

    atomic<tagged_ptr> pool_;
};

template <typename T, typename Alloc = std::allocator<T> >
class static_freelist:
    private Alloc
{
    struct freelist_node
    {
        lockfree::tagged_ptr<freelist_node> next;
    };

    typedef lockfree::tagged_ptr<freelist_node> tagged_ptr;

public:
    explicit static_freelist(std::size_t max_nodes):
        pool_(tagged_ptr(NULL, 0)), total_nodes(max_nodes)
    {
        chunks = Alloc::allocate(max_nodes);
        std::memset(chunks, '\0', max_nodes*sizeof(T));

        for (std::size_t i = 0; i != max_nodes; ++i)
        {
            T* node = chunks + i;   // initialize once
            deallocate(node);
        }
    }

    ~static_freelist(void)
    {
        Alloc::deallocate(chunks, total_nodes);
    }

    T * allocate (void)
    {
        for(;;)
        {
            tagged_ptr old_pool = pool_.load(memory_order_consume);

            if (!old_pool.get_ptr())
                return NULL; /* allocation fails */

            freelist_node * new_pool_ptr = old_pool->next.get_ptr();
            tagged_ptr new_pool (new_pool_ptr, old_pool.get_tag() + 1);

            if (pool_.compare_exchange_strong(old_pool, new_pool)) {
                void * ptr = old_pool.get_ptr();
                return reinterpret_cast<T*>(ptr);
            }
        }
    }

    void deallocate (T * n)
    {
        void * node = n;
        for(;;)
        {
            tagged_ptr old_pool = pool_.load(memory_order_consume);

            freelist_node * new_pool_ptr = reinterpret_cast<freelist_node*>(node);
            tagged_ptr new_pool (new_pool_ptr, old_pool.get_tag());

            new_pool->next.set_ptr(old_pool.get_ptr());

            if (pool_.compare_exchange_strong(old_pool, new_pool))
                return;
        }
    }

private:
    atomic<tagged_ptr> pool_;

    const std::size_t total_nodes;
    T* chunks;
};


struct caching_freelist_t {};
struct static_freelist_t {};

namespace detail
{

#if 0
template <typename T, typename Alloc, typename tag>
struct select_freelist
{
private:
    typedef typename Alloc::template rebind<T>::other Allocator;

    typedef typename boost::lockfree::caching_freelist<T, Allocator> cfl;
    typedef typename boost::lockfree::static_freelist<T, Allocator> sfl;

    typedef typename boost::mpl::map<
        boost::mpl::pair < caching_freelist_t, cfl/* typename boost::lockfree::caching_freelist<T, Alloc> */ >,
        boost::mpl::pair < static_freelist_t,  sfl/* typename boost::lockfree::static_freelist<T, Alloc> */ >,
        int
        > freelists;
public:
    typedef typename boost::mpl::at<freelists, tag>::type type;
};
#endif

} /* namespace detail */
} /* namespace lockfree */
} /* namespace boost */

#endif /* BOOST_LOCKFREE_FREELIST_HPP_INCLUDED */
