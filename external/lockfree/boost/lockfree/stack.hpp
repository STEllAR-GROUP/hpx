//  Copyright (C) 2008 Tim Blechmann
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//  Disclaimer: Not a Boost library.

#ifndef BOOST_LOCKFREE_STACK_HPP_INCLUDED
#define BOOST_LOCKFREE_STACK_HPP_INCLUDED

#include <boost/atomic.hpp>
#include <boost/checked_delete.hpp>

#include <boost/static_assert.hpp>
#include <boost/type_traits/is_base_of.hpp>

#include <boost/lockfree/detail/tagged_ptr.hpp>
#include <boost/lockfree/detail/freelist.hpp>
#include <boost/noncopyable.hpp>

namespace boost
{
namespace lockfree
{
template <typename T,
          typename freelist_t = caching_freelist_t,
          typename Alloc = std::allocator<T>
          >
class stack:
    boost::noncopyable
{
    struct node
    {
        typedef tagged_ptr<node> tagged_ptr_t;

        node(T const & v):
            v(v)
        {}

        tagged_ptr_t next;
        T v;
    };

    typedef tagged_ptr<node> tagged_ptr_t;

    typedef typename Alloc::template rebind<node>::other node_allocator;
/*     typedef typename detail::select_freelist<node, node_allocator, freelist_t>::type pool_t; */

    typedef typename boost::mpl::if_<boost::is_same<freelist_t, caching_freelist_t>,
                                     caching_freelist<node, node_allocator>,
                                     static_freelist<node, node_allocator>
                                     >::type pool_t;

public:
    const bool is_lock_free (void) const
    {
        return tos.is_lock_free();
    }

    stack(void):
        tos(tagged_ptr_t(NULL, 0)), pool(128)
    {}

    explicit stack(std::size_t n):
        tos(tagged_ptr_t(NULL, 0)), pool(n)
    {}

    ~stack(void)
    {
        if (!empty())
        {
            T dummy;
            for(;;)
            {
                if (!pop(&dummy))
                    break;
            }
        }
    }

    bool push(T const & v)
    {
        node * newnode = alloc_node(v);

        if (newnode == 0)
            return false;

        for (;;)
        {
            tagged_ptr_t old_tos = tos.load(memory_order_relaxed);
            tagged_ptr_t new_tos (newnode, old_tos.get_tag());
            newnode->next.set_ptr(old_tos.get_ptr());

            if (tos.compare_exchange_strong(old_tos, new_tos))
                return true;
        }
    }

    bool pop(T * ret)
    {
        for (;;)
        {
            tagged_ptr_t old_tos = tos.load(memory_order_consume);

            if (!old_tos.get_ptr())
                return false;

            node * new_tos_ptr = old_tos->next.get_ptr();
            tagged_ptr_t new_tos(new_tos_ptr, old_tos.get_tag() + 1);

            if (tos.compare_exchange_strong(old_tos, new_tos))
            {
                *ret = old_tos->v;
                dealloc_node(old_tos.get_ptr());
                return true;
            }
        }
    }

    bool empty(void) const
    {
        return tos.load().get_ptr() == NULL;
    }

private:
    node * alloc_node(T const & t)
    {
        node * chunk = pool.allocate();
        new(chunk) node(t);
        return chunk;
    }

    void dealloc_node(node * n)
    {
        n->~node();
        pool.deallocate(n);
    }

    atomic<tagged_ptr_t> tos;

    static const int padding_size = BOOST_LOCKFREE_CACHELINE_BYTES - sizeof(tagged_ptr_t);
    char padding[padding_size];

    pool_t pool;
};


} /* namespace lockfree */
} /* namespace boost */

#endif /* BOOST_LOCKFREE_STACK_HPP_INCLUDED */
