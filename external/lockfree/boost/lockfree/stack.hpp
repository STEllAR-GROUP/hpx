//  Copyright (C) 2008 Tim Blechmann
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//  Disclaimer: Not a Boost library.

#ifndef BOOST_LOCKFREE_STACK_HPP_INCLUDED
#define BOOST_LOCKFREE_STACK_HPP_INCLUDED

#include <boost/checked_delete.hpp>

#include <boost/static_assert.hpp>
#include <boost/type_traits/is_base_of.hpp>

#include <boost/lockfree/tagged_ptr.hpp>
#include <boost/lockfree/freelist.hpp>
#include <boost/noncopyable.hpp>


namespace boost
{
namespace lockfree
{
template <typename T,
          typename Alloc = std::allocator<T> >
class stack:
    boost::noncopyable
{
    struct node
    {
        node(T const & v):
            v(v)
        {}

        tagged_ptr<node> next;
        T v;
    };

    typedef tagged_ptr<node> ptr_type;

public:
    stack(void):
        tos(NULL)
    {}

    explicit stack(std::size_t n):
        tos(NULL), pool(n)
    {}

    void push(T const & v)
    {
        node * newnode = alloc_node(v);

        ptr_type old_tos;
        do
        {
            old_tos.set(tos);
            newnode->next.set_ptr(old_tos.get_ptr());
        }
        while (!tos.CAS(old_tos, newnode));
    }

    bool pop(T * ret)
    {
        for (;;)
        {
            ptr_type old_tos;
            old_tos.set(tos);

            if (!old_tos)
                return false;

            node * new_tos = old_tos->next.get_ptr();

            if (tos.CAS(old_tos, new_tos))
            {
                *ret = old_tos->v;
                dealloc_node(old_tos.get_ptr());
                return true;
            }
        }
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

    ptr_type tos;

    typedef typename Alloc::template rebind<node>::other node_allocator;
    boost::lockfree::caching_freelist<node, node_allocator> pool;
};


} /* namespace lockfree */
} /* namespace boost */

#endif /* BOOST_LOCKFREE_STACK_HPP_INCLUDED */
