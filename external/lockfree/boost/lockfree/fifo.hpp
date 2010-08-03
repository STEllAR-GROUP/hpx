//  lock-free fifo queue from
//  Michael, M. M. and Scott, M. L.,
//  "simple, fast and practical non-blocking and blocking concurrent queue algorithms"
//
//  implementation for c++
//
//  Copyright (C) 2008 Tim Blechmann
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//  Disclaimer: Not a Boost library.

#ifndef BOOST_LOCKFREE_FIFO_HPP_INCLUDED
#define BOOST_LOCKFREE_FIFO_HPP_INCLUDED

#include <boost/atomic.hpp>
#include <boost/lockfree/detail/tagged_ptr.hpp>
#include <boost/lockfree/detail/freelist.hpp>

#include <boost/static_assert.hpp>
#include <boost/type_traits/is_pod.hpp>

#include <memory>               /* std::auto_ptr */
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/noncopyable.hpp>

namespace boost
{
namespace lockfree
{

namespace detail
{

template <typename T, typename freelist_t, typename Alloc>
class fifo:
    boost::noncopyable
{
private:
#ifndef BOOST_DOXYGEN_INVOKED
    BOOST_STATIC_ASSERT(boost::is_pod<T>::value);

    struct BOOST_LOCKFREE_CACHELINE_ALIGNMENT node
    {
        typedef tagged_ptr<node> tagged_ptr_t;

        node(T const & v):
            data(v)
        {
            /* increment tag to avoid ABA problem */
            tagged_ptr_t old_next = next.load(memory_order_relaxed);
            tagged_ptr_t new_next (NULL, old_next.get_tag()+1);
            next.store(new_next, memory_order_release);
        }

        node (void):
            next(tagged_ptr_t(NULL, 0))
        {}

        atomic<tagged_ptr_t> next;
        T data;
    };

    typedef tagged_ptr<node> tagged_ptr_t;

    typedef typename Alloc::template rebind<node>::other node_allocator;
/*     typedef typename select_freelist<node, node_allocator, freelist_t>::type pool_t; */

    typedef typename boost::mpl::if_<boost::is_same<freelist_t, caching_freelist_t>,
                                     caching_freelist<node, node_allocator>,
                                     static_freelist<node, node_allocator>
                                     >::type pool_t;

    void initialize(void)
    {
        node * n = alloc_node();
        tagged_ptr_t dummy_node(n, 0);
        head_.store(dummy_node, memory_order_relaxed);
        tail_.store(dummy_node, memory_order_release);
    }
#endif

public:
    /**
     * \return true, if implementation is lock-free.
     *
     * \warning \b Warning: It only checks, if the fifo head node is lockfree. on most platforms, this should be sufficient, though
     * */
    const bool is_lock_free (void) const
    {
        return head_.is_lock_free();
    }

    //! Construct fifo, initially allocates 128 nodes
    fifo(void):
        pool(128)
    {
        initialize();
    }

    //! Construct fifo with a number of initially allocated fifo nodes.
    explicit fifo(std::size_t initial_nodes):
        pool(initial_nodes)
    {
        initialize();
    }

    /** Destroys fifo, free all nodes from freelist.
     *
     *  \warning not threadsafe
     *
     * */
    ~fifo(void)
    {
        if (!empty())
        {
            T dummy;
            for(;;)
            {
                if (!dequeue(&dummy))
                    break;
            }
        }
        dealloc_node(head_.load(memory_order_relaxed).get_ptr());
    }

    /**
     * \return true, if fifo is empty.
     *
     * \warning Not thread-safe, use for debugging purposes only
     * */
    bool empty(void)
    {
        return head_.load().get_ptr() == tail_.load().get_ptr();
    }

    /** Enqueues object t to the fifo. Enqueueing may fail, if the freelist is not able to allocate a new fifo node.
     *
     * \returns true, if the enqueue operation is successful.
     *
     * \note Thread-safe and non-blocking
     * \warning \b Warning: May block if node needs to be allocated from the operating system
     * */
    bool enqueue(T const & t)
    {
        node * n = alloc_node(t);

        if (n == NULL)
            return false;

        for (;;)
        {
            tagged_ptr_t tail = tail_.load(memory_order_acquire);
            tagged_ptr_t next = tail->next.load(memory_order_acquire);
            node * next_ptr = next.get_ptr();

            if (likely(tail == tail_.load(memory_order_acquire)))
            {
                if (next_ptr == 0)
                {
                    if ( tail->next.compare_exchange_strong(next, tagged_ptr_t(n, next.get_tag() + 1)) )
                    {
                        tail_.compare_exchange_strong(tail, tagged_ptr_t(n, tail.get_tag() + 1));
                        return true;
                    }
                }
                else
                    tail_.compare_exchange_strong(tail, tagged_ptr_t(next_ptr, tail.get_tag() + 1));
            }
        }
    }

    /** Dequeue object from fifo.
     *
     * if dequeue operation is successful, object is written to memory location denoted by ret.
     *
     * \returns true, if the dequeue operation is successful, false if fifo was empty.
     *
     * \note Thread-safe and non-blocking
     *
     * */
    bool dequeue (T * ret)
    {
        for (;;)
        {
            tagged_ptr_t head = head_.load(memory_order_acquire);
            tagged_ptr_t tail = tail_.load(memory_order_acquire);
            tagged_ptr_t next = head->next.load(memory_order_acquire);
            node * next_ptr = next.get_ptr();

            if (likely(head == head_.load(memory_order_acquire)))
            {
                if (head.get_ptr() == tail.get_ptr())
                {
                    if (next_ptr == 0)
                        return false;
                    tail_.compare_exchange_strong(tail, tagged_ptr_t(next_ptr, tail.get_tag() + 1));
                }
                else
                {
                    if (next_ptr == 0) /* this check shouldn't be needed, but it crashes without :/ */
                        continue;
                    *ret = next_ptr->data;
                    if (head_.compare_exchange_strong(head, tagged_ptr_t(next_ptr, head.get_tag() + 1)))
                    {
                        dealloc_node(head.get_ptr());
                        return true;
                    }
                }
            }
        }
    }

private:
#ifndef BOOST_DOXYGEN_INVOKED
    node * alloc_node(void)
    {
        node * chunk = pool.allocate();
        new(chunk) node();
        return chunk;
    }

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

    atomic<tagged_ptr_t> head_;
    static const int padding_size = BOOST_LOCKFREE_CACHELINE_BYTES - sizeof(tagged_ptr_t);
    char padding1[padding_size];
    atomic<tagged_ptr_t> tail_;
    char padding2[padding_size];

    pool_t pool;
#endif
};

} /* namespace detail */

/** The fifo class provides a multi-writer/multi-reader fifo, enqueueing and dequeueing is lockfree,
 *  construction/destruction has to be synchronized. It uses a freelist for memory management,
 *  freed nodes are pushed to the freelist, but not returned to the os. This may result in leaking memory.
 *
 *  The memory management of the fifo can be controlled via its freelist_t template argument. Two different
 *  freelists can be used. struct caching_freelist_t selects a caching freelist, which can allocate more nodes
 *  from the operating system, and struct static_freelist_t uses a fixed-sized freelist. With a fixed-sized
 *  freelist, the enqueue operation may fail, while with a caching freelist, the enqueue operation may block.
 *
 *  \b Limitation: The fifo class is limited to PODs
 *
 * */
template <typename T,
          typename freelist_t = caching_freelist_t,
          typename Alloc = std::allocator<T>
          >
class fifo:
    public detail::fifo<T, freelist_t, Alloc>
{
public:
    //! Construct fifo, initially allocates 128 nodes
    fifo(void)
    {}

    //! Construct fifo with a number of initially allocated fifo nodes.
    explicit fifo(std::size_t initial_nodes):
        detail::fifo<T, freelist_t, Alloc>(initial_nodes)
    {}
};


/** template specialization of the boost::lockfree::fifo class for pointer arguments.
 *
 *  it supports dequeue operations to stl/boost-style smart pointers
 * */
template <typename T, typename freelist_t, typename Alloc>
class fifo<T*, freelist_t, Alloc>:
    public detail::fifo<T*, freelist_t, Alloc>
{
#ifndef BOOST_DOXYGEN_INVOKED
    typedef detail::fifo<T*, freelist_t, Alloc> fifo_t;

    template <typename smart_ptr>
    bool dequeue_smart_ptr(smart_ptr & ptr)
    {
        T * result = 0;
        bool success = fifo_t::dequeue(&result);

        if (success)
            ptr.reset(result);
        return success;
    }
#endif

public:
    //! Construct fifo, initially allocates 128 nodes
    fifo(void)
    {}

    //! Construct fifo with a number of initially allocated fifo nodes.
    explicit fifo(std::size_t initial_nodes):
        fifo_t(initial_nodes)
    {}

    //! \copydoc detail::fifo::dequeue
    bool dequeue (T ** ret)
    {
        return fifo_t::dequeue(ret);
    }

    /** Dequeue object from fifo to std::auto_ptr
     *
     * if dequeue operation is successful, object is written to memory location denoted by ret.
     *
     * \returns true, if the dequeue operation is successful, false if fifo was empty.
     *
     * \note Thread-safe and non-blocking
     *
     * */
    bool dequeue (std::auto_ptr<T> & ret)
    {
        return dequeue_smart_ptr(ret);
    }

    /** Dequeue object from fifo to boost::scoped_ptr
     *
     * if dequeue operation is successful, object is written to memory location denoted by ret.
     *
     * \returns true, if the dequeue operation is successful, false if fifo was empty.
     *
     * \note Thread-safe and non-blocking
     *
     * */
    bool dequeue (boost::scoped_ptr<T> & ret)
    {
        BOOST_STATIC_ASSERT(sizeof(boost::scoped_ptr<T>) == sizeof(T*));
        return dequeue(reinterpret_cast<T**>((void*)&ret));
    }

    /** Dequeue object from fifo to boost::shared_ptr
     *
     * if dequeue operation is successful, object is written to memory location denoted by ret.
     *
     * \returns true, if the dequeue operation is successful, false if fifo was empty.
     *
     * \note Thread-safe and non-blocking
     *
     * */
    bool dequeue (boost::shared_ptr<T> & ret)
    {
        return dequeue_smart_ptr(ret);
    }
};

} /* namespace lockfree */
} /* namespace boost */


#endif /* BOOST_LOCKFREE_FIFO_HPP_INCLUDED */
