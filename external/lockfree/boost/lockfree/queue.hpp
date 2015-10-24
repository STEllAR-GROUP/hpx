//  lock-free fifo queue from
//  Michael, M. M. and Scott, M. L.,
//  "simple, fast and practical non-blocking and blocking concurrent queue algorithms"
//
//  implementation for c++
//
//  Copyright (C) 2008, 2009, 2010, 2011 Tim Blechmann
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//  Disclaimer: Not a Boost library.

#ifndef BOOST_LOCKFREE_FIFO_HPP_INCLUDED
#define BOOST_LOCKFREE_FIFO_HPP_INCLUDED

#include <memory>               /* std::auto_ptr */

#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/has_trivial_assign.hpp>

#include <boost/lockfree/detail/atomic.hpp>
#include <boost/lockfree/detail/tagged_ptr.hpp>
#include <boost/lockfree/detail/freelist.hpp>

namespace boost {
namespace lockfree {
namespace detail {

template <typename T, typename freelist_t, typename Alloc>
class queue:
    boost::noncopyable
{
private:
#ifndef BOOST_DOXYGEN_INVOKED
    BOOST_STATIC_ASSERT(boost::is_pod<T>::value);

    struct BOOST_LOCKFREE_CACHELINE_ALIGNMENT node
    {
        typedef tagged_ptr<node> tagged_node_ptr;

        node(T const & v):
            data(v)
        {
            /* increment tag to avoid ABA problem */
            tagged_node_ptr old_next = next.load(memory_order_relaxed);
            tagged_node_ptr new_next (NULL, old_next.get_tag()+1);
            next.store(new_next, memory_order_release);
        }

        node (void):
            next(tagged_node_ptr(NULL, 0))
        {}

        atomic<tagged_node_ptr> next;
        T data;
    };

    typedef tagged_ptr<node> tagged_node_ptr;

    typedef typename Alloc::template rebind<node>::other node_allocator;

    typedef typename boost::mpl::if_<boost::is_same<freelist_t, caching_freelist_t>,
                                     detail::freelist_stack<node, true, node_allocator>,
                                     detail::freelist_stack<node, false, node_allocator>
                                     >::type pool_t;

    void initialize(void)
    {
        node * n = pool.construct();
        tagged_node_ptr dummy_node(n, 0);
        head_.store(dummy_node, memory_order_relaxed);
        tail_.store(dummy_node, memory_order_release);
    }
#endif

public:
    /**
     * \return true, if implementation is lock-free.
     *
     * \warning \b Warning: It only checks, if the queue head node is lockfree.
     *                      On most platforms, the whole implementation is
     *                      lockfree, if this is true. Using c++0x-style atomics, there
     *                      is no possibility to provide a completely
     *                      accurate implementation, because one would need to test every
     *                      internal node, which is impossible
     *                      if further nodes will be allocated from the operating system.
     * */
    bool is_lock_free (void) const
    {
        return head_.is_lock_free() && pool.is_lock_free();
    }

    //! Construct queue.
    queue(void)
    {
        pool.reserve_unsafe(1);
        initialize();
    }

    //! Construct queue, allocate n nodes for the freelist.
    explicit queue(std::size_t n)
    {
        pool.reserve_unsafe(n+1);
        initialize();
    }

    //! \copydoc boost::lockfree::stack::reserve
    void reserve(std::size_t n)
    {
        pool.reserve(n);
    }

    //! \copydoc boost::lockfree::stack::reserve_unsafe
    void reserve_unsafe(std::size_t n)
    {
        pool.reserve_unsafe(n);
    }

    /** Destroys queue, free all nodes from freelist.
     * */
    ~queue(void)
    {
        if (!empty()) {
            T dummy;
            while(pop_unsafe(dummy))
                ;
        }
        pool.destruct(head_.load(memory_order_relaxed).get_ptr());
    }

    /** Check if the ringbuffer is empty
     *
     * \warning Not thread-safe, use for debugging purposes only
     * */
    bool empty(void)
    {
        return head_.load().get_ptr() == tail_.load().get_ptr();
    }

    /** Enqueues object t to the queue. Enqueueing may fail,
     *  if the freelist is not able to allocate a new queue node.
     *
     * \returns true, if the push operation is successful.
     *
     * \note Thread-safe and non-blocking
     * \warning \b Warning:
     *  May block if node needs to be allocated from the operating system
     * */
    bool push(T const & t)
    {
        node * n = pool.construct(t);

        if (n == NULL)
            return false;

        for (;;) {
            tagged_node_ptr tail = tail_.load(memory_order_acquire);
            tagged_node_ptr next = tail->next.load(memory_order_acquire);
            node * next_ptr = next.get_ptr();

            tagged_node_ptr tail2 = tail_.load(memory_order_acquire);
            if (likely(tail == tail2)) {
                if (next_ptr == 0) {
                    if ( tail->next.compare_exchange_weak(next, tagged_node_ptr(n,
                        next.get_tag() + 1)) ) {
                        tail_.compare_exchange_strong(tail, tagged_node_ptr(n,
                            tail.get_tag() + 1));
                        return true;
                    }
                }
                else
                    tail_.compare_exchange_strong(tail, tagged_node_ptr(next_ptr,
                        tail.get_tag() + 1));
            }
        }
    }

    /** Enqueues object t to the queue. Enqueueing may fail,
     *  if the freelist is not able to allocate a new queue node.
     *
     * \returns true, if the push operation is successful.
     *
     * \note Not thread-safe
     * \warning \b Warning: May block if node needs to be
     * allocated from the operating system
     * */
    bool push_unsafe(T const & t)
    {
        node * n = pool.construct_unsafe(t);

        if (n == NULL)
            return false;

        for (;;)
        {
            tagged_node_ptr tail = tail_.load(memory_order_relaxed);
            tagged_node_ptr next = tail->next.load(memory_order_relaxed);
            node * next_ptr = next.get_ptr();

            if (next_ptr == 0) {
                tail->next.store(tagged_node_ptr(n, next.get_tag() + 1),
                    memory_order_relaxed);
                tail_.store(tagged_node_ptr(n, tail.get_tag() + 1),
                    memory_order_relaxed);
                return true;
            }
            else
                tail_.store(tagged_node_ptr(next_ptr, tail.get_tag() + 1),
                    memory_order_relaxed);
        }
    }

    /** Dequeue object from queue.
     *
     * if pop operation is successful,
     * object is written to memory location denoted by ret.
     *
     * \returns true, if the pop operation is successful, false if queue was empty.
     *
     * \note Thread-safe and non-blocking
     *
     * */
    bool pop (T & ret)
    {
        for (;;) {
            tagged_node_ptr head = head_.load(memory_order_acquire);
            tagged_node_ptr tail = tail_.load(memory_order_acquire);
            tagged_node_ptr next = head->next.load(memory_order_acquire);
            node * next_ptr = next.get_ptr();

//             tagged_node_ptr head2 = head_.load(memory_order_acquire);
//             if (likely(head == head2))
            {
                if (head.get_ptr() == tail.get_ptr()) {
                    if (next_ptr == 0)
                        return false;
                    tail_.compare_exchange_strong(tail, tagged_node_ptr(next_ptr,
                        tail.get_tag() + 1));
                } else {
                    if (next_ptr == 0)
                        /* this check is not part of the original algorithm
                         * as published by michael and scott
                         *
                         * however we reuse the tagged_ptr part for the and
                         * clear the next part during node
                         * allocation. we can observe a null-pointer here.
                         * */
                        continue;
                    ret = next_ptr->data;
                    if (head_.compare_exchange_weak(head, tagged_node_ptr(next_ptr,
                        head.get_tag() + 1))) {
                        pool.destruct(head.get_ptr());
                        return true;
                    }
                }
            }
        }
    }

    /** Dequeue object from queue.
     *
     * if pop operation is successful,
     * object is written to memory location denoted by ret.
     *
     * \returns true, if the pop operation is successful, false if queue was empty.
     *
     * \note Not thread-safe
     *
     * */
    bool pop_unsafe (T & ret)
    {
        for (;;) {
            tagged_node_ptr head = head_.load(memory_order_relaxed);
            tagged_node_ptr tail = tail_.load(memory_order_relaxed);
            tagged_node_ptr next = head->next.load(memory_order_relaxed);
            node * next_ptr = next.get_ptr();

//             tagged_node_ptr head2 = head_.load(memory_order_relaxed);
            if (head.get_ptr() == tail.get_ptr()) {
                if (next_ptr == 0)
                    return false;
                tail_.store(tagged_node_ptr(next_ptr, tail.get_tag() + 1),
                    memory_order_relaxed);
            } else {
                if (next_ptr == 0)
                    /* this check is not part of the original algorithm as
                     * published by michael and scott
                     *
                     * however we reuse the tagged_ptr part for the and clear
                     * the next part during node
                     * allocation. we can observe a null-pointer here.
                     * */
                    continue;
                ret = next_ptr->data;
                head_.store(tagged_node_ptr(next_ptr, head.get_tag() + 1),
                    memory_order_relaxed);
                pool.destruct_unsafe(head.get_ptr());
                return true;
            }
        }
    }


private:
#ifndef BOOST_DOXYGEN_INVOKED
    atomic<tagged_node_ptr> head_;
    static const int padding_size = BOOST_LOCKFREE_CACHELINE_BYTES
        - sizeof(tagged_node_ptr);
    char padding1[padding_size];
    atomic<tagged_node_ptr> tail_;
    char padding2[padding_size];

    pool_t pool;
#endif
};

} /* namespace detail */

/** The queue class provides a multi-writer/multi-reader queue,
 *  pushing and poping is lockfree,
 *  construction/destruction has to be synchronized.
 *  It uses a freelist for memory management,
 *  freed nodes are pushed to the freelist and not returned to the os before
 *  the queue is destroyed.
 *
 *  The memory management of the queue can be controlled via its
 *  freelist_t template argument. Two different
 *  freelists can be used. struct caching_freelist_t selects a caching freelist,
 *  which can allocate more nodes
 *  from the operating system, and struct static_freelist_t uses a fixed-sized freelist.
 *  With a fixed-sized
 *  freelist, the push operation may fail, while with a caching freelist,
 *  the push operation may block.
 *
 *  \b Limitation: The class T is required to have a trivial assignment operator.
 *
 * */
template <typename T,
          typename freelist_t = caching_freelist_t,
          typename Alloc = std::allocator<T>
         >
class queue:
    public detail::queue<T, freelist_t, Alloc>
{
    BOOST_STATIC_ASSERT(boost::has_trivial_assign<T>::value);

public:
    //! Construct queue.
    queue(void)
    {}

    //! Construct queue, allocate n nodes for the freelist.
    explicit queue(std::size_t n):
        detail::queue<T, freelist_t, Alloc>(n)
    {}
};


/** Template specialization of the queue class for pointer arguments,
 *  that supports pop operations to
 *  stl/boost-style smart pointers
 *
 * */
template <typename T,
          typename freelist_t,
          typename Alloc
         >
class queue<T*, freelist_t, Alloc>:
    public detail::queue<T*, freelist_t, Alloc>
{
#ifndef BOOST_DOXYGEN_INVOKED
    typedef detail::queue<T*, freelist_t, Alloc> queue_t;

    template <typename smart_ptr>
    bool pop_smart_ptr(smart_ptr & ptr)
    {
        T * result = 0;
        bool success = queue_t::pop(result);

        if (success)
            ptr.reset(result);
        return success;
    }
#endif

public:
    //! Construct queue.
    queue(void)
    {}

    //! Construct queue, allocate n nodes for the freelist.
    explicit queue(std::size_t n):
        queue_t(n)
    {}

    //! \copydoc detail::queue::pop
    bool pop (T * & ret)
    {
        return queue_t::pop(ret);
    }

    /** Dequeue object from queue to std::auto_ptr
     *
     * if pop operation is successful,
     *  object is written to memory location denoted by ret.
     *
     * \returns true, if the pop operation is successful, false if queue was empty.
     *
     * \note Thread-safe and non-blocking
     *
     * */
    bool pop (std::auto_ptr<T> & ret)
    {
        return pop_smart_ptr(ret);
    }

    /** Dequeue object from queue to boost::scoped_ptr
     *
     * if pop operation is successful,
     *  object is written to memory location denoted by ret.
     *
     * \returns true, if the pop operation is successful, false if queue was empty.
     *
     * \note Thread-safe and non-blocking
     *
     * */
    bool pop (boost::scoped_ptr<T> & ret)
    {
        BOOST_STATIC_ASSERT(sizeof(boost::scoped_ptr<T>) == sizeof(T*));
        return pop(reinterpret_cast<T*&>(ret));
    }

    /** Dequeue object from queue to boost::shared_ptr
     *
     * if pop operation is successful,
     *  object is written to memory location denoted by ret.
     *
     * \returns true, if the pop operation is successful, false if queue was empty.
     *
     * \note Thread-safe and non-blocking
     *
     * */
    bool pop (boost::shared_ptr<T> & ret)
    {
        return pop_smart_ptr(ret);
    }
};

} /* namespace lockfree */
} /* namespace boost */

#endif /* BOOST_LOCKFREE_FIFO_HPP_INCLUDED */
