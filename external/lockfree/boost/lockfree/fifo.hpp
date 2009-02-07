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

#define BOOST_LOCKFREE_FIFO_LOGGING

#include <boost/lockfree/prefix.hpp>
#include <boost/lockfree/tagged_ptr.hpp>
#include <boost/lockfree/atomic_int.hpp>
#include <boost/lockfree/freelist.hpp>

#include <boost/concept_check.hpp>
#include <boost/static_assert.hpp>

#if defined(BOOST_LOCKFREE_FIFO_LOGGING)
#include <hpx/util/value_logger.hpp>
#include <hpx/util/block_profiler.hpp>
#endif

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

#if defined(BOOST_LOCKFREE_FIFO_LOGGING)
inline std::string get_enqueue_name(char const* desc)
{
    return std::string(desc) + " enqueue";
}

inline std::string get_dequeue_name(char const* desc)
{
    return std::string(desc) + " dequeue";
}
#endif

template <typename T, typename Alloc>
class fifo:
    boost::noncopyable
{
    BOOST_CLASS_REQUIRE(T, boost, CopyConstructibleConcept);
    BOOST_CLASS_REQUIRE(T, boost, DefaultConstructibleConcept);

    struct BOOST_LOCKFREE_CACHELINE_ALIGNMENT node
    {
        node(T const & v):
            data(v), next(NULL)
        {}

        node (void):
            next(NULL)
        {}

        tagged_ptr<node> next;
        T data;
    };

    typedef tagged_ptr<node> atomic_node_ptr;


public:
    explicit fifo(char const* description = "", bool enable_logging = true)
      : enqueue_spin_count_(0), dequeue_spin_count_(0),
        description_(description)
#if defined(BOOST_LOCKFREE_FIFO_LOGGING)
      , count_(-1), logger_(description, enable_logging)
      , enqueue_profiler_(get_enqueue_name(description).c_str(), enable_logging)
      , dequeue_profiler_(get_dequeue_name(description).c_str(), enable_logging)
#endif
    {
        node * n = alloc_node();
        head_.set_ptr(n);
        tail_.set_ptr(n);
    }

    fifo(std::size_t initial_nodes, char const* description = "",
            bool enable_logging = true)
      : pool(initial_nodes)
      , description_(description)
      , enqueue_spin_count_(0), dequeue_spin_count_(0)
#if defined(BOOST_LOCKFREE_FIFO_LOGGING)
      , count_(-1), logger_(description, enable_logging)
      , enqueue_profiler_(get_enqueue_name(description).c_str(), enable_logging)
      , dequeue_profiler_(get_dequeue_name(description).c_str(), enable_logging)
#endif
    {
        node * n = alloc_node();
        head_.set_ptr(n);
        tail_.set_ptr(n);
    }

    ~fifo(void)
    {
        assert(empty());
        dealloc_node(head_.get_ptr());
    }

    bool empty(void) const
    {
        return head_.get_ptr() == tail_.get_ptr();
    }

    void enqueue(T const & t)
    {
        hpx::util::block_profiler_wrapper<fifo_enqueue_tag> pw(enqueue_profiler_);

        node * n = alloc_node(t);

        for (unsigned int cnt = 0; /**/; spin((unsigned char)++cnt))
        {
            atomic_node_ptr tail (tail_);
            memory_barrier();
            atomic_node_ptr next (tail->next);
            memory_barrier();

            if (likely(tail == tail_))
            {
                if (next.get_ptr() == 0)
                {
                    if (tail->next.CAS(next, n))
                    {
                        tail_.CAS(tail, n);
                        enqueue_spin_count_ += cnt;
                        return;
                    }
                }
                else
                    tail_.CAS(tail, next.get_ptr());
            }
        }
    }

    bool dequeue (T * ret)
    {
        hpx::util::block_profiler_wrapper<fifo_dequeue_tag> pw(dequeue_profiler_);

        for (unsigned int cnt = 0; /**/; spin((unsigned char)++cnt))
        {
            atomic_node_ptr head(head_);
            memory_barrier();

            atomic_node_ptr tail(tail_);
            node * next = head->next.get_ptr();
            memory_barrier();

            if (likely(head == head_))
            {
                if (head.get_ptr() == tail.get_ptr())
                {
                    if (next == 0) {
                        dequeue_spin_count_ += cnt;
                        return false;
                    }
                    tail_.CAS(tail, next);
                }
                else
                {
                    *ret = next->data;
                    if (head_.CAS(head, next))
                    {
                        dealloc_node(head.get_ptr());
                        dequeue_spin_count_ += cnt;
                        return true;
                    }
                }
            }
        }
    }

private:
    node * alloc_node(void)
    {
        node * chunk = pool.allocate();
        new(chunk) node();
#if defined(BOOST_LOCKFREE_FIFO_LOGGING)
        ++count_;
        logger_.snapshot(count_);
#endif
        return chunk;
    }

    node * alloc_node(T const & t)
    {
        node * chunk = pool.allocate();
        new(chunk) node(t);
#if defined(BOOST_LOCKFREE_FIFO_LOGGING)
        ++count_;
        logger_.snapshot(count_);
#endif
        return chunk;
    }

    void dealloc_node(node * n)
    {
#if defined(BOOST_LOCKFREE_FIFO_LOGGING)
        --count_;
        logger_.snapshot(count_);
#endif
        n->~node();
        pool.deallocate(n);
    }

    typedef typename Alloc::template rebind<node>::other node_allocator;

    boost::lockfree::caching_freelist<node, node_allocator> pool;

    /* force head_ and tail_ to different cache lines! */
    atomic_node_ptr head_;
    BOOST_LOCKFREE_CACHELINE_ALIGNMENT_PREFIX atomic_node_ptr tail_ BOOST_LOCKFREE_CACHELINE_ALIGNMENT; 

public:
#if defined(BOOST_LOCKFREE_FIFO_LOGGING)
    atomic_int<long> count_;
    hpx::util::value_logger<long> logger_;

    struct fifo_enqueue_tag {};
    hpx::util::block_profiler<fifo_enqueue_tag> enqueue_profiler_;

    struct fifo_dequeue_tag {};
    hpx::util::block_profiler<fifo_dequeue_tag> dequeue_profiler_;
#endif
    atomic_int<long> enqueue_spin_count_;
    atomic_int<long> dequeue_spin_count_;
    std::string description_;
};

} /* namespace detail */

/** lockfree fifo
 *
 *  - wrapper for detail::fifo
 * */
template <typename T,
          typename Alloc = std::allocator<T> >
class fifo:
    public detail::fifo<T, Alloc>
{
public:
    fifo(char const* description = "", bool enable_logging = true)
      : detail::fifo<T, Alloc>(description, enable_logging)
    {}

    explicit fifo(std::size_t initial_nodes, char const* description = "",
            bool enable_logging = true)
      : detail::fifo<T, Alloc>(initial_nodes, description, enable_logging)
    {}
};


/** lockfree fifo, template specialization for pointer-types
 *
 *  - wrapper for detail::fifo
 *  - overload dequeue to support smart pointers
 * */
template <typename T, typename Alloc>
class fifo<T*, Alloc>:
    public detail::fifo<T*, Alloc>
{
    typedef detail::fifo<T*, Alloc> fifo_t;

    template <typename smart_ptr>
    bool dequeue_smart_ptr(smart_ptr & ptr)
    {
        T * result = 0;
        bool success = fifo_t::dequeue(&result);

        if (success)
            ptr.reset(result);
        return success;
    }

public:
    fifo(char const* description = "", bool enable_logging = true)
      : fifo_t(description, enable_logging)
    {}

    explicit fifo(std::size_t initial_nodes, char const* description = "",
            bool enable_logging = true)
      : fifo_t(initial_nodes, description, enable_logging)
    {}

    void enqueue(T * t)
    {
        fifo_t::enqueue(t);
    }

    bool dequeue (T ** ret)
    {
        return fifo_t::dequeue(ret);
    }

    bool dequeue (std::auto_ptr<T> & ret)
    {
        return dequeue_smart_ptr(ret);
    }

    bool dequeue (boost::scoped_ptr<T> & ret)
    {
        BOOST_STATIC_ASSERT(sizeof(boost::scoped_ptr<T>) == sizeof(T*));
        return dequeue(reinterpret_cast<T**>(&ret));
    }

    bool dequeue (boost::shared_ptr<T> & ret)
    {
        return dequeue_smart_ptr(ret);
    }
};

} /* namespace lockfree */
} /* namespace boost */


#endif /* BOOST_LOCKFREE_FIFO_HPP_INCLUDED */
