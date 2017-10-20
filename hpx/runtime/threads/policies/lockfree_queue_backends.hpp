////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_FB3518C8_4493_450E_A823_A9F8A3185B2D)
#define HPX_FB3518C8_4493_450E_A823_A9F8A3185B2D

#include <hpx/config.hpp>

#include <hpx/util/lockfree/deque.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/lockfree/stack.hpp>

#include <cstddef>
#include <cstdint>

namespace hpx { namespace threads { namespace policies
{

struct lockfree_fifo;
struct lockfree_lifo;

///////////////////////////////////////////////////////////////////////////////
template <typename T, typename Queuing>
struct basic_lockfree_queue_backend
{
    typedef Queuing container_type;
    typedef T value_type;
    typedef T& reference;
    typedef T const& const_reference;
    typedef std::uint64_t size_type;

    basic_lockfree_queue_backend(
        size_type initial_size = 0
      , size_type num_thread = size_type(-1)
        )
      : queue_(std::size_t(initial_size))
    {}

    bool push(const_reference val, bool /*other_end*/ = false)
    {
        return queue_.push(val);
    }

    bool pop(reference val, bool /*steal*/ = true)
    {
        return queue_.pop(val);
    }

    bool empty()
    {
        return queue_.empty();
    }

  private:
    container_type queue_;
};

struct lockfree_fifo
{
    template <typename T>
    struct apply
    {
        typedef basic_lockfree_queue_backend<
            T, boost::lockfree::queue<T>
        > type;
    };
};

struct lockfree_lifo
{
    template <typename T>
    struct apply
    {
        typedef basic_lockfree_queue_backend<
            T, boost::lockfree::stack<T>
        > type;
    };
};

///////////////////////////////////////////////////////////////////////////////
// FIFO + stealing at opposite end.
#if defined(HPX_HAVE_ABP_SCHEDULER)
struct lockfree_abp_fifo;
struct lockfree_abp_lifo;

template <typename T>
struct lockfree_abp_fifo_backend
{
    typedef boost::lockfree::deque<T> container_type;
    typedef T value_type;
    typedef T& reference;
    typedef T const& const_reference;
    typedef std::uint64_t size_type;

    lockfree_abp_fifo_backend(
        size_type initial_size = 0
      , size_type num_thread = size_type(-1)
        )
      : queue_(std::size_t(initial_size))
    {}

    bool push(const_reference val, bool /*other_end*/ = false)
    {
        return queue_.push_left(val);
    }

    bool pop(reference val, bool steal = true)
    {
        if (steal)
            return queue_.pop_left(val);
        return queue_.pop_right(val);
    }

    bool empty()
    {
        return queue_.empty();
    }

  private:
    container_type queue_;
};

struct lockfree_abp_fifo
{
    template <typename T>
    struct apply
    {
        typedef lockfree_abp_fifo_backend<T> type;
    };
};

///////////////////////////////////////////////////////////////////////////////
// LIFO + stealing at opposite end.
// E.g. ABP (Arora, Blumofe and Plaxton) queuing
// http://dl.acm.org/citation.cfm?id=277678
template <typename T>
struct lockfree_abp_lifo_backend
{
    typedef boost::lockfree::deque<T> container_type;
    typedef T value_type;
    typedef T& reference;
    typedef T const& const_reference;
    typedef std::uint64_t size_type;

    lockfree_abp_lifo_backend(
        size_type initial_size = 0
      , size_type num_thread = size_type(-1)
        )
      : queue_(std::size_t(initial_size))
    {}

    bool push(const_reference val, bool other_end = false)
    {
        if (other_end)
            return queue_.push_right(val);
        return queue_.push_left(val);
    }

    bool pop(reference val, bool steal = true)
    {
        if (steal)
            return queue_.pop_right(val);
        return queue_.pop_left(val);
    }

    bool empty()
    {
        return queue_.empty();
    }

  private:
    container_type queue_;
};

struct lockfree_abp_lifo
{
    template <typename T>
    struct apply
    {
        typedef lockfree_abp_lifo_backend<T> type;
    };
};

#endif // HPX_HAVE_ABP_SCHEDULER

}}}

#endif // HPX_FB3518C8_4493_450E_A823_A9F8A3185B2D

