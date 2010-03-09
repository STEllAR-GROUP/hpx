//  lock-free single-producer/single-consumer ringbuffer
//  this algorithm is implemented in various projects (linux kernel)
//
//  implementation for c++
//
//  Copyright (C) 2009 Tim Blechmann
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//  Disclaimer: Not a Boost library.

#ifndef BOOST_LOCKFREE_RINGBUFFER_HPP_INCLUDED
#define BOOST_LOCKFREE_RINGBUFFER_HPP_INCLUDED

#include <boost/atomic.hpp>
#include <boost/array.hpp>
#include <boost/noncopyable.hpp>
#include <boost/smart_ptr/scoped_array.hpp>

#include "detail/branch_hints.hpp"
#include "detail/prefix.hpp"

#include <algorithm>

namespace boost
{
namespace lockfree
{

namespace detail
{

template <typename T>
class ringbuffer_internal:
    boost::noncopyable
{
    typedef std::size_t size_t;
    static const int padding_size = BOOST_LOCKFREE_CACHELINE_BYTES - sizeof(size_t);
    atomic<size_t> write_index_;
    char padding1[padding_size]; /* force read_index and write_index to different cache lines */
    atomic<size_t> read_index_;

protected:
    ringbuffer_internal(void):
        write_index_(0), read_index_(0)
    {}

    static size_t next_index(size_t arg, size_t max_size)
    {
        size_t ret = arg + 1;
        while (unlikely(ret >= max_size))
            ret -= max_size;
        return ret;
    }

    static size_t read_available(size_t write_index, size_t read_index, size_t max_size)
    {
        if (write_index >= read_index)
            return write_index - read_index;

        size_t ret = write_index + max_size - read_index;
        return ret;
    }

    static size_t write_available(size_t write_index, size_t read_index, size_t max_size)
    {
        size_t ret = read_index - write_index - 1;
        if (write_index > read_index)
            ret += max_size;
        return ret;
    }

    bool enqueue(T const & t, T * buffer, size_t max_size)
    {
        size_t next = next_index(write_index_.load(memory_order_acquire), max_size);

        if (next == read_index_.load(memory_order_acquire))
            return false; /* ringbuffer is full */

        buffer[next] = t;

        write_index_.store(next, memory_order_release);

        return true;
    }

    bool dequeue (T * ret, T * buffer, size_t max_size)
    {
        size_t write_index = write_index_.load(memory_order_acquire);
        size_t read_index  = read_index_.load(memory_order_acquire);
        if (empty(write_index, read_index))
            return false;

        size_t next = next_index(read_index, max_size);
        *ret = buffer[next];
        read_index_.store(next, memory_order_release);
        return true;
    }

    size_t enqueue(const T * input_buffer, size_t input_count, T * internal_buffer, size_t max_size)
    {
        size_t write_index = write_index_.load(memory_order_acquire);
        const size_t read_index  = read_index_.load(memory_order_acquire);
        const size_t avail = write_available(write_index, read_index, max_size);

        if (avail == 0)
            return 0;

        input_count = std::min(input_count, avail);

        size_t new_write_index = write_index + input_count;

        if (write_index + input_count > max_size)
        {
            /* copy data in two sections */
            size_t count0 = max_size - write_index;

            std::copy(input_buffer, input_buffer + count0, internal_buffer + write_index);
            std::copy(input_buffer + count0, input_buffer + input_count, internal_buffer);
            new_write_index -= max_size;
        }
        else
        {
            std::copy(input_buffer, input_buffer + input_count, internal_buffer + write_index);

            if (new_write_index == max_size)
                new_write_index = 0;
        }

        write_index_.store(new_write_index, memory_order_release);
        return input_count;
    }

    size_t dequeue (T * output_buffer, size_t output_count, const T * internal_buffer, size_t max_size)
    {
        const size_t write_index = write_index_.load(memory_order_acquire);
        size_t read_index = read_index_.load(memory_order_acquire);

        const size_t avail = read_available(write_index, read_index, max_size);

        if (avail == 0)
            return 0;

        output_count = std::min(output_count, avail);

        size_t new_read_index = read_index + output_count;

        if (read_index + output_count > max_size)
        {
            /* copy data in two sections */
            size_t count0 = max_size - read_index;
            size_t count1 = output_count - count0;

            std::copy(internal_buffer + read_index, internal_buffer + max_size, output_buffer);
            std::copy(internal_buffer, internal_buffer + count1, output_buffer + count0);

            new_read_index -= max_size;
        }
        else
        {
            std::copy(internal_buffer + read_index, internal_buffer + read_index + output_count, output_buffer);
            if (new_read_index == max_size)
                new_read_index = 0;
        }

        read_index_.store(new_read_index, memory_order_release);
        return output_count;
    }


public:
    void reset(void)
    {
        write_index_.store(0, memory_order_relaxed);
        read_index_.store(0, memory_order_release);
    }

    bool empty(void)
    {
        return empty(write_index_.load(memory_order_relaxed), read_index_.load(memory_order_relaxed));
    }

    bool is_lock_free(void) const
    {
        return write_index_.is_lock_free() && read_index_.is_lock_free();
    }

private:
    bool empty(size_t write_index, size_t read_index)
    {
        return write_index == read_index;
    }
};

} /* namespace detail */

template <typename T, size_t max_size>
class ringbuffer:
    public detail::ringbuffer_internal<T>
{
    boost::array<T, max_size> array_;

public:
    bool enqueue(T const & t)
    {
        return detail::ringbuffer_internal<T>::enqueue(t, array_.c_array(), max_size);
    }

    bool dequeue(T * ret)
    {
        return detail::ringbuffer_internal<T>::dequeue(ret, array_.c_array(), max_size);
    }

    size_t enqueue(T const * t, size_t size)
    {
        return detail::ringbuffer_internal<T>::enqueue(t, size, array_.c_array(), max_size);
    }

    size_t dequeue(T * ret, size_t size)
    {
        return detail::ringbuffer_internal<T>::dequeue(ret, size, array_.c_array(), max_size);
    }
};

template <typename T>
class ringbuffer<T, 0>:
    public detail::ringbuffer_internal<T>
{
    size_t max_size_;
    scoped_array<T> array_;

public:
    explicit ringbuffer(size_t max_size):
        max_size_(max_size), array_(new T[max_size])
    {}

    bool enqueue(T const & t)
    {
        return detail::ringbuffer_internal<T>::enqueue(t, array_.get(), max_size_);
    }

    bool dequeue(T * ret)
    {
        return detail::ringbuffer_internal<T>::dequeue(ret, array_.get(), max_size_);
    }

    size_t enqueue(T const * t, size_t size)
    {
        return detail::ringbuffer_internal<T>::enqueue(t, size, array_.get(), max_size_);
    }

    size_t dequeue(T * ret, size_t size)
    {
        return detail::ringbuffer_internal<T>::dequeue(ret, size, array_.get(), max_size_);
    }
};


} /* namespace lockfree */
} /* namespace boost */


#endif /* BOOST_LOCKFREE_RINGBUFFER_HPP_INCLUDED */
