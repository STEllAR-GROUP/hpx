//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXAMPLE_BFS_CONCURRENT_BGL_QUEUE_JAN_02_2012_0737PM)
#define HPX_EXAMPLE_BFS_CONCURRENT_BGL_QUEUE_JAN_02_2012_0737PM

#include <boost/lockfree/fifo.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace concurrent_bgl
{
    template <typename T>
    struct queue
    {
        void push(T const& t)
        {
            queue_.enqueue(t);
        }

        bool pop(T& ret)
        {
            return queue_.dequeue(ret);
        }

        boost::lockfree::fifo<T> queue_;
    };
}

#endif

