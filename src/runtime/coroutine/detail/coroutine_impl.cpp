//  Copyright (c) 2006, Giovanni P. Deretta
//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  This code may be used under either of the following two licences:
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE. OF SUCH DAMAGE.
//
//  Or:
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/runtime/coroutine/coroutine.hpp>
#include <hpx/runtime/coroutine/detail/coroutine_impl.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/util/reinitializable_static.hpp>

#include <boost/lockfree/stack.hpp>

#include <cstddef>

namespace hpx { namespace coroutines { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // the memory for the threads is managed by a lockfree caching_freelist
    struct coroutine_heap
    {
        typedef coroutine_impl<coroutine> coroutine_type;

        coroutine_heap()
          : heap_(128)
        {}

        ~coroutine_heap()
        {
            while (coroutine_type* next = get_locked())
                delete next;
        }

        coroutine_type* allocate()
        {
            return get_locked();
        }

        void deallocate(coroutine_type* p)
        {
            //p->reset();          // reset bound function
            heap_.push(p);
        }

    private:
        coroutine_type* get_locked()
        {
            coroutine_type* result = 0;
            heap_.pop(result);
            return result;
        }

        boost::lockfree::stack<coroutine_type*> heap_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct heap_tag_small {};
    struct heap_tag_medium {};
    struct heap_tag_large {};
    struct heap_tag_huge {};

    template <std::size_t NumHeaps, typename Tag>
    static coroutine_heap& get_heap(std::size_t i)
    {
        // ensure thread-safe initialization
        util::reinitializable_static<coroutine_heap, Tag, NumHeaps> heap;
        return heap.get(i);
    }

    static coroutine_heap& get_heap(std::size_t i, std::ptrdiff_t stacksize)
    {
        // FIXME: This should check the sizes in runtime_configuration, not the
        // default macro sizes
        if (stacksize > HPX_MEDIUM_STACK_SIZE)
        {
            if (stacksize > HPX_LARGE_STACK_SIZE)
                return get_heap<HPX_COROUTINE_NUM_HEAPS / 4,
                heap_tag_huge>(i % (HPX_COROUTINE_NUM_HEAPS / 4)); //-V112

            return get_heap<HPX_COROUTINE_NUM_HEAPS / 4,
                heap_tag_large>(i % (HPX_COROUTINE_NUM_HEAPS / 4)); //-V112
        }

        if (stacksize > HPX_SMALL_STACK_SIZE)
            return get_heap<HPX_COROUTINE_NUM_HEAPS / 2,
            heap_tag_medium>(i % (HPX_COROUTINE_NUM_HEAPS / 2));

        return get_heap<HPX_COROUTINE_NUM_HEAPS,
            heap_tag_small>(i % HPX_COROUTINE_NUM_HEAPS);
    }

    static std::size_t get_heap_count(ptrdiff_t stacksize)
    {
        if (stacksize > HPX_MEDIUM_STACK_SIZE)
            return HPX_COROUTINE_NUM_HEAPS / 4; //-V112

        if (stacksize > HPX_SMALL_STACK_SIZE)
            return HPX_COROUTINE_NUM_HEAPS / 2;

        return HPX_COROUTINE_NUM_HEAPS;
    }

    template <typename CoroutineType>
    coroutine_impl<CoroutineType>* coroutine_impl<CoroutineType>::allocate(
        thread_id_repr_type id, std::ptrdiff_t stacksize)
    {
        // start looking at the matching heap
        std::size_t const heap_num = std::size_t(id) / 32; //-V112
        std::size_t const heap_count = get_heap_count(stacksize);

        // look through all heaps to find an available coroutine object
        coroutine_impl* p = get_heap(heap_num, stacksize).allocate();
        if (!p)
        {
            for (std::size_t i = 1; i != heap_count && !p; ++i)
            {
                p = get_heap(heap_num + i, stacksize).allocate();
            }
        }
        return p;
    }

    template <typename CoroutineType>
    void coroutine_impl<CoroutineType>::deallocate(
        coroutine_impl<CoroutineType>* p)
    {
        std::size_t const heap_num = std::size_t(p->get_thread_id()) / 32; //-V112
        std::ptrdiff_t const stacksize = p->get_stacksize();

        get_heap(heap_num, stacksize).deallocate(p);
    }
}}}

///////////////////////////////////////////////////////////////////////////////
// explicit instantiation of the coroutine_impl functions
template HPX_EXPORT
hpx::threads::thread_self_impl_type*
hpx::threads::thread_self_impl_type::allocate(
    thread_id_repr_type id, std::ptrdiff_t stacksize);

template HPX_EXPORT void
hpx::threads::thread_self_impl_type::deallocate(
      hpx::threads::thread_self_impl_type* p);
