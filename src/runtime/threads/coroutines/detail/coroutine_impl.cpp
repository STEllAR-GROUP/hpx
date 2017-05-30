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
#include <hpx/compat/exception.hpp>

#include <hpx/runtime/threads/coroutines/coroutine.hpp>
#include <hpx/runtime/threads/coroutines/detail/coroutine_impl.hpp>
#include <hpx/runtime/threads/coroutines/detail/coroutine_self.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/reinitializable_static.hpp>

#include <boost/lockfree/stack.hpp>

#include <cstddef>
#include <utility>

namespace hpx { namespace threads { namespace coroutines { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    namespace
    {
        struct reset_self_on_exit
        {
            reset_self_on_exit(coroutine_self* val,
                    coroutine_self* old_val = nullptr)
              : old_self(old_val)
            {
                coroutine_self::set_self(val);
            }

            ~reset_self_on_exit()
            {
                coroutine_self::set_self(old_self);
            }

            coroutine_self* old_self;
        };
    }

#if defined(HPX_DEBUG)
    coroutine_impl::~coroutine_impl()
    {
        HPX_ASSERT(!m_fun);   // functor should have been reset by now
    }
#endif

    void coroutine_impl::operator()()
    {
        typedef super_type::context_exit_status context_exit_status;
        context_exit_status status = super_type::ctx_exited_return;

        // loop as long this coroutine has been rebound
        do
        {
            compat::exception_ptr tinfo;
            try
            {
                this->check_exit_state();

                HPX_ASSERT(this->count() > 0);

                {
                    coroutine_self* old_self = coroutine_self::get_self();
                    coroutine_self self(this, old_self);
                    reset_self_on_exit on_exit(&self, old_self);

                    this->m_result_last = m_fun(*this->args());

                    // if this thread returned 'terminated' we need to reset
                    // the functor and the bound arguments
                    if (this->m_result_last.first == terminated)
                        this->reset();
                }

                // return value to other side of the fence
                this->bind_result(&this->m_result_last);
            }
            catch (exit_exception const&) {
                status = super_type::ctx_exited_exit;
                tinfo = compat::current_exception();
                this->reset();            // reset functor
            }
            catch (boost::exception const&) {
                status = super_type::ctx_exited_abnormally;
                tinfo = compat::current_exception();
                this->reset();
            }
            catch (std::exception const&) {
                status = super_type::ctx_exited_abnormally;
                tinfo = compat::current_exception();
                this->reset();
            }
            catch (...) {
                status = super_type::ctx_exited_abnormally;
                tinfo = compat::current_exception();
                this->reset();
            }

            this->do_return(status, std::move(tinfo));

        } while (this->m_state == super_type::ctx_running);

        // should not get here, never
        HPX_ASSERT(this->m_state == super_type::ctx_running);
    }

    ///////////////////////////////////////////////////////////////////////////
    // the memory for the threads is managed by a lockfree caching_freelist
    struct coroutine_heap
    {
        coroutine_heap()
          : heap_(128)
        {}

        ~coroutine_heap()
        {
            while (coroutine_impl* next = get_locked())
                delete next;
        }

        coroutine_impl* allocate()
        {
            return get_locked();
        }

        void deallocate(coroutine_impl* p)
        {
            //p->reset();          // reset bound function
            heap_.push(p);
        }

    private:
        coroutine_impl* get_locked()
        {
            coroutine_impl* result = nullptr;
            heap_.pop(result);
            return result;
        }

        boost::lockfree::stack<coroutine_impl*> heap_;
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

    coroutine_impl* coroutine_impl::allocate(
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

    void coroutine_impl::deallocate(coroutine_impl* p)
    {
        std::size_t const heap_num = std::size_t(p->get_thread_id()) / 32; //-V112
        std::ptrdiff_t const stacksize = p->get_stacksize();

        get_heap(heap_num, stacksize).deallocate(p);
    }
}}}}
