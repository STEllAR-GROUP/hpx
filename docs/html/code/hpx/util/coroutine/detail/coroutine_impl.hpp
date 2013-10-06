//  Copyright (c) 2006, Giovanni P. Deretta
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

#ifndef HPX_COROUTINE_COROUTINE_IMPL_HPP_20060601
#define HPX_COROUTINE_COROUTINE_IMPL_HPP_20060601

#if defined(_MSC_VER)
#pragma warning (push)
#pragma warning (disable: 4355) //this used in base member initializer
#endif

#include <cstddef>
#include <boost/assert.hpp>
#include <boost/optional.hpp>
#include <boost/config.hpp>
#include <boost/move/move.hpp>

#include <hpx/config.hpp>
#include <hpx/util/coroutine/detail/config.hpp>
#include <hpx/util/coroutine/detail/argument_unpacker.hpp>
#include <hpx/util/coroutine/detail/coroutine_accessor.hpp>
#include <hpx/util/coroutine/detail/context_base.hpp>
#include <hpx/util/coroutine/detail/self.hpp>
#include <hpx/util/reinitializable_static.hpp>
#include <hpx/util/thread_specific_ptr.hpp>
#include <hpx/util/decay.hpp>

namespace hpx { namespace util { namespace coroutines { namespace detail
{
  /////////////////////////////////////////////////////////////////////////////
  // This class augment the contest_base class with
  // the coroutine signature type.
  // This is mostly just a place to put
  // typesafe argument and result type pointers.
  template<typename CoroutineType, typename ContextImpl,
      template <typename> class Heap>
  class coroutine_impl
    : public context_base<ContextImpl>
  {
  public:
    template <typename DerivedType, typename ResultType>
    friend class add_result;

    typedef ContextImpl context_impl;
    typedef CoroutineType coroutine_type;
    typedef coroutine_impl<coroutine_type, context_impl, Heap> type;
    typedef context_base<context_impl> context_base_;
    typedef typename coroutine_type::arg_slot_type arg_slot_type;
    typedef typename coroutine_type::arg_slot_traits arg_slot_traits;
    typedef typename coroutine_type::result_type result_type;
    typedef typename coroutine_type::result_slot_type result_slot_type;
    typedef typename context_base_::thread_id_type thread_id_type;

    typedef boost::intrusive_ptr<type> pointer;

    template <typename DerivedType>
    coroutine_impl(DerivedType *this_, thread_id_type id,
            std::ptrdiff_t stack_size)
      : context_base_(*this_, stack_size, id),
        m_arg(0),
        m_result(0)
    {}

    template <typename Functor>
    static inline pointer create(BOOST_FWD_REF(Functor), thread_id_type = 0,
        std::ptrdiff_t = default_stack_size);

#if HPX_COROUTINE_ARG_MAX > 1
    result_slot_type * result() {
      BOOST_ASSERT(m_result);
      BOOST_ASSERT(*m_result);
      return *this->m_result;
    }

    arg_slot_type * args() {
      BOOST_ASSERT(m_arg);
      return m_arg;
    };

    void bind_args(arg_slot_type* arg) {
      m_arg = arg;
    }

    void bind_result(result_slot_type* res) {
      *m_result = res;
    }

    // Another level of indirection is needed to handle
    // yield_to correctly.
    void bind_result_pointer(result_slot_type** resp) {
      m_result = resp;
    }

    result_slot_type** result_pointer() {
      return m_result;
    }
#else
    result_type * result() {
      BOOST_ASSERT(m_result);
      return *this->m_result;
    }

    typedef typename arg_slot_traits::template at<0>::type arg0_type;
    arg0_type * args() {
      BOOST_ASSERT(m_arg);
      return m_arg;
    };

    void bind_args(arg0_type* arg) {
      m_arg = arg;
    }

    void bind_result(result_type* res) {
      *m_result = res;
    }

    // Another level of indirection is needed to handle
    // yield_to correctly.
    void bind_result_pointer(result_type** resp) {
      m_result = resp;
    }

    result_type** result_pointer() {
      return m_result;
    }
#endif

    // This function must be called only for void
    // coroutines. It wakes up the coroutine.
    // Entering the wait state does not cause this
    // method to throw.
    void run() {
      arg_slot_type void_args;
      result_slot_type * ptr = 0;

      // This dummy binding is required because
      // do_call expect args() and result()
      // to return a non NULL result.
      bind_args(&void_args);
      bind_result_pointer(&ptr);
      this->wake_up();
    }

#if HPX_THREAD_MAINTAIN_PHASE_INFORMATION
    std::size_t get_thread_phase() const
    {
        return this->phase();
    }
#endif

    struct tls_tag {};

  private:
    typedef detail::coroutine_self<coroutine_type> self_type;
    static hpx::util::thread_specific_ptr<self_type*, tls_tag> self_;

  public:
    HPX_COROUTINE_EXPORT static void set_self(self_type* self);
    HPX_COROUTINE_EXPORT static self_type* get_self();
    HPX_COROUTINE_EXPORT static void init_self();
    HPX_COROUTINE_EXPORT static void reset_self();

#if defined(HPX_GENERIC_COROUTINES)
  protected:
    boost::optional<result_slot_type> m_result_last;

  private:
    arg_slot_type * m_arg;
    result_slot_type ** m_result;
#else
  protected:
    result_type m_result_last;
    arg0_type* m_arg;
    result_type** m_result;
#endif
  };

  // the TLS holds a pointer to the self instance as stored on the stack
  template<typename CoroutineType, typename ContextImpl,
      template <typename> class Heap>
  hpx::util::thread_specific_ptr<
      typename coroutine_impl<CoroutineType, ContextImpl, Heap>::self_type*
    , typename coroutine_impl<CoroutineType, ContextImpl, Heap>::tls_tag
  > coroutine_impl<CoroutineType, ContextImpl, Heap>::self_;

  /////////////////////////////////////////////////////////////////////////////
  template <typename Coroutine, template <typename> class Heap>
  struct coroutine_heap
  {
      ~coroutine_heap()
      {
          Coroutine* next = heap_.get();
          while (next) {
              delete next;
              next = heap_.get();
          }
      }

      Coroutine* allocate()
      {
          return heap_.get();
      }

      Coroutine* try_allocate()
      {
          return heap_.try_get();
      }

      void deallocate(Coroutine* p)
      {
//          p->reset();          // reset bound function
          heap_.deallocate(p);
      }

      Heap<Coroutine> heap_;
  };

  /////////////////////////////////////////////////////////////////////////////
  // This type augments coroutine_impl type with the type of the stored
  // functor. The type of this object is erased right after construction
  // when it is assigned to a pointer to coroutine_impl. A deleter is
  // passed down to make sure that the correct derived type is deleted.
  template<typename FunctorType, typename CoroutineType, typename ContextImpl,
      template <typename> class Heap>
  class coroutine_impl_wrapper :
    public coroutine_impl<CoroutineType, ContextImpl, Heap>
  {
  public:
    typedef coroutine_impl_wrapper<
        FunctorType, CoroutineType, ContextImpl, Heap> type;
    typedef CoroutineType coroutine_type;
    typedef typename CoroutineType::result_type result_type;
    typedef coroutine_impl<CoroutineType, ContextImpl, Heap> super_type;
    typedef typename super_type::thread_id_type thread_id_type;

    typedef typename util::decay<FunctorType>::type functor_type;

    template <typename Functor>
    coroutine_impl_wrapper(BOOST_FWD_REF(Functor) f, thread_id_type id,
                std::ptrdiff_t stack_size)
      : super_type(this, id, stack_size),
        m_fun(boost::forward<Functor>(f))
    {}

    ~coroutine_impl_wrapper()
    {
        BOOST_ASSERT(!m_fun);   // functor should have been reset by now
    }

    void operator()()
    {
      typedef typename super_type::context_exit_status
        context_exit_status;
      context_exit_status status = super_type::ctx_exited_return;

      // loop as long this coroutine has been rebound
      do {
        boost::exception_ptr tinfo;
        try {
          this->check_exit_state();

#if defined(HPX_GENERIC_COROUTINES)
          do_call<result_type>();
#else
          BOOST_ASSERT(this->count() > 0);

          typedef typename coroutine_type::self self_type;
          {
              self_type* old_self = super_type::get_self();
              self_type self(this, old_self);
              reset_self_on_exit on_exit(&self, old_self);

              this->m_result_last = m_fun(*this->args());

              // if this thread returned 'terminated' we need to reset the functor
              // and the bound arguments
              //
              // Note: threads::terminated == 5
              //
              if (this->m_result_last == 5)
                  this->reset();
          }

          // return value to other side of the fence
          this->bind_result(&this->m_result_last);
#endif
        }
        catch (exit_exception const&) {
          status = super_type::ctx_exited_exit;
          tinfo = boost::current_exception();
          this->reset();            // reset functor
        }
        catch (boost::exception const&) {
          status = super_type::ctx_exited_abnormally;
          tinfo = boost::current_exception();
          this->reset();
        }
        catch (std::exception const&) {
          status = super_type::ctx_exited_abnormally;
          tinfo = boost::current_exception();
          this->reset();
        }
        catch (...) {
          status = super_type::ctx_exited_abnormally;
          tinfo = boost::current_exception();
          this->reset();
        }

        this->do_return(status, tinfo);

      } while (this->m_state == super_type::ctx_running);

      // should not get here, never
      BOOST_ASSERT(this->m_state == super_type::ctx_running);
    }

  protected:
    struct reset_self_on_exit
    {
        typedef typename coroutine_type::self self_type;

        reset_self_on_exit(self_type* val, self_type* old_val = 0)
          : old_self(old_val)
        {
            super_type::set_self(val);
        }
        ~reset_self_on_exit()
        {
            super_type::set_self(old_self);
        }

        self_type* old_self;
    };

  public:

#if defined(HPX_GENERIC_COROUTINES)
    //GCC workaround as per enable_if docs
    template <int> struct dummy { dummy(int) {} };

    /*
     * Implementation for operator()
     * This is for void result types.
     * Can throw if m_fun throws. At least it can throw exit_exception.
     */
    template <typename ResultType>
    typename boost::enable_if<boost::is_void<ResultType> >::type
    do_call(dummy<0> = 0)
    {
      BOOST_ASSERT(this->count() > 0);

      typedef typename coroutine_type::self self_type;

      // In this particular case result_slot_type is guaranteed to be
      // default constructible.
      typedef typename coroutine_type::result_slot_type result_slot_type;

      {
          self_type* old_self = super_type::get_self();
          self_type self(this, old_self);
          reset_self_on_exit on_exit(&self, old_self);
          detail::unpack(m_fun, *this->args(),
              detail::trait_tag<typename coroutine_type::arg_slot_traits>());
      }

      this->m_result_last = result_slot_type();
      this->bind_result(&*this->m_result_last);
    }

    // Same as above, but for non void result types.
    template <typename ResultType>
    typename boost::disable_if<boost::is_void<ResultType> >::type
    do_call(dummy<1> = 1)
    {
      BOOST_ASSERT(this->count() > 0);

      typedef typename coroutine_type::self self_type;
      typedef typename coroutine_type::arg_slot_traits traits;
      typedef typename coroutine_type::result_slot_type result_slot_type;

      {
          self_type* old_self = super_type::get_self();
          self_type self(this, old_self);
          reset_self_on_exit on_exit(&self, old_self);
          this->m_result_last = boost::in_place(result_slot_type(
                  detail::unpack(m_fun, *this->args(), detail::trait_tag<traits>())
              ));

          // if this thread returned 'terminated' we need to reset the functor
          // and the bound arguments
          //
          // Note: threads::terminated == 5
          //
          if (this->m_result_last && boost::get<0>(this->m_result_last.get()) == 5)
              this->reset();
      }

      // return value to other side of the fence
      this->bind_result(&*this->m_result_last);
    }
#endif

    static inline void destroy(type* p);
    static inline void reset(type* p);

    void reset()
    {
        this->reset_stack();
        m_fun.clear();    // just reset the bound function
    }

    template <typename Functor>
    void rebind(BOOST_FWD_REF(Functor) f, thread_id_type id)
    {
        this->rebind_stack();     // count how often a coroutines object was reused
        m_fun = boost::forward<Functor>(f);
        this->super_type::rebind(id);
    }

    // the memory for the threads is managed by a lockfree caching_freelist
    typedef coroutine_heap<coroutine_impl_wrapper, Heap> heap_type;

  private:
    struct heap_tag_small {};
    struct heap_tag_medium {};
    struct heap_tag_large {};
    struct heap_tag_huge {};

    template <std::size_t NumHeaps, typename Tag>
    static heap_type& get_heap(std::size_t i)
    {
        // ensure thread-safe initialization
        util::reinitializable_static<heap_type, Tag, NumHeaps> heap;
        return heap.get(i);
    }

    static heap_type& get_heap(std::size_t i, ptrdiff_t stacksize)
    {
        if (stacksize > HPX_MEDIUM_STACK_SIZE) {
            if (stacksize > HPX_LARGE_STACK_SIZE)
                return get_heap<HPX_COROUTINE_NUM_HEAPS/4, heap_tag_huge>(i % (HPX_COROUTINE_NUM_HEAPS/4)); //-V112

            return get_heap<HPX_COROUTINE_NUM_HEAPS/4, heap_tag_large>(i % (HPX_COROUTINE_NUM_HEAPS/4)); //-V112
        }
        if (stacksize > HPX_SMALL_STACK_SIZE)
            return get_heap<HPX_COROUTINE_NUM_HEAPS/2, heap_tag_medium>(i % (HPX_COROUTINE_NUM_HEAPS/2));

        return get_heap<HPX_COROUTINE_NUM_HEAPS, heap_tag_small>(i % HPX_COROUTINE_NUM_HEAPS);
    }

  public:
    static std::size_t get_heap_count(ptrdiff_t stacksize)
    {
        if (stacksize > HPX_MEDIUM_STACK_SIZE)
            return HPX_COROUTINE_NUM_HEAPS/4; //-V112

        if (stacksize > HPX_SMALL_STACK_SIZE)
            return HPX_COROUTINE_NUM_HEAPS/2;

        return HPX_COROUTINE_NUM_HEAPS;
    }

    static coroutine_impl_wrapper* allocate(std::size_t i, ptrdiff_t stacksize)
    {
        return get_heap(i, stacksize).allocate();
    }
    static coroutine_impl_wrapper* try_allocate(std::size_t i, ptrdiff_t stacksize)
    {
        return get_heap(i, stacksize).try_allocate();
    }
    static void deallocate(coroutine_impl_wrapper* wrapper, std::size_t i)
    {
        ptrdiff_t stacksize = wrapper->get_stacksize();
        get_heap(i, stacksize).deallocate(wrapper);
    }

#if defined(_DEBUG)
    static heap_type const* get_first_heap_address(ptrdiff_t stacksize)
    {
        return &get_heap(0, stacksize);
    }
#endif

    functor_type m_fun;
  };

  template<typename CoroutineType, typename ContextImpl, template <typename> class Heap>
  template<typename Functor>
  inline typename coroutine_impl<CoroutineType, ContextImpl, Heap>::pointer
  coroutine_impl<CoroutineType, ContextImpl, Heap>::
      create(BOOST_FWD_REF(Functor) f, thread_id_type id, std::ptrdiff_t stack_size)
  {
      typedef typename hpx::util::decay<Functor>::type functor_type;

      typedef coroutine_impl_wrapper<
          functor_type, CoroutineType, ContextImpl, Heap> wrapper_type;

      // start looking at the matching heap
      std::size_t const heap_count = wrapper_type::get_heap_count(stack_size);
      std::size_t const heap_num = std::size_t(id)/8;

      // look through all heaps to find an available coroutine object
      wrapper_type* wrapper = wrapper_type::allocate(heap_num, stack_size);
      for (std::size_t i = 1; i < heap_count && !wrapper; ++i) {
          wrapper = wrapper_type::try_allocate(heap_num + i, stack_size);
      }

#if defined(_DEBUG)
      wrapper_type::heap_type const* heaps =
          wrapper_type::get_first_heap_address(stack_size);
#endif

      // allocate a new coroutine object, if non is available (or all heaps are locked)
      if (NULL == wrapper) {
          context_base<ContextImpl>::increment_allocation_count(heap_num);
          return new wrapper_type(boost::forward<Functor>(f), id, stack_size);
      }

      // if we reuse an existing  object, we need to rebind its function
      wrapper->rebind(boost::forward<Functor>(f), id);
      return wrapper;
  }

  template<typename Functor, typename CoroutineType, typename ContextImpl,
      template <typename> class Heap>
  inline void
  coroutine_impl_wrapper<Functor, CoroutineType, ContextImpl, Heap>::destroy(type* p)
  {
#if defined(_DEBUG)
      typedef coroutine_impl_wrapper<
          functor_type, CoroutineType, ContextImpl, Heap> wrapper_type;
      wrapper_type::heap_type const* heaps =
          wrapper_type::get_first_heap_address(p->get_stacksize());
#endif
      // always hand the stack back to the matching heap
      deallocate(p, std::size_t(p->get_thread_id())/8);
  }


  template<typename Functor, typename CoroutineType, typename ContextImpl,
      template <typename> class Heap>
  inline void
  coroutine_impl_wrapper<Functor, CoroutineType, ContextImpl, Heap>::reset(type* p)
  {
      p->reset();
  }
}}}}

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif

