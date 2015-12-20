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

#ifndef HPX_COROUTINE_COROUTINE_HPP_20060512
#define HPX_COROUTINE_COROUTINE_HPP_20060512

#include <hpx/config.hpp>

// This needs to be first for building on Macs
#include <hpx/util/coroutine/detail/default_context_impl.hpp>

#include <cstddef>
#include <boost/preprocessor/repetition.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/type_traits.hpp>
#include <boost/call_traits.hpp>
#include <utility>

#include <hpx/util/coroutine/detail/coroutine_impl.hpp>
#include <hpx/util/coroutine/detail/is_callable.hpp>
#include <hpx/util/coroutine/detail/signature.hpp>
#include <hpx/util/coroutine/detail/index.hpp>
#include <hpx/util/coroutine/detail/coroutine_traits.hpp>
#include <hpx/util/coroutine/detail/coroutine_accessor.hpp>
#include <hpx/util/coroutine/detail/fix_result.hpp>
#include <hpx/util/coroutine/detail/self.hpp>
#include <hpx/runtime/naming/id_type.hpp>

namespace hpx { namespace util { namespace coroutines
{
  namespace detail {
    template<typename T>
    struct optional_result_type :
      boost::mpl::if_<boost::is_same<T, void>,
                      void,
                      boost::optional<T> > { };

    template<typename T>
    typename
    boost::enable_if<boost::is_same<T, void> >::type
    optional_result() {}

    template<typename T>
    typename
    boost::disable_if<boost::is_same<T, void>,
                      typename
                      optional_result_type<T>::type
                      >::type
    optional_result() {
      return typename
        optional_result_type<T>::type();
    }
  }

  template<typename Signature, template <typename> class Heap, typename Context>
  class coroutine;

  template<typename T>
  struct is_coroutine : boost::mpl::false_{};

  template<typename Signature, template <typename> class Heap, typename Context>
  struct is_coroutine<coroutine<Signature, Heap, Context> >
    : boost::mpl::true_ {};

  /////////////////////////////////////////////////////////////////////////////
  namespace detail
  {
    template <typename CoroutineImpl>
    struct coroutine_allocator
    {
        CoroutineImpl* get()
        {
            return NULL;
        }

        void deallocate(CoroutineImpl* c)
        {
            delete c;
        }
    };
  }

  /////////////////////////////////////////////////////////////////////////////
  template<typename Signature,
           template <typename> class Heap,
           typename ContextImpl>
  class coroutine
  {
  private:
    HPX_MOVABLE_BUT_NOT_COPYABLE(coroutine)

  public:
    typedef coroutine<Signature, Heap, ContextImpl> type;
    typedef ContextImpl context_impl;
    typedef Signature signature_type;
    typedef detail::coroutine_traits<signature_type> traits_type;

    friend struct detail::coroutine_accessor;

    typedef typename traits_type::result_type result_type;
    typedef typename traits_type::result_slot_type result_slot_type;
    typedef typename traits_type::yield_result_type yield_result_type;
    typedef typename traits_type::result_slot_traits result_slot_traits;
    typedef typename traits_type::arg_slot_type arg_slot_type;
    typedef typename traits_type::arg_slot_traits arg_slot_traits;

    typedef detail::coroutine_impl<type, context_impl, Heap> impl_type;
    typedef typename impl_type::pointer impl_ptr;
    typedef typename impl_type::thread_id_repr_type thread_id_repr_type;

    typedef detail::coroutine_self<type> self;
    coroutine() : m_pimpl(0) {}

    template <typename Functor>
    coroutine (Functor && f, naming::id_type && target,
            thread_id_repr_type id = 0, std::ptrdiff_t stack_size =
               detail::default_stack_size)
      : m_pimpl(impl_type::create(std::forward<Functor>(f),
            std::move(target), id, stack_size))
    {
        HPX_ASSERT(m_pimpl->is_ready());
    }

    //coroutine (impl_ptr p)
    //  : m_pimpl(p)
    //{}

    coroutine(coroutine && src)
      : m_pimpl(src->m_pimpl)
    {
      src->m_pimpl = 0;
    }

    coroutine& operator=(coroutine && src)
    {
      coroutine(src).swap(*this);
      return *this;
    }

    coroutine& swap(coroutine& rhs)
    {
      std::swap(m_pimpl, rhs.m_pimpl);
      return *this;
    }

    friend void swap(coroutine& lhs, coroutine& rhs)
    {
      lhs.swap(rhs);
    }

    thread_id_repr_type get_thread_id() const
    {
        return m_pimpl->get_thread_id();
    }

#if defined(HPX_HAVE_THREAD_PHASE_INFORMATION)
    std::size_t get_thread_phase() const
    {
        return m_pimpl->get_thread_phase();
    }
#endif

#if defined(HPX_HAVE_THREAD_LOCAL_STORAGE)
    std::size_t get_thread_data() const
    {
        return m_pimpl.get() ? m_pimpl->get_thread_data() : 0;
    }
    std::size_t set_thread_data(std::size_t data)
    {
        return m_pimpl.get() ? m_pimpl->set_thread_data(data) : 0;
    }
#endif

    template <typename Functor>
    void rebind(Functor && f, naming::id_type && target,
        thread_id_repr_type id = 0)
    {
        HPX_ASSERT(exited());
        impl_type::rebind(m_pimpl.get(), boost::forward<Functor>(f),
            std::move(target), id);
    }

    typedef typename arg_slot_traits::template at<0>::type arg0_type;
    static const int arity = 1;

    struct yield_traits
    {
        typedef typename result_slot_traits::template at<0>::type arg0_type;
        static const int arity = 1;
    };

    HPX_FORCEINLINE result_type operator()(arg0_type arg0 = arg0_type())
    {
      HPX_ASSERT(m_pimpl);
      HPX_ASSERT(m_pimpl->is_ready());

      result_type* ptr = 0;
      m_pimpl->bind_args(&arg0);
      m_pimpl->bind_result_pointer(&ptr);

      m_pimpl->invoke();

      return *m_pimpl->result();
    }

    typedef void(coroutine::*bool_type)();
    operator bool_type() const
    {
      return good() ? &coroutine::bool_type_f : 0;
    }

    bool operator==(const coroutine& rhs) const
    {
      return m_pimpl == rhs.m_pimpl;
    }

    void exit()
    {
      HPX_ASSERT(m_pimpl);
      m_pimpl->exit();
    }

    bool waiting() const
    {
      HPX_ASSERT(m_pimpl);
      return m_pimpl->waiting();
    }

    bool pending() const
    {
      HPX_ASSERT(m_pimpl);
      return m_pimpl->pending();
    }

    bool exited() const
    {
      HPX_ASSERT(m_pimpl);
      return m_pimpl->exited();
    }

    bool is_ready() const
    {
      HPX_ASSERT(m_pimpl);
      return m_pimpl->is_ready();
    }

    bool empty() const
    {
      return m_pimpl == 0;
    }

  protected:
    // The second parameter is used to avoid calling this constructor
    // by mistake from other member functions (specifically operator=).
    //coroutine(impl_type * pimpl, detail::init_from_impl_tag)
    //  : m_pimpl(pimpl)
    //{}

    void bool_type_f() {}

    bool good() const
    {
      return !empty() && !exited() && !waiting();
    }

    impl_ptr m_pimpl;

    std::size_t count() const
    {
      return m_pimpl->count();
    }

    impl_ptr get_impl()
    {
      return m_pimpl;
    }
  };

}}}
#endif
