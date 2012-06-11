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

#ifndef BOOST_COROUTINE_COROUTINE_HPP_20060512
#define BOOST_COROUTINE_COROUTINE_HPP_20060512

// This needs to be first for building on Macs
#include <boost/coroutine/detail/default_context_impl.hpp>

#include <cstddef>
#include <boost/preprocessor/repetition.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/type_traits.hpp>
#include <boost/call_traits.hpp>
#include <boost/coroutine/detail/arg_max.hpp>
#include <boost/coroutine/detail/coroutine_impl.hpp>
#include <boost/coroutine/detail/is_callable.hpp>
#include <boost/coroutine/detail/argument_packer.hpp>
#include <boost/coroutine/detail/argument_unpacker.hpp>
#include <boost/coroutine/detail/signature.hpp>
#include <boost/coroutine/detail/index.hpp>
#include <boost/coroutine/detail/coroutine_traits.hpp>
#include <boost/coroutine/detail/coroutine_accessor.hpp>
#include <boost/coroutine/move.hpp>
#include <boost/coroutine/detail/fix_result.hpp>
#include <boost/coroutine/detail/self.hpp>

namespace boost { namespace coroutines {
  namespace detail {
    template<typename T>
    struct optional_result_type :
      boost::mpl::if_<boost::is_same<T, void>,
                      void,
                      boost::optional<T> > { };

    template<typename T>
    BOOST_DEDUCED_TYPENAME
    boost::enable_if<boost::is_same<T, void> >::type
    optional_result() {}

    template<typename T>
    BOOST_DEDUCED_TYPENAME
    boost::disable_if<boost::is_same<T, void>,
                      BOOST_DEDUCED_TYPENAME
                      optional_result_type<T>::type
                      >::type
    optional_result() {
      return BOOST_DEDUCED_TYPENAME
        optional_result_type<T>::type();
    }
  }

  template<typename Signature, template <typename> class Heap, typename Context>
  class coroutine;

//   template<typename Signature, typename Functor, typename Context>
//   class static_coroutine;

  template<typename T>
  struct is_coroutine : boost::mpl::false_{};

  template<typename Signature, template <typename> class Heap, typename Context>
  struct is_coroutine<coroutine<Signature, Heap, Context> >
    : boost::mpl::true_ {};

//   template<typename Sig, typename F, typename Con>
//   struct is_coroutine<static_coroutine<Sig, F, Con> > : boost::mpl::true_{};

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
           template <typename> class Heap = detail::coroutine_allocator,
           typename ContextImpl = detail::default_context_impl>
  class coroutine
    : public movable<coroutine<Signature, Heap, ContextImpl> >
  {
  public:
    typedef coroutine<Signature, Heap, ContextImpl> type;
    typedef ContextImpl context_impl;
    typedef Signature signature_type;
    typedef detail::coroutine_traits<signature_type> traits_type;

    friend struct detail::coroutine_accessor;

    typedef BOOST_DEDUCED_TYPENAME traits_type::result_type result_type;
    typedef BOOST_DEDUCED_TYPENAME traits_type::result_slot_type result_slot_type;
    typedef BOOST_DEDUCED_TYPENAME traits_type::yield_result_type yield_result_type;
    typedef BOOST_DEDUCED_TYPENAME traits_type::result_slot_traits result_slot_traits;
    typedef BOOST_DEDUCED_TYPENAME traits_type::arg_slot_type arg_slot_type;
    typedef BOOST_DEDUCED_TYPENAME traits_type::arg_slot_traits arg_slot_traits;

    typedef detail::coroutine_impl<type, context_impl, Heap> impl_type;
    typedef BOOST_DEDUCED_TYPENAME impl_type::pointer impl_ptr;
    typedef BOOST_DEDUCED_TYPENAME impl_type::thread_id_type thread_id_type;

    typedef detail::coroutine_self<type> self;
    coroutine() : m_pimpl(0) {}

    template<typename Functor>
    coroutine (BOOST_FWD_REF(Functor) f, thread_id_type id = 0,
            std::ptrdiff_t stack_size = detail::default_stack_size)
      : m_pimpl(impl_type::create(boost::forward<Functor>(f), id, stack_size))
    {}

    coroutine (impl_ptr p)
      : m_pimpl(p)
    {}

    coroutine(move_from<coroutine> src)
      : m_pimpl(src->m_pimpl)
    {
      src->m_pimpl = 0;
    }

    coroutine& operator=(move_from<coroutine> src) {
      coroutine(src).swap(*this);
      return *this;
    }

    coroutine& swap(coroutine& rhs) {
      std::swap(m_pimpl, rhs.m_pimpl);
      return *this;
    }

    friend
    void swap(coroutine& lhs, coroutine& rhs) {
      lhs.swap(rhs);
    }

    thread_id_type get_thread_id() const
    {
        return m_pimpl->get_thread_id();
    }

    std::size_t get_thread_phase() const
    {
        return m_pimpl->get_thread_phase();
    }

    template <typename Functor>
    void rebind(Functor f, thread_id_type id = 0)
    {
        BOOST_ASSERT(exited());
        impl_type::rebind(m_pimpl, f, id);
    }

    void reset()
    {
        BOOST_ASSERT(exited());
        m_pimpl->reset();
    }

#   define BOOST_COROUTINE_generate_argument_n_type(z, n, traits_type) \
    typedef BOOST_DEDUCED_TYPENAME traits_type ::template at<n>::type  \
    BOOST_PP_CAT(BOOST_PP_CAT(arg, n), _type);                         \
    /**/

    BOOST_PP_REPEAT(BOOST_COROUTINE_ARG_MAX,
                    BOOST_COROUTINE_generate_argument_n_type,
                    arg_slot_traits)

    static const int arity = arg_slot_traits::length;

    struct yield_traits {
      BOOST_PP_REPEAT(BOOST_COROUTINE_ARG_MAX,
                      BOOST_COROUTINE_generate_argument_n_type,
                      result_slot_traits)
      static const int arity = result_slot_traits::length;
    };

#   undef BOOST_COROUTINE_generate_argument_n_type

#   define BOOST_COROUTINE_param_with_default(z, n, type_prefix)    \
    BOOST_DEDUCED_TYPENAME call_traits                              \
    <BOOST_PP_CAT(BOOST_PP_CAT(type_prefix, n), _type)>::param_type \
    BOOST_PP_CAT(arg, n) =                                          \
    BOOST_PP_CAT(BOOST_PP_CAT(type_prefix, n), _type)()             \
    /**/

    result_type operator()
      (BOOST_PP_ENUM
       (BOOST_COROUTINE_ARG_MAX,
        BOOST_COROUTINE_param_with_default,
        arg)) {
      return call_impl
        (arg_slot_type(BOOST_PP_ENUM_PARAMS
          (BOOST_COROUTINE_ARG_MAX,
           arg)));
    }

    BOOST_DEDUCED_TYPENAME
    detail::optional_result_type<result_type>::type
    operator()
      (const std::nothrow_t&
       BOOST_PP_ENUM_TRAILING
       (BOOST_COROUTINE_ARG_MAX,
        BOOST_COROUTINE_param_with_default,
        arg)) {
      return call_impl_nothrow
        (arg_slot_type(BOOST_PP_ENUM_PARAMS
          (BOOST_COROUTINE_ARG_MAX,
           arg)));
    }

#   undef BOOST_COROUTINE_param_typedef
#   undef BOOST_COROUTINE_param_with_default

    typedef void(coroutine::*bool_type)();
    operator bool_type() const {
      return good()? &coroutine::bool_type_f: 0;
    }

    bool operator==(const coroutine& rhs) {
      return m_pimpl == rhs.m_pimpl;
    }

    void exit() {
      BOOST_ASSERT(m_pimpl);
      m_pimpl->exit();
    }

    bool waiting() const {
      BOOST_ASSERT(m_pimpl);
      return m_pimpl->waiting();
    }

    bool pending() const {
      BOOST_ASSERT(m_pimpl);
      return m_pimpl->pending();
    }

    bool exited() const {
      BOOST_ASSERT(m_pimpl);
      return m_pimpl->exited();
    }

    bool empty() const {
      return m_pimpl == 0;
    }
  protected:

    // The second parameter is used to avoid calling this constructor
    // by mistake from other member functions (specifically operator=).
    coroutine(impl_type * pimpl, detail::init_from_impl_tag) :
      m_pimpl(pimpl) {}

    void bool_type_f() {}

    bool good() const  {
      return !empty() && !exited() && !waiting();
    }

    result_type call_impl(arg_slot_type args) {
      BOOST_ASSERT(m_pimpl);
      m_pimpl->bind_args(&args);
      result_slot_type * ptr;
      m_pimpl->bind_result_pointer(&ptr);
      m_pimpl->invoke();

      return detail::fix_result<result_slot_traits>(*m_pimpl->result());
    }

    BOOST_DEDUCED_TYPENAME
    detail::optional_result_type<result_type>::type
    call_impl_nothrow(arg_slot_type args) {
      BOOST_ASSERT(m_pimpl);
      m_pimpl->bind_args(&args);
      result_slot_type * ptr;
      m_pimpl->bind_result_pointer(&ptr);
      if(!m_pimpl->wake_up())
        return detail::optional_result<result_type>();

      return detail::fix_result<result_slot_traits>(*m_pimpl->result());
    }

    impl_ptr m_pimpl;

    void acquire() {
      m_pimpl->acquire();
    }

    void release() {
      m_pimpl->release();
    }

    std::size_t
    count() const {
      return m_pimpl->count();
    }

    impl_ptr get_impl() {
      return m_pimpl;
    }
  };

  /////////////////////////////////////////////////////////////////////////////
  // essentially this is the same as above except it doesn't allocate the
  // coroutine implementation but includes it as a member
//   template<
//       typename Signature, typename Functor,
//       typename ContextImpl = detail::default_context_impl
//   >
//   class static_coroutine
//   {
//   public:
//     typedef static_coroutine type;
//     typedef ContextImpl context_impl;
//     typedef Signature signature_type;
//     friend struct detail::coroutine_accessor;
//
//     typedef BOOST_DEDUCED_TYPENAME
//         detail::coroutine_traits<signature_type>::result_type
//     result_type;
//
//     typedef BOOST_DEDUCED_TYPENAME
//         detail::coroutine_traits<signature_type>::result_slot_type
//     result_slot_type;
//
//     typedef BOOST_DEDUCED_TYPENAME
//         detail::coroutine_traits<signature_type>::yield_result_type
//     yield_result_type;
//
//     typedef BOOST_DEDUCED_TYPENAME
//         detail::coroutine_traits<signature_type>::result_slot_traits
//     result_slot_traits;
//
//     typedef BOOST_DEDUCED_TYPENAME
//         detail::coroutine_traits<signature_type>::arg_slot_type
//     arg_slot_type;
//
//     typedef BOOST_DEDUCED_TYPENAME
//         detail::coroutine_traits<signature_type>::arg_slot_traits
//     arg_slot_traits;
//
//     typedef detail::coroutine_impl_wrapper<Functor, type, context_impl> impl_type;
//     typedef typename impl_type::thread_id_type thread_id_type;
//     typedef detail::coroutine_self<type> self;
//
//     static_coroutine (Functor f, thread_id_type id = 0,
//             std::ptrdiff_t stack_size = detail::default_stack_size)
//       : impl_(f, id, stack_size)
//     {
//         impl_.acquire();    // make sure refcount is not zero
//     }
//
//     thread_id_type get_thread_id() const
//     {
//         return impl_.get_thread_id();
//     }
//
//     std::size_t get_thread_phase() const
//     {
//         return impl_.get_thread_phase();
//     }
//
//     void rebind(Functor f, thread_id_type id = 0)
//     {
//         BOOST_ASSERT(exited());
//         impl_type::rebind(f, id);
//     }
//
#if 0
 #define BOOST_COROUTINE_generate_argument_n_type(z, n, traits_type)           \
     typedef BOOST_DEDUCED_TYPENAME traits_type ::template at<n>::type         \
     BOOST_PP_CAT(BOOST_PP_CAT(arg, n), _type);                                \
     /**/
#endif
//
//     BOOST_PP_REPEAT(BOOST_COROUTINE_ARG_MAX,
//                     BOOST_COROUTINE_generate_argument_n_type,
//                     arg_slot_traits);
//
//     static const int arity = arg_slot_traits::length;
//
//     struct yield_traits {
//       BOOST_PP_REPEAT(BOOST_COROUTINE_ARG_MAX,
//                       BOOST_COROUTINE_generate_argument_n_type,
//                       result_slot_traits);
//       static const int arity = result_slot_traits::length;
//     };
// #undef BOOST_COROUTINE_generate_argument_n_type
#if 0
 #define BOOST_COROUTINE_param_with_default(z, n, type_prefix)                 \
     BOOST_DEDUCED_TYPENAME                                                    \
     call_traits<BOOST_PP_CAT(BOOST_PP_CAT(type_prefix, n), _type)>::param_type\
     BOOST_PP_CAT(arg, n) =                                                    \
         BOOST_PP_CAT(BOOST_PP_CAT(type_prefix, n), _type)()                   \
     /**/
#endif
//     result_type operator()(
//         BOOST_PP_ENUM(BOOST_COROUTINE_ARG_MAX, BOOST_COROUTINE_param_with_default, arg))
//     {
//       return call_impl(arg_slot_type(
//               BOOST_PP_ENUM_PARAMS(BOOST_COROUTINE_ARG_MAX, arg)
//           ));
//     }
//
//     BOOST_DEDUCED_TYPENAME
//     detail::optional_result_type<result_type>::type
//     operator() (std::nothrow_t const&
//         BOOST_PP_ENUM_TRAILING(BOOST_COROUTINE_ARG_MAX, BOOST_COROUTINE_param_with_default, arg))
//     {
//       return call_impl_nothrow(arg_slot_type(
//               BOOST_PP_ENUM_PARAMS(BOOST_COROUTINE_ARG_MAX, arg)
//           ));
//     }
// #undef BOOST_COROUTINE_param_with_default
//
//     typedef void(static_coroutine::*bool_type)();
//     operator bool_type() const {
//         return good()? &static_coroutine::bool_type_f: 0;
//     }
//
//     void exit() {
//         impl_.exit();
//     }
//
//     bool waiting() const {
//         return impl_.waiting();
//     }
//
//     bool pending() const {
//         return impl_.pending();
//     }
//
//     bool exited() const {
//         return impl_.exited();
//     }
//
//     bool empty() const {
//       return false;
//     }
//
//   protected:
//     void bool_type_f() {}
//
//     bool good() const  {
//         return !empty() && !exited() && !waiting();
//     }
//
//     result_type call_impl(arg_slot_type args)
//     {
//         impl_.bind_args(&args);
//         result_slot_type* ptr;
//         impl_.bind_result_pointer(&ptr);
//
//         impl_.invoke();
//
//         return detail::fix_result<result_slot_traits>(*impl_.result());
//     }
//
//     BOOST_DEDUCED_TYPENAME detail::optional_result_type<result_type>::type
//     call_impl_nothrow(arg_slot_type args)
//     {
//         impl_.bind_args(&args);
//         result_slot_type * ptr;
//         impl_.bind_result_pointer(&ptr);
//         if(!impl_.wake_up())
//             return detail::optional_result<result_type>();
//
//         return detail::fix_result<result_slot_traits>(*impl_.result());
//     }
//
//     impl_type impl_;      // coroutine implementation type
//
//     std::size_t
//     count() const {
//         return impl_.count();
//     }
//   };

} }
#endif
