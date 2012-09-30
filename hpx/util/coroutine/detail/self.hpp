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

#ifndef HPX_COROUTINE_DETAIL_SELF_HPP_20060809
#define HPX_COROUTINE_DETAIL_SELF_HPP_20060809

#include <boost/noncopyable.hpp>
#include <boost/assert.hpp>
#include <hpx/util/coroutine/detail/fix_result.hpp>
#include <hpx/util/coroutine/detail/coroutine_accessor.hpp>

namespace hpx { namespace util { namespace coroutines { namespace detail 
{
  template <typename Coroutine>
  class coroutine_self : boost::noncopyable 
  {
  public:
    typedef Coroutine coroutine_type;
    typedef coroutine_self<coroutine_type> type;
    friend struct detail::coroutine_accessor;

    typedef typename coroutine_type::impl_type impl_type;

    // Note, no reference counting here.
    typedef impl_type* impl_ptr;

    typedef typename coroutine_type::result_type result_type;
    typedef typename coroutine_type::result_slot_type result_slot_type;
    typedef typename coroutine_type::yield_result_type yield_result_type;
    typedef typename coroutine_type::result_slot_traits result_slot_traits;
    typedef typename coroutine_type::arg_slot_type arg_slot_type;
    typedef typename coroutine_type::arg_slot_traits arg_slot_traits;
    typedef typename coroutine_type::yield_traits yield_traits;
    typedef typename coroutine_type::thread_id_type thread_id_type;

#define HPX_COROUTINE_PARAM_WITH_DEFAULT(z, n, type_prefix)                   \
    typename boost::call_traits<                                              \
        BOOST_PP_CAT(BOOST_PP_CAT(type_prefix, n), _type)>::param_type        \
            BOOST_PP_CAT(arg, n) =                                            \
                BOOST_PP_CAT(BOOST_PP_CAT(type_prefix, n), _type)()           \
/**/

    yield_result_type yield(BOOST_PP_ENUM(HPX_COROUTINE_ARG_MAX, 
        HPX_COROUTINE_PARAM_WITH_DEFAULT, typename yield_traits::arg))
    {
        return yield_impl(typename coroutine_type::result_slot_type(
            BOOST_PP_ENUM_PARAMS(HPX_COROUTINE_ARG_MAX, arg)));
    }

    template <typename Target>
    yield_result_type yield_to(Target& target
        BOOST_PP_ENUM_TRAILING(HPX_COROUTINE_ARG_MAX, 
            HPX_COROUTINE_PARAM_WITH_DEFAULT, typename Target::arg))
    {
        typedef typename Target::arg_slot_type slot_type;
        return yield_to_impl(target, slot_type(
            BOOST_PP_ENUM_PARAMS(HPX_COROUTINE_ARG_MAX, arg)));
    }

#undef  HPX_COROUTINE_PARAM_WITH_DEFAULT

    BOOST_ATTRIBUTE_NORETURN void exit() {
      m_pimpl -> exit_self();
      std::terminate(); // FIXME: replace with hpx::terminate();
    }

    yield_result_type result() {
      return detail::fix_result<
        typename coroutine_type::arg_slot_traits>(*m_pimpl->args());
    }

    bool pending() const {
      BOOST_ASSERT(m_pimpl);
      return m_pimpl->pending();
    }

    thread_id_type get_thread_id() const {
      BOOST_ASSERT(m_pimpl);
      return m_pimpl->get_thread_id();
    }

    std::size_t get_thread_phase() const {
      BOOST_ASSERT(m_pimpl);
      return m_pimpl->get_thread_phase();
    }

    explicit coroutine_self(impl_type * pimpl)
      : m_pimpl(pimpl)
    {}

  private:
    coroutine_self(impl_type * pimpl, detail::init_from_impl_tag) 
      : m_pimpl(pimpl) 
    {}

    // store the current this and write it to the TSS on exit
    struct reset_self_on_exit
    {
        reset_self_on_exit(coroutine_self* self)
          : self_(self)
        {
            impl_type::set_self(NULL);
        }
        ~reset_self_on_exit()
        {
            impl_type::set_self(self_);
        }

        coroutine_self* self_;
    };

    yield_result_type yield_impl(
        typename coroutine_type::result_slot_type result_)
    {
      typedef typename coroutine_type::result_slot_type slot_type;

      BOOST_ASSERT(m_pimpl);

      this->m_pimpl->bind_result(&result_);
      {
        reset_self_on_exit on_exit(this);
        this->m_pimpl->yield();
      }

      typedef typename coroutine_type::arg_slot_traits traits_type;
      return detail::fix_result<traits_type>(*m_pimpl->args());
    }

    template <typename TargetCoroutine>
    yield_result_type yield_to_impl(TargetCoroutine& target,
        typename TargetCoroutine::arg_slot_type args)
    {
      BOOST_ASSERT(m_pimpl);

      coroutine_accessor::get_impl(target)->bind_args(&args);
      coroutine_accessor::get_impl(target)->bind_result_pointer(m_pimpl->result_pointer());

      {
        reset_self_on_exit on_exit(this);
        this->m_pimpl->yield_to(*coroutine_accessor::get_impl(target));
      }

      typedef typename coroutine_type::arg_slot_traits traits_type;
      return detail::fix_result<traits_type>(*m_pimpl->args());
    }

    impl_ptr get_impl() {
      return m_pimpl;
    }
    impl_ptr m_pimpl;
  };

}}}}

#endif
