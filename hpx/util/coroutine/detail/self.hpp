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

#include <hpx/util/assert.hpp>
#include <hpx/util/coroutine/detail/fix_result.hpp>
#include <hpx/util/coroutine/detail/coroutine_accessor.hpp>
#include <hpx/util/function.hpp>

#include <boost/noncopyable.hpp>
#include <utility>

namespace hpx { namespace util { namespace coroutines { namespace detail
{
  template <typename Coroutine>
  class coroutine_self : boost::noncopyable
  {
    // store the current this and write it to the TSS on exit
    struct reset_self_on_exit
    {
        reset_self_on_exit(coroutine_self* self)
          : self_(self)
        {
            impl_type::set_self(self->next_self_);
        }
        ~reset_self_on_exit()
        {
            impl_type::set_self(self_);
        }

        coroutine_self* self_;
    };

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
    typedef typename coroutine_type::thread_id_repr_type thread_id_repr_type;

#if defined(HPX_GENERIC_COROUTINES)

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

#else

    typedef typename yield_traits::arg0_type arg0_type;
    typedef util::function_nonser<yield_result_type(arg0_type)>
        yield_decorator_type;

    yield_result_type yield(arg0_type arg0 = arg0_type())
    {
        return !yield_decorator_.empty() ? yield_decorator_(arg0) : yield_impl(arg0);
    }

    yield_result_type yield_impl(arg0_type arg0)
    {
      HPX_ASSERT(m_pimpl);

      this->m_pimpl->bind_result(&arg0);
      {
        reset_self_on_exit on_exit(this);
        this->m_pimpl->yield();
      }

      return *m_pimpl->args();
    }

    template <typename F>
    yield_decorator_type decorate_yield(F && f)
    {
        yield_decorator_type tmp(std::forward<F>(f));
        std::swap(tmp, yield_decorator_);
        return std::move(tmp);
    }

    yield_decorator_type decorate_yield(yield_decorator_type const& f)
    {
        yield_decorator_type tmp(f);
        std::swap(tmp, yield_decorator_);
        return std::move(tmp);
    }

    yield_decorator_type decorate_yield(yield_decorator_type && f)
    {
        std::swap(f, yield_decorator_);
        return std::move(f);
    }

    yield_decorator_type undecorate_yield()
    {
        yield_decorator_type tmp;
        std::swap(tmp, yield_decorator_);
        return std::move(tmp);
    }

#endif

    HPX_ATTRIBUTE_NORETURN void exit() {
      m_pimpl -> exit_self();
      std::terminate(); // FIXME: replace with hpx::terminate();
    }

    yield_result_type result() {
      return detail::fix_result<
        typename coroutine_type::arg_slot_traits>(*m_pimpl->args());
    }

    bool pending() const
    {
      HPX_ASSERT(m_pimpl);
      return m_pimpl->pending();
    }

    thread_id_repr_type get_thread_id() const
    {
      HPX_ASSERT(m_pimpl);
      return m_pimpl->get_thread_id();
    }

    std::size_t get_thread_phase() const
    {
#if HPX_THREAD_MAINTAIN_PHASE_INFORMATION
      HPX_ASSERT(m_pimpl);
      return m_pimpl->get_thread_phase();
#else
      return 0;
#endif
    }

    explicit coroutine_self(impl_type * pimpl, coroutine_self* next_self = 0)
      : m_pimpl(pimpl), next_self_(next_self)
    {}

#if HPX_THREAD_MAINTAIN_LOCAL_STORAGE
    std::size_t get_thread_data() const
    {
        HPX_ASSERT(m_pimpl);
        return m_pimpl->get_thread_data();
    }
    std::size_t set_thread_data(std::size_t data)
    {
        HPX_ASSERT(m_pimpl);
        return m_pimpl->set_thread_data(data);
    }

    tss_storage* get_thread_tss_data()
    {
        HPX_ASSERT(m_pimpl);
        return m_pimpl->get_thread_tss_data(false);
    }

    tss_storage* get_or_create_thread_tss_data()
    {
        HPX_ASSERT(m_pimpl);
        return m_pimpl->get_thread_tss_data(true);
    }
#endif

#if defined(HPX_GENERIC_COROUTINES)
  private:
    coroutine_self(impl_type * pimpl, detail::init_from_impl_tag)
      : m_pimpl(pimpl), next_self_(0)
    {}

    yield_result_type yield_impl(
        typename coroutine_type::result_slot_type result_)
    {
      HPX_ASSERT(m_pimpl);

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
      HPX_ASSERT(m_pimpl);

      coroutine_accessor::get_impl(target)->bind_args(&args);
      coroutine_accessor::get_impl(target)->bind_result_pointer(m_pimpl->result_pointer());

      {
        reset_self_on_exit on_exit(this);
        this->m_pimpl->yield_to(*coroutine_accessor::get_impl(target));
      }

      typedef typename coroutine_type::arg_slot_traits traits_type;
      return detail::fix_result<traits_type>(*m_pimpl->args());
    }

#else

  private:
    yield_decorator_type yield_decorator_;

#endif

    impl_ptr get_impl()
    {
      return m_pimpl;
    }
    impl_ptr m_pimpl;
    coroutine_self* next_self_;
  };

}}}}

#endif
