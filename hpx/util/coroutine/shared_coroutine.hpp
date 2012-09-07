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

#ifndef HPX_COROUTINE_SHARED_COROUTINE_HPP_20060812
#define HPX_COROUTINE_SHARED_COROUTINE_HPP_20060812
#include <hpx/util/coroutine/coroutine.hpp>
namespace hpx { namespace util { namespace coroutines 
{
  // This class is a workaround for the widespread lack of move
  // semantics support. It is a reference counted wrapper around
  // the coroutine object.
  // FIXME: ATM a shared_coroutine is-a coroutine. This is to avoid
  // inheriting privately and cluttering the code with lots of using
  // declarations to unhide coroutine members and nested types.
  // From a purity point of view, coroutines and shared_coroutines should
  // be two different types.
  template<typename Signature, typename ContextImpl = detail::default_context_impl>
  class shared_coroutine : public coroutine<Signature, ContextImpl> {
  public:
    typedef coroutine<Signature, ContextImpl> coroutine_type;
    typedef typename coroutine_type::thread_id_type thread_id_type;

    shared_coroutine() {}

    template<typename Functor>
    shared_coroutine(Functor f, thread_id_type id  = 0,
            std::ptrdiff_t stack_size = detail::default_stack_size)
      : coroutine_type(f, id, stack_size)
    {}

    shared_coroutine(move_from<coroutine_type> src):
      coroutine_type(src) {}

    shared_coroutine(const shared_coroutine& rhs) :
      coroutine_type(rhs.m_pimpl.get(), detail::init_from_impl_tag()) {}

    shared_coroutine& operator=(move_from<coroutine_type> src) {
      shared_coroutine(src).swap(*this);
      return *this;
    }

    shared_coroutine& operator=(const shared_coroutine& rhs) {
      shared_coroutine(rhs).swap(*this);
      return *this;
    }
  private:
  };
}}}

#endif
