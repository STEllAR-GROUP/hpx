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

#ifndef HPX_COROUTINE_DETAIL_COROUTINE_ACCESSOR_HPP_20060709
#define HPX_COROUTINE_DETAIL_COROUTINE_ACCESSOR_HPP_20060709
#include  <boost/utility/in_place_factory.hpp>
namespace hpx { namespace util { namespace coroutines { namespace detail
{
  /*
   * This is a private interface used for coroutine
   * implementation.
   */
  struct init_from_impl_tag{};
  struct coroutine_accessor {

    // Initialize coroutine from implementation type.
    // used by the in_place_assing in place factory.
    template<typename Coroutine, typename Ctx>
    static
    void construct(Ctx * src, void * address) {
      new (address) Coroutine(src, init_from_impl_tag());
    }

    template<typename Coroutine>
    static
    void acquire(Coroutine& x) {
      x.acquire();
    }

    template<typename Coroutine>
    static
    void release(Coroutine& x) {
      x.release();
    }

    template<typename Ctx>
    struct in_place_assign : boost::in_place_factory_base {

      in_place_assign(Ctx * ctx) :
        m_ctx(ctx) {}

      template<typename Coroutine>
      void apply(void * memory) const {
        construct<Coroutine>(m_ctx, memory);
      }
      Ctx * m_ctx;
    };

    template<typename Ctx>
    static
    in_place_assign<Ctx>
    in_place(Ctx * ctx) {
      return in_place_assign<Ctx>(ctx);
    }

    template<typename Coroutine>
    static
    typename Coroutine::impl_ptr
    get_impl(Coroutine& x) {
      return x.get_impl();
    }

    template<typename Coroutine>
    static
    typename Coroutine::impl_type *
    pilfer_impl(Coroutine& x) {
      return x.pilfer_impl();
    }

    template<typename Coroutine>
    static
    std::size_t
    count(const Coroutine& x) {
      return x.count();
    }
  };
}}}}

#endif
