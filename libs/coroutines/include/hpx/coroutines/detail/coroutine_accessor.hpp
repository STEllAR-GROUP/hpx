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
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_COROUTINES_DETAIL_COROUTINE_ACCESSOR_HPP
#define HPX_RUNTIME_THREADS_COROUTINES_DETAIL_COROUTINE_ACCESSOR_HPP

#include <utility>

namespace hpx { namespace threads { namespace coroutines { namespace detail {

    struct coroutine_accessor
    {
        template <typename Coroutine>
        HPX_FORCEINLINE static typename Coroutine::impl_ptr get_impl(
            Coroutine& x)
        {
            return x.get_impl();
        }

        template <typename Coroutine>
        HPX_FORCEINLINE static void set_impl(
            Coroutine& x, typename Coroutine::impl_ptr coro)
        {
            return x.set_impl(coro);
        }

        template <typename Coroutine>
        HPX_FORCEINLINE static typename Coroutine::impl_ptr get_impl_next(
            Coroutine& x)
        {
            return x.get_impl_next();
        }

        template <typename Coroutine>
        HPX_FORCEINLINE static void set_impl_next(
            Coroutine& x, typename Coroutine::impl_ptr coro)
        {
            return x.set_impl_next(coro);
        }

//         template <typename Coroutine>
//         HPX_FORCEINLINE static std::pair<typename Coroutine::impl_ptr,
//             typename Coroutine::impl_ptr>
//         get_impls(Coroutine const& x) noexcept
//         {
//             return x.get_impls();
//         }
//
//         template <typename Coroutine>
//         HPX_FORCEINLINE static void set_impls(Coroutine& x,
//             typename Coroutine::impl_ptr coro,
//             typename Coroutine::impl_ptr next) noexcept
//         {
//             x.set_impls(coro, next);
//         }
    };
}}}}    // namespace hpx::threads::coroutines::detail

#endif /*HPX_RUNTIME_THREADS_COROUTINES_DETAIL_COROUTINE_ACCESSOR_HPP*/
