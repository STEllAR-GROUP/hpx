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

#ifndef HPX_RUNTIME_THREADS_COROUTINES_EXCEPTION_HPP
#define HPX_RUNTIME_THREADS_COROUTINES_EXCEPTION_HPP

#include <hpx/config.hpp>

#include <exception>
#include <typeinfo>

namespace hpx { namespace threads { namespace coroutines
{
    // All coroutine exceptions are derived from this base.
    class exception_base : public std::exception {};

    // This exception is thrown when a coroutine is requested
    // to exit.
    class exit_exception : public exception_base {};

    // This exception is thrown on a coroutine invocation
    // if a coroutine exits without
    // returning a result. Note that calling invoke()
    // on an already exited coroutine is undefined behavior.
    class coroutine_exited : public exception_base {};

    // This exception is thrown on a coroutine invocation
    // if a coroutine enter the wait state without
    // returning a result. Note that calling invoke()
    // on a waiting coroutine is undefined behavior.
    class waiting : public exception_base {};

    class unknown_exception_tag {};

    // This exception is thrown on a coroutine invocation
    // if the coroutine is exited by an un-catched exception
    // (not derived from exit_exception). abnormal_exit::type()
    // returns the typeid of that exception if it is derived
    // from std::exception, else returns typeid(unknonw_exception_tag)
    class abnormal_exit : public std::exception
    {
    public:
        abnormal_exit(std::type_info const& e) : m_e(e) {};

        char const* what() const throw()
        {
            return m_e == typeid(unknown_exception_tag) ?
                "unknown exception" : m_e.name();
        }

        std::type_info const& type() const throw()
        {
            return m_e;
        }

    private:
        std::type_info const& m_e;
    };

    class null_thread_id_exception : public exception_base {};
}}}

#endif /*HPX_RUNTIME_THREADS_COROUTINES_EXCEPTION_HPP*/
