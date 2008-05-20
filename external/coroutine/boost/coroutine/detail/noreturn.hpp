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

#ifndef BOOST_COROUTINE_DETAIL_NORETURN_HPP_20060812
#define BOOST_COROUTINE_DETAIL_NORETURN_HPP_20060812
/*
 * The BOOST_COROUTINE_NORETURN macro provides a way to 
 * tell the compiler that a function will not return through
 * the normal return path (it could return throgh a thrown exception).
 * This not only provieds a possible optimization hint, but also
 * prevents the compiler from complaining if a function that call
 * a noreturn function does not call return itself.
 */
#include <boost/config.hpp>

#if defined(__GNUC__)

#define BOOST_COROUTINE_NORETURN(function) \
    function __attribute__((__noreturn__)) \
/**/
  
#elif defined (BOOST_MSVC)

#define BOOST_COROUTINE_NORETURN(function) \
    __declspec(noreturn) function          \
/**/

#else
//just for testing, remove the following error.
#error no default
#define BOOST_COROUTINE_NORETURN(function) \
    function
/**/

#endif

#endif
