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

#ifndef BOOST_COROUTINE_DETAIL_POSIX_UTILITY_HPP_02012006
#define BOOST_COROUTINE_DETAIL_POSIX_UTILITY_HPP_02012006
#include <boost/config.hpp>

#if defined(_POSIX_VERSION)
/**
 * Most of these utilities are really pure C++, but they are useful
 * only on posix systems.
 */
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <boost/type_traits.hpp>
/**
 * Stack allocation routines and trampolines for setcontext
 */
namespace boost { namespace coroutines { namespace detail { namespace posix {

  
  //this should be a fine default.
  static const std::size_t stack_alignment = sizeof(void*) > 16? sizeof(void*): 16;

  struct stack_aligner {
    boost::type_with_alignment<stack_alignment>::type dummy;
  };

  /**
   * Stack allocator and deleter functions.
   * Better implementations are possible using
   * mmap (might be required on some systems) and/or
   * using a pooling allocator.
   * NOTE: the SuSv3 documentation explicitly allows
   * the use of malloc to allocate stacks for makectx.
   * We use new/delete for guaranteed alignment.
   */
  inline
  void* alloc_stack(std::size_t size) {
    return new stack_aligner[size/sizeof(stack_aligner)];
  }

  inline
  void free_stack(void* stack, std::size_t size) {
    delete [] static_cast<stack_aligner*>(stack);
  }

  /**
   * The splitter is needed for 64 bit systems. 
   * @note The current implementation does NOT use
   * (for debug reasons).
   * Thus it is not 64 bit clean.
   * Use it for 64 bits systems.
   */
  template<typename T>
  union splitter {
    int int_[2];
    T* ptr;
    splitter(int first, int second) {
      int_[0] = first;
      int_[1] = second;
    }

    int first() {
      return int_[0];
    }

    int second() {
      return int_[1];
    }

    splitter(T* ptr) :ptr(ptr) {}

    void operator()() {
      (*ptr)();
    }
  };

  template<typename T>
  inline
  void
  trampoline_split(int first, int second) {
    splitter<T> split(first, second);
    split();
  }

  template<typename T>
  inline
  void
  trampoline(T * fun) {
    (*fun)();
  }
}
} } }

#if defined(_POSIX_MAPPED_FILES) && _POSIX_MAPPED_FILES > 0
#include <sys/mman.h>
namespace boost { namespace coroutines { namespace detail { namespace posix {
  inline 
  void * 
  alloc_stack_mmap(std::size_t size) {
    void * stack = ::mmap(NULL,
			  size,
			  PROT_EXEC|PROT_READ|PROT_WRITE,
			  MAP_PRIVATE|MAP_ANONYMOUS,
			  -1,
			  0
			  );
    if(stack == MAP_FAILED) {
      std::cerr <<strerror(errno)<<"\n";
      abort();
    }
    return stack;
  }

  inline
  void free_stack_mmap(void* stack, std::size_t size) {
    ::munmap(stack, size);
  }
} } } }
#endif
#else
#error This header can only be included when compiling for posix systems.
#endif

#endif
