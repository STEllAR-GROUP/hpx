//  Copyright (c) 2006, Giovanni P. Deretta
//  Copyright (c) 2011, Bryce Adelstein-Lelbach
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

#ifndef HPX_COROUTINE_DETAIL_POSIX_UTILITY_HPP_02012006
#define HPX_COROUTINE_DETAIL_POSIX_UTILITY_HPP_02012006
#include <boost/config.hpp>

#if defined(_POSIX_VERSION)
/**
 * Most of these utilities are really pure C++, but they are useful
 * only on posix systems.
 */
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <new>
#include <iostream>
#include <boost/type_traits.hpp>
#include <boost/assert.hpp>

#if defined(_POSIX_MAPPED_FILES) && _POSIX_MAPPED_FILES > 0
#include <sys/mman.h>
#include <sys/param.h>
#include <errno.h>
#endif

/**
 * Stack allocation routines and trampolines for setcontext
 */
namespace hpx { namespace util { namespace coroutines { namespace detail { namespace posix {

HPX_EXPORT extern bool use_guard_pages;

#if defined(_POSIX_MAPPED_FILES) && _POSIX_MAPPED_FILES > 0

  inline
  void*
  alloc_stack(std::size_t size) {
    void* real_stack = ::mmap(NULL,
                              size + EXEC_PAGESIZE,
                              PROT_EXEC|PROT_READ|PROT_WRITE,
                              MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,
                              -1,
                              0
                              );

    if (real_stack == MAP_FAILED) {
      if (ENOMEM == errno)
        throw std::runtime_error("mmap() failed to allocate thread stack due "
                                 "to insufficient resources, "
                                 "increase /proc/sys/vm/max_map_count or add "
                                 "-Ihpx.use_guard_pages=0 to the command line");
      else
        throw std::runtime_error("mmap() failed to allocate thread stack");
    }

#if HPX_THREAD_GUARD_PAGE
    if (use_guard_pages) {
        // Add a guard page.
        ::mprotect(real_stack, EXEC_PAGESIZE, PROT_NONE);

        void** stack = static_cast<void**>(real_stack) + (EXEC_PAGESIZE / sizeof(void*));
        return static_cast<void*>(stack);
    }
    return real_stack;
#else
    return real_stack;
#endif
  }

  inline
  void watermark_stack(void* stack, std::size_t size) {
    BOOST_ASSERT(size > EXEC_PAGESIZE);

    // Fill the bottom 8 bytes of the first page with 1s.
    void** watermark = static_cast<void**>(stack) + ((size - EXEC_PAGESIZE) / sizeof(void*));
    *watermark = reinterpret_cast<void*>(0xDEADBEEFDEADBEEFull);
  }

  inline
  bool reset_stack(void* stack, std::size_t size) {
    void** watermark = static_cast<void**>(stack) + ((size - EXEC_PAGESIZE) / sizeof(void*));

    // If the watermark has been overwritten, then we've gone past the first
    // page.
    if((reinterpret_cast<void*>(0xDEADBEEFDEADBEEFull)) != *watermark)
    {
      // We never free up the first page, as it's initialized only when the
      // stack is created.
      ::madvise(stack, size - EXEC_PAGESIZE, MADV_DONTNEED);
      return true;
    }

    return false;
  }

  inline
  void free_stack(void* stack, std::size_t size) {
#if HPX_THREAD_GUARD_PAGE
    if (use_guard_pages) {
        void** real_stack = static_cast<void**>(stack) - (EXEC_PAGESIZE / sizeof(void*));
        ::munmap(static_cast<void*>(real_stack), size + EXEC_PAGESIZE);
    }
    else {
        ::munmap(stack, size);
    }
#else
    ::munmap(stack, size);
#endif
  }

#else  // non-mmap()

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
  void watermark_stack(void* stack, std::size_t size) {
    // no-op
  }

  inline
  bool reset_stack(void* stack, std::size_t size) {
    return false;
  }

  inline
  void free_stack(void* stack, std::size_t size) {
    delete [] static_cast<stack_aligner*>(stack);
  }

#endif  // non-mmap() implementation of alloc_stack()/free_stack()

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
    splitter(int first_, int second_) {
      int_[0] = first_;
      int_[1] = second_;
    }

    int first() {
      return int_[0];
    }

    int second() {
      return int_[1];
    }

    splitter(T* ptr_) :ptr(ptr_) {}

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

}}}}

//#if defined(_POSIX_MAPPED_FILES) && _POSIX_MAPPED_FILES > 0
//#include <sys/mman.h>
//
//#if defined(MAP_ANONYMOUS)
//# define HPX_MAP_ANONYMOUS MAP_ANONYMOUS
//#elif defined(MAP_ANON)
//# define HPX_MAP_ANONYMOUS MAP_ANON
//#else
//# error "Anonymous mmap not available on this platform!"
//#endif
//
//namespace hpx { namespace util { namespace coroutines { namespace detail { namespace posix {
//  inline
//  void *
//  alloc_stack_mmap(std::size_t size) {
//    void * stack = ::mmap(NULL,
//                          size,
//                          PROT_EXEC|PROT_READ|PROT_WRITE,
//                          MAP_PRIVATE|HPX_MAP_ANONYMOUS,
//                          -1,
//                          0
//                          );
//    if(stack == MAP_FAILED) {
//      std::cerr <<strerror(errno)<<"\n";
//      abort();
//    }
//    return stack;
//  }
//
//  inline
//  void free_stack_mmap(void* stack, std::size_t size) {
//    ::munmap(stack, size);
//  }
//} } } } }
//#endif
#else
#error This header can only be included when compiling for posix systems.
#endif

#endif
