//  Copyright (c) 2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This is partially taken from: http://www.garret.ru/threadalloc/readme.html

#if !defined(HPX_UTIL_THREAD_ALLOCATOR_AUG_08_2018_1047AM)
#define HPX_UTIL_THREAD_ALLOCATOR_AUG_08_2018_1047AM

#include <hpx/config.hpp>

#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT void* thread_alloc(std::size_t);
    HPX_EXPORT void thread_free(void*);

    ///////////////////////////////////////////////////////////////////////////
    template <typename T = int>
    struct thread_allocator
    {
        typedef T value_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef T const& const_reference;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;

        template <typename U>
        struct rebind
        {
            typedef thread_allocator<U> other;
        };

        typedef std::true_type is_always_equal;
        typedef std::true_type propagate_on_container_move_assignment;

        thread_allocator() = default;

        template <typename U>
        explicit thread_allocator(thread_allocator<U> const&)
        {
        }

        pointer address(reference x) const noexcept
        {
            return &x;
        }

        const_pointer address(const_reference x) const noexcept
        {
            return &x;
        }

        pointer allocate(size_type n,
            std::allocator<void>::const_pointer hint = nullptr)
        {
            return reinterpret_cast<pointer>(thread_alloc(n * sizeof(T)));
        }

        void deallocate(pointer p, size_type n)
        {
            thread_free(p);
        }

        size_type max_size() const noexcept
        {
            return (std::numeric_limits<size_type>::max)() / sizeof(T);
        }

        template <typename U, typename ... Args>
        void construct(U* p, Args &&... args)
        {
            ::new((void *)p) U(std::forward<Args>(args)...);
        }

        template <typename U>
        void destroy(U* p)
        {
            p->~U();
        }
    };

    template <typename T>
    HPX_CONSTEXPR
    bool operator==(thread_allocator<T> const&, thread_allocator<T> const&)
    {
        return true;
    }

    template <typename T>
    HPX_CONSTEXPR
    bool operator!=(thread_allocator<T> const&, thread_allocator<T> const&)
    {
        return false;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Allocator>
    struct allocator_deleter
    {
        template <typename SharedState>
        void operator()(SharedState* state)
        {
            using traits = std::allocator_traits<Allocator>;
            traits::deallocate(alloc_, state, 1);
        }

        Allocator alloc_;
    };
}}

#include <hpx/config/warnings_suffix.hpp>

#endif


