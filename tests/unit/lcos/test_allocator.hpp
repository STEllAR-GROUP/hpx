//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (C) 2011 Vicente J. Botet Escriba
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TEST_ALLOCATOR_HPP)
#define HPX_TEST_ALLOCATOR_HPP

#include <hpx/hpx_main.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

struct test_alloc_base
{
    static int count;
    static int throw_after;
};

int test_alloc_base::count = 0;
int test_alloc_base::throw_after = INT_MAX;

template <typename T>
class test_allocator : public test_alloc_base
{
    int data_;

    template <typename U> friend class test_allocator;

public:
    typedef std::size_t size_type;
    typedef std::int64_t difference_type;
    typedef T value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef typename std::add_lvalue_reference<value_type>::type reference;
    typedef typename std::add_lvalue_reference<value_type const>::type
        const_reference;

    template <typename U>
    struct rebind
    {
        typedef test_allocator<U> other;
    };

    test_allocator() HPX_NOEXCEPT
      : data_(-1)
    {}

    explicit test_allocator(int i) HPX_NOEXCEPT
      : data_(i)
    {}

    test_allocator(test_allocator const& a) HPX_NOEXCEPT
      : data_(a.data_)
    {}

    template <typename U>
    test_allocator(test_allocator<U> const& a) HPX_NOEXCEPT
      : data_(a.data_)
    {}

    ~test_allocator() HPX_NOEXCEPT
    {
        data_ = 0;
    }

    pointer address(reference x) const { return &x; }
    const_pointer address(const_reference x) const { return &x; }

    pointer allocate(size_type n, const void* = 0)
    {
        if (count >= throw_after)
            throw std::bad_alloc();
        ++count;
        return static_cast<pointer>(std::malloc(n * sizeof(T)));
    }

    void deallocate(pointer p, size_type)
    {
        --count;
        std::free(p);
    }

    size_type max_size() const HPX_NOEXCEPT
    {
        return UINT_MAX / sizeof(T);
    }

    template <typename U, typename ... Ts>
    void construct(U* p, Ts && ... ts)
    {
        ::new((void*)p) T(std::forward<Ts>(ts)...);
    }

    void destroy(pointer p)
    {
        p->~T();
    }

    friend bool operator==(test_allocator const& x, test_allocator const& y)
    {
        return x.data_ == y.data_;
    }
    friend bool operator!=(test_allocator const& x, test_allocator const& y)
    {
        return !(x == y);
    }
};

template <>
class test_allocator<void> : public test_alloc_base
{
    int data_;

    template <typename U> friend class test_allocator;

public:
    typedef std::size_t size_type;
    typedef std::int64_t difference_type;
    typedef void value_type;
    typedef value_type* pointer;
    typedef value_type const* const_pointer;

    template <typename U>
    struct rebind
    {
        typedef test_allocator<U> other;
    };

    test_allocator() HPX_NOEXCEPT
      : data_(-1)
    {}

    explicit test_allocator(int i) HPX_NOEXCEPT
      : data_(i)
    {}

    test_allocator(test_allocator const& a) HPX_NOEXCEPT
      : data_(a.data_)
    {}

    template <typename U>
    test_allocator(test_allocator<U> const& a) HPX_NOEXCEPT
      : data_(a.data_)
    {}

    ~test_allocator() HPX_NOEXCEPT
    {
        data_ = 0;
    }

    friend bool operator==(test_allocator const& x, test_allocator const& y)
    {
        return x.data_ == y.data_;
    }
    friend bool operator!=(test_allocator const& x, test_allocator const& y)
    {
        return !(x == y);
    }
};

#endif
