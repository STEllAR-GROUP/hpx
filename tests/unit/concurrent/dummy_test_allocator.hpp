//  Copyright (c) 2016 John Biddiscombe
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// (C) Copyright Ion Gaztanaga 2004-2013.

#ifndef HPX_CONTAINER_DUMMY_TEST_ALLOCATOR_HPP
#define HPX_CONTAINER_DUMMY_TEST_ALLOCATOR_HPP

#include <hpx/config.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <type_traits>

namespace test {

// Very simple version 1 allocator
template <class T>
class simple_allocator
{
public:
    typedef T value_type;

    simple_allocator()
    {
    }

    template <class U>
    simple_allocator(const simple_allocator<U> &)
    {
    }

    T *allocate(std::size_t n)
    {
        return (T *) ::new char[sizeof(T) * n];
    }

    void deallocate(T *p, std::size_t)
    {
        delete[]((char *) p);
    }

    friend bool operator==(const simple_allocator &, const simple_allocator &)
    {
        return true;
    }

    friend bool operator!=(const simple_allocator &, const simple_allocator &)
    {
        return false;
    }
};

// Version 2 allocator with rebind
template <class T>
class dummy_test_allocator
{
private:
    typedef dummy_test_allocator<T> self_t;
    typedef void *aux_pointer_t;
    typedef const void *cvoid_ptr;

public:
    typedef T value_type;
    typedef T *pointer;
    typedef const T *const_pointer;
    typedef typename std::add_lvalue_reference<value_type>::type reference;
    typedef typename std::add_lvalue_reference<value_type const>::type
        const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;


    template <class T2>
    struct rebind
    {
        typedef dummy_test_allocator<T2> other;
    };

    //!Default constructor. Never throws
    dummy_test_allocator()
    {
    }

    //!Constructor from other dummy_test_allocator. Never throws
    dummy_test_allocator(const dummy_test_allocator &)
    {
    }

    //!Constructor from related dummy_test_allocator. Never throws
    template <class T2>
    dummy_test_allocator(const dummy_test_allocator<T2> &)
    {
    }

    pointer address(reference value)
    {
        return pointer(std::addressof(value));
    }

    const_pointer address(const_reference value) const
    {
        return const_pointer(std::addressof(value));
    }

    pointer allocate(size_type, cvoid_ptr = 0)
    {
        return 0;
    }

    void deallocate(const pointer &, size_type)
    {
    }

    template <class Convertible>
    void construct(pointer, const Convertible &)
    {
    }

    void destroy(pointer)
    {
    }

    size_type max_size() const
    {
        return 0;
    }

    friend void swap(self_t &, self_t &)
    {
    }
};

//!Equality test for same type of dummy_test_allocator
template <class T>
inline bool operator==(
    const dummy_test_allocator<T> &, const dummy_test_allocator<T> &)
{
    return true;
}

//!Inequality test for same type of dummy_test_allocator
template <class T>
inline bool operator!=(
    const dummy_test_allocator<T> &, const dummy_test_allocator<T> &)
{
    return false;
}

}    //namespace test {

#endif
