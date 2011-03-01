////////////////////////////////////////////////////////////////////////////////
//  Copyright 2003 & onward LASMEA UMR 6602 CNRS/Univ. Clermont II
//  Copyright 2009 & onward LRI    UMR 8623 CNRS/Univ Paris Sud XI
//  Copyright (c) 2011 Bryce Lelbach
//
//  Most of this code was blatantly copied, pasted and hacked from nt2's
//  codebase (https://github.com/MetaScale/nt2) - wash
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_5CFC999F_B036_459D_A22A_EF679F85471E)
#define HPX_5CFC999F_B036_459D_A22A_EF679F85471E

#include <cstddef>

namespace hpx { namespace memory
{

// conforms to the C++ STD allocator interface
template <typename Tag, typename T>
struct allocator
{
    typedef T               value_type;
    typedef T*              pointer;
    typedef T const*        const_pointer;
    typedef T&              reference;
    typedef T const&        const_reference;
    typedef std::size_t     size_type;
    typedef std::ptrdiff_t  difference_type;

    template <typename U>
    struct rebind {
      typedef allocator<Tag, U> other;
    };

    allocator() {}

    template <typename U>
    allocator(allocator<Tag, U> const&) {}

    ~allocator() {}

    allocator& operator=(allocator const&)
    {
        return *this;
    }

    pointer address(reference r)
    {
        return &r;
    }

    const_pointer address(const_reference r)
    {
        return &r;
    }

    size_type max_size() const
    {
        return size_type(~0);
    }

    void construct(pointer p, T const& t)
    {
        p = new (p) value_type(t);
    }

    void destroy(pointer p)
    {
        p->~value_type();
    }

    pointer allocate(size_type c, const void* = 0) const
    {
        return Tag::template object_malloc<value_type>(c);
    }

    void deallocate(pointer p, size_type) const
    {
        Tag::free(p);
    }
};

template<class Tag, class T>
bool operator== (allocator<Tag, T> const&, allocator<Tag, T> const&)
{
    return true;
}

template<class Tag, class T>
bool operator!= (allocator<Tag, T> const&, allocator<Tag, T> const&)
{
    return false;
}

}}

#endif // HPX_5CFC999F_B036_459D_A22A_EF679F85471E

