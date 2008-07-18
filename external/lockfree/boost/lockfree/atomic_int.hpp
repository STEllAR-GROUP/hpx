//  Copyright (C) 2007, 2008 Tim Blechmann & Thomas Grill
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//  Disclaimer: Not a Boost library.

#ifndef BOOST_LOCKFREE_ATOMIC_INT_HPP
#define BOOST_LOCKFREE_ATOMIC_INT_HPP

#include <boost/lockfree/prefix.hpp>
#include <boost/noncopyable.hpp>

namespace boost
{
namespace lockfree
{

#if defined(__GNUC__) && ( (__GNUC__ > 4) || ((__GNUC__ >= 4) && (__GNUC_MINOR__ >= 1)) )

template <typename T>
class atomic_int:
    boost::noncopyable
{
public:
    explicit atomic_int(T v = 0):
        value(v)
    {}

    operator T(void) const
    {
        return __sync_fetch_and_add(&value, 0);
    }

    void operator =(T v)
    {
        value = v;
        __sync_synchronize();
    }

    T operator +=(T v)
    {
        return __sync_add_and_fetch(&value, v);
    }

    T operator -=(T v)
    {
        return __sync_sub_and_fetch(&value, v);
    }

    /* prefix operator */
    T operator ++(void)
    {
        return __sync_add_and_fetch(&value, 1);
    }

    /* prefix operator */
    T operator --(void)
    {
        return __sync_sub_and_fetch(&value, 1);
    }

    /* postfix operator */
    T operator ++(int)
    {
        return __sync_fetch_and_add(&value, 1);
    }

    /* postfix operator */
    T operator --(int)
    {
        return __sync_fetch_and_sub(&value, 1);
    }

private:
    mutable T value;
};

#elif defined(__GLIBCPP__) || defined(__GLIBCXX__)

template <typename T>
class atomic_int:
    boost::noncopyable
{
public:
    explicit atomic_int(T v = 0):
        value(v)
    {}

    operator T(void) const
    {
        return __gnu_cxx::__exchange_and_add(&value, 0);
    }

    void operator =(T v)
    {
        value = v;
    }

    T operator +=(T v)
    {
        return __gnu_cxx::__exchange_and_add(&value, v) + v;
    }

    T operator -=(T v)
    {
        return __gnu_cxx::__exchange_and_add(&value, -v) - v;
    }

    /* prefix operator */
    T operator ++(void)
    {
        return operator+=(1);
    }

    /* prefix operator */
    T operator --(void)
    {
        return operator-=(1);
    }

    /* postfix operator */
    T operator ++(int)
    {
        return __gnu_cxx::__exchange_and_add(&value, 1);
    }

    /* postfix operator */
    T operator --(int)
    {
        return __gnu_cxx::__exchange_and_add(&value, -1);
    }

private:
    mutable _Atomic_word value;
};

#else /* emulate via CAS */

template <typename T>
class atomic_int:
    boost::noncopyable
{
public:
    explicit atomic_int(T v = T(0))
    {
        *this = v;
    }

    operator T(void) const
    {
        memory_barrier();
        return value;
    }

    void operator =(T v)
    {
        value = v;
        memory_barrier();
    }

    /* prefix operator */
    T operator ++()
    {
        return *this += 1;
    }

    /* prefix operator */
    T operator --()
    {
        return *this -= 1;
    }

    T operator +=(T v)
    {
        for(;;)
        {
            T oldv = value;
            T newv = oldv + v;
            if(likely(CAS(&value, oldv, newv)))
                return newv;
        }
    }

    T operator -=(T v)
    {
        for(;;)
        {
            T oldv = value;
            T newv = oldv + v;

            if(likely(CAS(&value, oldv, newv)))
                return newv;
        }
    }

    /* postfix operator */
    T operator ++(int)
    {
        for(;;)
        {
            T oldv = value;
            if(likely(CAS(&value, oldv, oldv+1)))
                return oldv;
        }
    }

    /* postfix operator */
    T operator --(int)
    {
        for(;;)
        {
            T oldv = value;
            if(likely(CAS(&value, oldv, oldv-1)))
                return oldv;
        }
    }

private:
    T value;
};


#endif

} /* namespace lockfree */
} /* namespace boost */

#endif /* BOOST_LOCKFREE_ATOMIC_INT_HPP */
