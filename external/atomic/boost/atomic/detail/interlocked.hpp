#ifndef BOOST_DETAIL_ATOMIC_INTERLOCKED_HPP
#define BOOST_DETAIL_ATOMIC_INTERLOCKED_HPP

//  Copyright (c) 2009 Helge Bahmann
//  Copyright (c) 2011-2012 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/detail/interlocked.hpp>

#include <boost/atomic/detail/base.hpp>
#include <boost/atomic/detail/builder.hpp>

namespace boost { namespace detail { namespace atomic {

static inline void full_fence(void)
{
    long tmp;
    BOOST_INTERLOCKED_EXCHANGE(&tmp, 0);
}

template<>
inline void platform_atomic_thread_fence(memory_order order)
{
    switch(order) {
        case memory_order_seq_cst:
            full_fence();
        default:;
    }
}

static inline void fence_after_load(memory_order order)
{
    switch(order) {
        case memory_order_seq_cst:
            full_fence();
        case memory_order_acquire:
        case memory_order_acq_rel:
        default:;
    }
}


template<typename T>
class atomic_interlocked_32 {
public:
    explicit atomic_interlocked_32(T v) : i(v) {}
    atomic_interlocked_32() {}
    T load(memory_order order=memory_order_seq_cst) const volatile
    {
        T v=*reinterpret_cast<volatile const T *>(&i);
        fence_after_load(order);
        return v;
    }
    void store(T v, memory_order order=memory_order_seq_cst) volatile
    {
        if (order!=memory_order_seq_cst) {
            *reinterpret_cast<volatile T *>(&i)=v;
        } else {
            exchange(v);
        }
    }
    bool compare_exchange_strong(
        T &expected,
        T desired,
        memory_order success_order,
        memory_order failure_order) volatile
    {
        T prev=expected;
        expected=(T)BOOST_INTERLOCKED_COMPARE_EXCHANGE((long *)(&i),
            (long)desired, (long)expected);
        bool success=(prev==expected);
        return success;
    }
    bool compare_exchange_weak(
        T &expected,
        T desired,
        memory_order success_order,
        memory_order failure_order) volatile
    {
        return compare_exchange_strong(expected, desired, success_order, failure_order);
    }
    T exchange(T r, memory_order order=memory_order_seq_cst) volatile
    {
        return (T)BOOST_INTERLOCKED_EXCHANGE((long *)&i, (long)r);
    }
    T fetch_add(T c, memory_order order=memory_order_seq_cst) volatile
    {
        return (T)BOOST_INTERLOCKED_EXCHANGE_ADD((long *)&i, c);
    }

    bool is_lock_free(void) const volatile {return true;}

    typedef T integral_type;
private:
    T i;
};

}}}

# if defined(_M_IA64) || defined(_M_AMD64)

#if defined( BOOST_USE_WINDOWS_H )

# include <windows.h>

# define BOOST_INTERLOCKED_EXCHANGE_ADD64 InterlockedExchangeAdd64
# define BOOST_INTERLOCKED_EXCHANGE64 InterlockedExchange64
# define BOOST_INTERLOCKED_COMPARE_EXCHANGE64 InterlockedCompareExchange64

#else

extern "C" boost::int64_t __cdecl _InterlockedExchangeAdd64(boost::int64_t volatile *,
    boost::int64_t);
extern "C" boost::int64_t __cdecl _InterlockedExchange64(boost::int64_t volatile *,
    boost::int64_t);
extern "C" boost::int64_t __cdecl _InterlockedCompareExchange64(
    boost::int64_t volatile *, boost::int64_t, boost::int64_t);

# pragma intrinsic( _InterlockedExchangeAdd64 )
# pragma intrinsic( _InterlockedExchange64 )
# pragma intrinsic( _InterlockedCompareExchange64 )

# define BOOST_INTERLOCKED_EXCHANGE_ADD64 _InterlockedExchangeAdd64
# define BOOST_INTERLOCKED_EXCHANGE64 _InterlockedExchange64
# define BOOST_INTERLOCKED_COMPARE_EXCHANGE64 _InterlockedCompareExchange64

#endif

namespace boost { namespace detail { namespace atomic {

template<typename T>
class __declspec(align(8)) atomic_interlocked_64 {
public:
    explicit atomic_interlocked_64(T v) : i(v) {}
    atomic_interlocked_64() {}
    T load(memory_order order=memory_order_seq_cst) const volatile
    {
        T v=*reinterpret_cast<volatile const T *>(&i);
        fence_after_load(order);
        return v;
    }
    void store(T v, memory_order order=memory_order_seq_cst) volatile
    {
        if (order!=memory_order_seq_cst) {
            *reinterpret_cast<volatile T *>(&i)=v;
        } else {
            exchange(v);
        }
    }
    bool compare_exchange_strong(
        T &expected,
        T desired,
        memory_order success_order,
        memory_order failure_order) volatile
    {
        T prev=expected;
        expected=(T)BOOST_INTERLOCKED_COMPARE_EXCHANGE64((boost::int64_t *)(&i),
            (boost::int64_t)desired, (boost::int64_t)expected);
        bool success=(prev==expected);
        return success;
    }
    bool compare_exchange_weak(
        T &expected,
        T desired,
        memory_order success_order,
        memory_order failure_order) volatile
    {
        return compare_exchange_strong(expected, desired, success_order, failure_order);
    }
    T exchange(T r, memory_order order=memory_order_seq_cst) volatile
    {
        return (T)BOOST_INTERLOCKED_EXCHANGE64((boost::int64_t *)&i, (boost::int64_t)r);
    }
    T fetch_add(T c, memory_order order=memory_order_seq_cst) volatile
    {
      return (T)BOOST_INTERLOCKED_EXCHANGE_ADD64((boost::int64_t *)&i, c);
    }

    bool is_lock_free(void) const volatile {return true;}

    typedef T integral_type;
private:
    T i;
};

}}}

// _InterlockedCompareExchange128 is available only starting with VS2008
#if BOOST_MSVC >= 1500 && defined(BOOST_ATOMIC_HAVE_SSE2)

# include <emmintrin.h>

extern "C" unsigned char __cdecl _InterlockedCompareExchange128(
    boost::int64_t volatile *Destination,
    boost::int64_t ExchangeHigh, boost::int64_t ExchangeLow,
    boost::int64_t *Comparand);
extern "C" __m128i _mm_load_si128(__m128i const*_P);
extern "C" void _mm_store_si128(__m128i *_P, __m128i _B);

# pragma intrinsic( _InterlockedCompareExchange128 )
# pragma intrinsic( _mm_load_si128 )
# pragma intrinsic( _mm_store_si128 )

# define BOOST_INTERLOCKED_COMPARE_EXCHANGE128 _InterlockedCompareExchange128

namespace boost { namespace detail { namespace atomic {

template<typename T>
class __declspec(align(16)) atomic_interlocked_128 {
public:
    explicit atomic_interlocked_128(T v) : i(v) {}
    atomic_interlocked_128() {}
    T load(memory_order order=memory_order_seq_cst) const volatile
    {
        T v;
        if (order!=memory_order_seq_cst) {
            v = *(T const*)(&i);
        }
        else {
            v = _mm_load_si128((__m128i const*)(&i));
        }
        fence_after_load(order);
        return v;
    }
    void store(T v, memory_order order=memory_order_seq_cst) volatile
    {
        if (order!=memory_order_seq_cst) {
            *(T*)(&i)=v;
        }
        else {
            _mm_store_si128(*(__m128i*)(&i), v);
        }
    }
    bool compare_exchange_strong(
        T &expected,
        T desired,
        memory_order success_order,
        memory_order failure_order) volatile
    {
        boost::int64_t* desired_raw = (boost::int64_t*)&desired;
        T prev = *(__m128i*)(&i);
        bool success = BOOST_INTERLOCKED_COMPARE_EXCHANGE128(
            (boost::int64_t volatile *)(&i),
            desired_raw[1], desired_raw[0], (boost::int64_t*)&expected) != 0;
        if (!success)
            expected = prev;
        return success;
    }
    bool compare_exchange_weak(
        T &expected,
        T desired,
        memory_order success_order,
        memory_order failure_order) volatile
    {
        return compare_exchange_strong(expected, desired, success_order, failure_order);
    }
    T exchange(T r, memory_order order=memory_order_seq_cst) volatile
    {
        boost::int64_t* desired_raw = (boost::int64_t*)&r;
        T prev = i;

        while (!BOOST_INTERLOCKED_COMPARE_EXCHANGE128(
                  (boost::int64_t volatile*)&i, desired_raw[1], desired_raw[0],
                  (boost::int64_t*)&i))
        {}

        return prev;
    }
    T fetch_add(T c, memory_order order=memory_order_seq_cst) volatile
    {
        T expected = i;
        __m128i desired;

        do {
            desired = _mm_add_epi32(*(__m128i*)(&expected), *(__m128i*)(&c));
        } while (!compare_exchange_strong(expected, *(T*)(&desired),
            order, memory_order_relaxed));

        return expected;
    }

    bool is_lock_free(void) const volatile {return true;}

    typedef T integral_type;
private:
    T i;
};

}}}

#endif

#endif

namespace boost { namespace detail { namespace atomic {

template<typename T>
class platform_atomic_integral<T, 4>
    : public build_atomic_from_add<atomic_interlocked_32<T> > {
public:
    typedef build_atomic_from_add<atomic_interlocked_32<T> > super;
    explicit platform_atomic_integral(T v) : super(v) {}
    platform_atomic_integral(void) {}
};

template<typename T>
class platform_atomic_integral<T, 1>
    : public build_atomic_from_larger_type<atomic_interlocked_32<uint32_t>, T> {
public:
    typedef build_atomic_from_larger_type<atomic_interlocked_32<uint32_t>, T> super;

    explicit platform_atomic_integral(T v) : super(v) {}
    platform_atomic_integral(void) {}
};

template<typename T>
class platform_atomic_integral<T, 2>
    : public build_atomic_from_larger_type<atomic_interlocked_32<uint32_t>, T> {
public:
    typedef build_atomic_from_larger_type<atomic_interlocked_32<uint32_t>, T> super;

    explicit platform_atomic_integral(T v) : super(v) {}
    platform_atomic_integral(void) {}
};

# if defined(_M_IA64) || defined(_M_AMD64)
template<typename T>
class platform_atomic_integral<T, 8>
  : public build_atomic_from_add<atomic_interlocked_64<uint64_t> >
{
public:
    typedef build_atomic_from_add<atomic_interlocked_64<uint64_t> > super;

    explicit platform_atomic_integral(T v) : super(v) {}
    platform_atomic_integral(void) {}
};

template<>
class platform_atomic_integral<void*, 8>
  : public build_atomic_from_add<atomic_interlocked_64<void*> >
{
public:
    typedef build_atomic_from_add<atomic_interlocked_64<void*> > super;

    explicit platform_atomic_integral(void* v) : super(v) {}
    platform_atomic_integral(void) {}
};

#if BOOST_MSVC >= 1500 && defined(BOOST_ATOMIC_HAVE_SSE2)

template<typename T>
class platform_atomic_integral<T, 16>
  : public build_atomic_from_add<atomic_interlocked_128<T> >
{
public:
    typedef build_atomic_from_add<atomic_interlocked_128<T> > super;
    explicit platform_atomic_integral(T v) : super(v) {}
    platform_atomic_integral(void) {}
};

#endif

#endif

}}}

#endif
