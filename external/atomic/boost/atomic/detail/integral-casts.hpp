#ifndef BOOST_DETAIL_ATOMIC_INTEGRAL_CASTS_HPP
#define BOOST_DETAIL_ATOMIC_INTEGRAL_CASTS_HPP

//  Copyright (c) 2009 Helge Bahmann
//  Copyright (c) 2011 Bryce Lelbach & Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string.h>
#include <boost/cstdint.hpp>

namespace boost { namespace detail { namespace atomic {

template<typename T>
class platform_atomic<T, 1> : private platform_atomic_integral<boost::uint8_t> {
public:
    typedef platform_atomic_integral<boost::uint8_t> super;
#if defined(BOOST_ATOMIC_ENFORCE_PODNESS)
    typedef union { T e; boost::uint8_t i;} conv;
#endif

    platform_atomic() {}
    explicit platform_atomic(T t) : super(to_integral(t))
    {
    }

    void store(T t, memory_order order=memory_order_seq_cst) volatile
    {
        super::store(to_integral(t), order);
    }
    T load(memory_order order=memory_order_seq_cst) volatile const
    {
        return from_integral(super::load(order));
    }
    bool compare_exchange_strong(
        T &expected,
        T desired,
        memory_order success_order,
        memory_order failure_order) volatile
    {
        boost::uint8_t _expected, _desired;
        _expected=to_integral(expected);
        _desired=to_integral(desired);
        bool success=super::compare_exchange_strong(_expected, _desired, success_order,
            failure_order);
        expected=from_integral(_expected);
        return success;
    }
    bool compare_exchange_weak(
        T &expected,
        T desired,
        memory_order success_order,
        memory_order failure_order) volatile
    {
        boost::uint8_t _expected, _desired;
        _expected=to_integral(expected);
        _desired=to_integral(desired);
        bool success=super::compare_exchange_weak(_expected, _desired, success_order,
            failure_order);
        expected=from_integral(_expected);
        return success;
    }

    T exchange(T replacement, memory_order order=memory_order_seq_cst) volatile
    {
        return from_integral(super::exchange(to_integral(replacement), order));
    }

    operator T(void) const volatile {return load();}
    T operator=(T v) volatile {store(v); return v;}

    using super::is_lock_free;
protected:
    static inline boost::uint8_t to_integral(T &t)
    {
        boost::uint8_t tmp;
        memcpy(&tmp, &t, sizeof(t));
        return tmp;
    }
    static inline T from_integral(boost::uint8_t t)
    {
        T tmp;
        memcpy(&tmp, &t, sizeof(t));
        return tmp;
    }
};

template<typename T>
class platform_atomic<T, 2> : private platform_atomic_integral<boost::uint16_t> {
public:
    typedef platform_atomic_integral<boost::uint16_t> super;
#if defined(BOOST_ATOMIC_ENFORCE_PODNESS)
    typedef union { T e; boost::uint16_t i;} conv;
#endif

    platform_atomic() {}
    explicit platform_atomic(T t) : super(to_integral(t))
    {
    }

    void store(T t, memory_order order=memory_order_seq_cst) volatile
    {
        super::store(to_integral(t), order);
    }
    T load(memory_order order=memory_order_seq_cst) volatile const
    {
        return from_integral(super::load(order));
    }
    bool compare_exchange_strong(
        T &expected,
        T desired,
        memory_order success_order,
        memory_order failure_order) volatile
    {
        boost::uint16_t _expected, _desired;
        _expected=to_integral(expected);
        _desired=to_integral(desired);
        bool success=super::compare_exchange_strong(_expected, _desired, success_order,
            failure_order);
        expected=from_integral(_expected);
        return success;
    }
    bool compare_exchange_weak(
        T &expected,
        T desired,
        memory_order success_order,
        memory_order failure_order) volatile
    {
        boost::uint16_t _expected, _desired;
        _expected=to_integral(expected);
        _desired=to_integral(desired);
        bool success=super::compare_exchange_weak(_expected, _desired, success_order,
            failure_order);
        expected=from_integral(_expected);
        return success;
    }

    T exchange(T replacement, memory_order order=memory_order_seq_cst) volatile
    {
        return from_integral(super::exchange(to_integral(replacement), order));
    }

    operator T(void) const volatile {return load();}
    T operator=(T v) volatile {store(v); return v;}

    using super::is_lock_free;
protected:
    static inline boost::uint16_t to_integral(T &t)
    {
        boost::uint16_t tmp;
        memcpy(&tmp, &t, sizeof(t));
        return tmp;
    }
    static inline T from_integral(boost::uint16_t t)
    {
        T tmp;
        memcpy(&tmp, &t, sizeof(t));
        return tmp;
    }
};

template<typename T>
class platform_atomic<T, 4> : private platform_atomic_integral<boost::uint32_t> {
public:
    typedef platform_atomic_integral<boost::uint32_t> super;
#if defined(BOOST_ATOMIC_ENFORCE_PODNESS)
    typedef union { T e; boost::uint32_t i;} conv;
#endif

    platform_atomic() {}
    explicit platform_atomic(T t) : super(to_integral(t))
    {
    }

    void store(T t, memory_order order=memory_order_seq_cst) volatile
    {
        super::store(to_integral(t), order);
    }
    T load(memory_order order=memory_order_seq_cst) volatile const
    {
        return from_integral(super::load(order));
    }
    bool compare_exchange_strong(
        T &expected,
        T desired,
        memory_order success_order,
        memory_order failure_order) volatile
    {
        boost::uint32_t _expected, _desired;
        _expected=to_integral(expected);
        _desired=to_integral(desired);
        bool success=super::compare_exchange_strong(_expected, _desired, success_order,
            failure_order);
        expected=from_integral(_expected);
        return success;
    }
    bool compare_exchange_weak(
        T &expected,
        T desired,
        memory_order success_order,
        memory_order failure_order) volatile
    {
        boost::uint32_t _expected, _desired;
        _expected=to_integral(expected);
        _desired=to_integral(desired);
        bool success=super::compare_exchange_weak(_expected, _desired, success_order,
            failure_order);
        expected=from_integral(_expected);
        return success;
    }

    T exchange(T replacement, memory_order order=memory_order_seq_cst) volatile
    {
        return from_integral(super::exchange(to_integral(replacement), order));
    }

    operator T(void) const volatile {return load();}
    T operator=(T v) volatile {store(v); return v;}

    using super::is_lock_free;
protected:
    static inline boost::uint32_t to_integral(T &t)
    {
        boost::uint32_t tmp;
        memcpy(&tmp, &t, sizeof(t));
        return tmp;
    }
    static inline T from_integral(boost::uint32_t t)
    {
        T tmp;
        memcpy(&tmp, &t, sizeof(t));
        return tmp;
    }
};

template<typename T>
class platform_atomic<T, 8> : private platform_atomic_integral<boost::uint64_t> {
public:
    typedef platform_atomic_integral<boost::uint64_t> super;
#if defined(BOOST_ATOMIC_ENFORCE_PODNESS)
    typedef union { T e; boost::uint64_t i;} conv;
#endif

    platform_atomic() {}
    explicit platform_atomic(T t) : super(to_integral(t))
    {
    }

    void store(T t, memory_order order=memory_order_seq_cst) volatile
    {
        super::store(to_integral(t), order);
    }
    T load(memory_order order=memory_order_seq_cst) volatile const
    {
        return from_integral(super::load(order));
    }
    bool compare_exchange_strong(
        T &expected,
        T desired,
        memory_order success_order,
        memory_order failure_order) volatile
    {
        boost::uint64_t _expected, _desired;
        _expected=to_integral(expected);
        _desired=to_integral(desired);
        bool success=super::compare_exchange_strong(_expected, _desired, success_order,
            failure_order);
        expected=from_integral(_expected);
        return success;
    }
    bool compare_exchange_weak(
        T &expected,
        T desired,
        memory_order success_order,
        memory_order failure_order) volatile
    {
        boost::uint64_t _expected, _desired;
        _expected=to_integral(expected);
        _desired=to_integral(desired);
        bool success=super::compare_exchange_weak(_expected, _desired, success_order,
            failure_order);
        expected=from_integral(_expected);
        return success;
    }

    T exchange(T replacement, memory_order order=memory_order_seq_cst) volatile
    {
        return from_integral(super::exchange(to_integral(replacement), order));
    }

    operator T(void) const volatile {return load();}
    T operator=(T v) volatile {store(v); return v;}

    using super::is_lock_free;
protected:
    static inline boost::uint64_t to_integral(T &t)
    {
        boost::uint64_t tmp;
        memcpy(&tmp, &t, sizeof(t));
        return tmp;
    }
    static inline T from_integral(boost::uint64_t t)
    {
        T tmp;
        memcpy(&tmp, &t, sizeof(t));
        return tmp;
    }
};

#if (defined(__amd64__) || defined(__x86_64__)) && \
    defined(BOOST_ATOMIC_HAVE_SSE2) && \
    defined(BOOST_ATOMIC_HAVE_GNU_SYNC_16) && \
    defined(BOOST_ATOMIC_HAVE_GNU_ALIGNED_16) && \
    defined(BOOST_ATOMIC_HAVE_GNU_128BIT_INTEGERS)

#define BOOST_ATOMIC_HAVE_128BIT_SUPPORT

template<typename T>
class platform_atomic<T, 16> : private platform_atomic_integral<__uint128_t> {
public:
    typedef platform_atomic_integral<__uint128_t> super;
#if defined(BOOST_ATOMIC_ENFORCE_PODNESS)
    typedef union { T e; __uint128_t i;} conv;
#endif

    platform_atomic() {}
    explicit platform_atomic(T t) : super(to_integral(t))
    {
    }

    void store(T t, memory_order order=memory_order_seq_cst) volatile
    {
        super::store(to_integral(t), order);
    }
    T load(memory_order order=memory_order_seq_cst) volatile const
    {
        return from_integral(super::load(order));
    }
    bool compare_exchange_strong(
        T &expected,
        T desired,
        memory_order success_order,
        memory_order failure_order) volatile
    {
        __uint128_t _expected, _desired;
        _expected=to_integral(expected);
        _desired=to_integral(desired);
        bool success=super::compare_exchange_strong(_expected, _desired,
            success_order, failure_order);
        expected=from_integral(_expected);
        return success;
    }
    bool compare_exchange_weak(
        T &expected,
        T desired,
        memory_order success_order,
        memory_order failure_order) volatile
    {
        __uint128_t _expected, _desired;
        _expected=to_integral(expected);
        _desired=to_integral(desired);
        bool success=super::compare_exchange_weak(_expected, _desired,
            success_order, failure_order);
        expected=from_integral(_expected);
        return success;
    }

    T exchange(T replacement, memory_order order=memory_order_seq_cst) volatile
    {
        return from_integral(super::exchange(to_integral(replacement), order));
    }

    operator T(void) const volatile {return load();}
    T operator=(T v) volatile {store(v); return v;}

    using super::is_lock_free;
protected:
    static inline __uint128_t to_integral(T &t)
    {
        __uint128_t tmp;
        memcpy(&tmp, &t, sizeof(t));
        return tmp;
    }
    static inline T from_integral(__uint128_t t)
    {
        T tmp;
        memcpy(&tmp, &t, sizeof(t));
        return tmp;
    }
};

#elif BOOST_MSVC >= 1500 && (defined(_M_IA64) || defined(_M_AMD64)) && \
      defined(BOOST_ATOMIC_HAVE_SSE2)

#define BOOST_ATOMIC_HAVE_128BIT_SUPPORT

}}}

#include <emmintrin.h>

namespace boost { namespace detail { namespace atomic {

template<typename T>
class platform_atomic<T, 16> : private platform_atomic_integral<__m128i> {
public:
    typedef platform_atomic_integral<__m128i> super;
#if defined(BOOST_ATOMIC_ENFORCE_PODNESS)
    typedef union { T e; __m128i i;} conv;
#endif

    platform_atomic() {}
    explicit platform_atomic(T t) : super(to_integral(t))
    {
    }

    void store(T t, memory_order order=memory_order_seq_cst) volatile
    {
        super::store(to_integral(t), order);
    }
    T load(memory_order order=memory_order_seq_cst) volatile const
    {
        return from_integral(super::load(order));
    }
    bool compare_exchange_strong(
        T &expected,
        T desired,
        memory_order success_order,
        memory_order failure_order) volatile
    {
        __m128i _expected, _desired;
        _expected=to_integral(expected);
        _desired=to_integral(desired);
        bool success=super::compare_exchange_strong(_expected, _desired,
            success_order, failure_order);
        expected=from_integral(_expected);
        return success;
    }
    bool compare_exchange_weak(
        T &expected,
        T desired,
        memory_order success_order,
        memory_order failure_order) volatile
    {
        __m128i _expected, _desired;
        _expected=to_integral(expected);
        _desired=to_integral(desired);
        bool success=super::compare_exchange_weak(_expected, _desired,
            success_order, failure_order);
        expected=from_integral(_expected);
        return success;
    }

    T exchange(T replacement, memory_order order=memory_order_seq_cst) volatile
    {
        return from_integral(super::exchange(to_integral(replacement), order));
    }

    operator T(void) const volatile {return load();}
    T operator=(T v) volatile {store(v); return v;}

    using super::is_lock_free;
protected:
    static inline __m128i to_integral(T &t)
    {
        __m128i tmp;
        memcpy(&tmp, &t, sizeof(t));
        return tmp;
    }
    static inline T from_integral(__m128i t)
    {
        T tmp;
        memcpy(&tmp, &t, sizeof(t));
        return tmp;
    }
};

#endif

} } }

#endif
