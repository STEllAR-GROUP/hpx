//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(BOOST_LOCKFREE_DETAIL_APPLE_PRIMITIVES_JUL_11_2008_0408PM)
#define BOOST_LOCKFREE_DETAIL_APPLE_PRIMITIVES_JUL_11_2008_0408PM

#if !defined(__APPLE__)
#error "do not include this file on non-Apple platforms"
#endif

#include <libkern/OSAtomic.h>

#if (__GNUC__ > 4) || ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 2))
#include <ext/atomicity.h>
#else
#include <bits/atomicity.h>
#endif

#include <boost/cstdint.hpp>

#define BOOST_LOCKFREE_DCAS_ALIGNMENT
#define BOOST_LOCKFREE_CACHELINE_ALIGNMENT
#define BOOST_LOCKFREE_CACHELINE_ALIGNMENT_PREFIX

namespace boost { namespace lockfree
{
    ///////////////////////////////////////////////////////////////////////////
    inline void memory_barrier()
    {
        OSMemoryBarrier();
    }

    ///////////////////////////////////////////////////////////////////////////
    inline void spin(unsigned char i)
    {
        // do nothing
    }

    ///////////////////////////////////////////////////////////////////////////
    inline boost::uint64_t hrtimer_ticks()
    {
        return 0;     // no timings are recorded
    }

    ///////////////////////////////////////////////////////////////////////////
    template <class C, class D>
    inline bool CAS(volatile C * addr, D old, D nw)
    {
#if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
#warning "CAS: using OSAtomicCompareAndSwap32"
#endif
        return OSAtomicCompareAndSwap32((int32_t)old, (int32_t)nw, 
            (volatile int32_t*)addr);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <class C, class D, class E>
    inline bool CAS2(volatile C * addr, D old1, E old2, D new1, E new2)
    {
    // FIXME: for 32 bit processes on Intel hardware we can add a specialization 
    // using OSAtomicCompareAndSwap64 (what PP constants do we need to use?):

// # if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
// # warning ("CAS2: using OSAtomicCompareAndSwap64")
// # endif
//         struct packed_c
//         {
//             boost::int32_t d;
//             boost::int32_t e;
//         };
// 
//         typedef int DItype __attribute__ ((mode (DI)));
// 
//         BOOST_STATIC_ASSERT(sizeof(packed_c) == sizeof(TItype));
// 
//         union cu
//         {
//             packed_c c;
//             DItype l;
//         };
// 
//         cu old;
//         old.c.d = (boost::int32_t)old1;
//         old.c.e = (boost::int32_t)old2;
// 
//         cu nw;
//         nw.c.d = (boost::int32_t)new1;
//         nw.c.e = (boost::int32_t)new2;
// 
//         return OSAtomicCompareAndSwap64(old.l, nw.l,
//             reinterpret_cast<volatile DItype*>(addr));

# if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
# warning ("CAS2: blocking CAS2 emulation")
# endif

        struct packed_c
        {
            D d;
            E e;
        };

        volatile packed_c * packed_addr = reinterpret_cast<volatile packed_c*>(addr);

        boost::detail::lightweight_mutex::scoped_lock lock(detail::get_CAS2_mutex());

        if (packed_addr->d == old1 && packed_addr->e == old2)
        {
            packed_addr->d = new1;
            packed_addr->e = new2;
            return true;
        }
        return false;
    }

    ///////////////////////////////////////////////////////////////////////////
    inline int32_t interlocked_compare_exchange(int32_t volatile* addr, int32_t old, int32_t nw)
    {
        if (OSAtomicCompareAndSwap32(old, nw, addr))
            return old;
        return *addr;
    }

    inline D interlocked_compare_exchange(int64_t volatile* addr, int64_t old, int64_t nw)
    {
        if (OSAtomicCompareAndSwap64(old, nw, addr))
            return old;
        return *addr;
    }

    inline bool interlocked_bit_test_and_set(boost::int32_t volatile* x, boost::int32_t bit)
    {
        boost::uint32_t const value = 1u << bit;
        boost::int32_t old = *x;
        do {
            boost::int32_t const current = interlocked_compare_exchange(x, old, boost::int32_t(old | value));
            if (current == old)
            {
                break;
            }
            old = current;
        } while(true);
        return (old & value) != 0;
    }

    inline bool interlocked_bit_test_and_reset(boost::int32_t volatile* x, boost::int32_t bit)
    {
        boost::uint32_t const value = 1u << bit;
        boost::int32_t old = *x;
        do {
            boost::int32_t const current = interlocked_compare_exchange(x, old, boost::int32_t(old & ~value));
            if (current == old)
            {
                break;
            }
            old = current;
        } while(true);
        return (old & value) != 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    inline boost::int32_t interlocked_decrement(boost::int32_t volatile* value)
    {
        return OSAtomicDecrement32(value);
    }

    inline boost::int64_t interlocked_decrement(boost::int64_t volatile* value)
    {
        return OSAtomicDecrement64(value);
    }

    inline boost::int32_t interlocked_increment(boost::int32_t volatile* value)
    {
        return OSAtomicIncrement32(value);
    }

    inline boost::int64_t interlocked_increment(boost::int64_t volatile* value)
    {
        return OSAtomicIncrement64(value);
    }

    ///////////////////////////////////////////////////////////////////////////
    inline boost::int32_t interlocked_exchange_sub(boost::int32_t volatile* value, boost::int32_t sub)
    {
        return OSAtomicAdd32(-sub, value);
    }

    inline boost::int64_t interlocked_exchange_sub(boost::int64_t volatile* value, boost::int64_t sub)
    {
        return OSAtomicAdd64(-sub, value);
    }

    inline boost::int32_t interlocked_exchange_add(boost::int32_t volatile* value, boost::int32_t add)
    {
        return OSAtomicAdd32(add, value);
    }

    inline boost::int64_t interlocked_exchange_add(boost::int64_t volatile* value, boost::int64_t add)
    {
        return OSAtomicAdd64(add, value);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    inline T interlocked_exchange(T volatile *orig, T val)
    {
        return interlocked_compare_exchange(orig, *(T*)orig, val);
    }

    ///////////////////////////////////////////////////////////////////////////
    inline long interlocked_read_acquire(long volatile* x)
    {
        return interlocked_compare_exchange(x, 0, 0);
    }
    inline void* interlocked_read_acquire(void* volatile* x)
    {
        long t = interlocked_compare_exchange((long volatile*)x, 0, 0);
        return *reinterpret_cast<void**>(&t);
    }
    inline void interlocked_write_release(long volatile* x, long value)
    {
        interlocked_exchange(x, value);
    }
    inline void interlocked_write_release(void* volatile* x, void* value)
    {
        interlocked_exchange(reinterpret_cast<long volatile*>(x), 
            *reinterpret_cast<long*>(&value));
    }

}}

#endif
