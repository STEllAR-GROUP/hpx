//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Part of this code has been adopted from code published under the BSL by:
//
//  Copyright (C) 2007, 2008 Tim Blechmann & Thomas Grill
//  (C) Copyright 2005-7 Anthony Williams 
//  (C) Copyright 2007 David Deakins 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(BOOST_LOCKFREE_DETAIL_WINDOWS_PRIMITIVES_JUL_11_2008_0407PM)
#define BOOST_LOCKFREE_DETAIL_WINDOWS_PRIMITIVES_JUL_11_2008_0407PM

#include <boost/config.hpp>
#include <boost/cstdint.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/detail/interlocked.hpp>

#if !defined(BOOST_MSVC)
#error "do not include this file on non-MSVC platforms"
#endif

#include <windows.h>

#if BOOST_MSVC < 1400
extern "C" void __cdecl _ReadWriteBarrier();
#if defined(_M_IA64) || defined(_WIN64)
extern "C" LONG64 __cdecl _InterlockedCompareExchange64(LONG64 volatile*, LONG64 Exchange, LONG64 Comp);
#endif
#else
#include <intrin.h>
#endif

#pragma intrinsic(_ReadWriteBarrier)
#if defined(_M_IA64) || defined(_WIN64)
#pragma intrinsic(_InterlockedCompareExchange64)
#endif

#define BOOST_LOCKFREE_CACHELINE_ALIGNMENT 
#define BOOST_LOCKFREE_CACHELINE_ALIGNMENT_PREFIX __declspec(align(64))

#if defined(_M_IX86)
    #define BOOST_LOCKFREE_DCAS_ALIGNMENT
#elif defined(_M_X64) || defined(_M_IA64)
    #define BOOST_LOCKFREE_DCAS_ALIGNMENT __declspec(align(16))
    #if !defined(BOOST_LOCKFREE_HAS_CMPXCHG16B)
        #define BOOST_LOCKFREE_PTR_COMPRESSION 1
    #endif
#endif

#if defined(BOOST_LOCKFREE_HAS_CMPXCHG16B)
extern "C" bool CAS2_windows64(volatile __int64* addr, 
    __int64 old1, __int64 old2, __int64 new1, __int64 new2) throw();
#endif

namespace boost { namespace lockfree
{
    ///////////////////////////////////////////////////////////////////////////
    inline void memory_barrier()
    {
#if BOOST_MSVC >= 1300
        _ReadWriteBarrier();
#else
#warning "no memory barrier implemented for this platform"
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    inline void spin(unsigned char i)
    {
        // do nothing
    }

    ///////////////////////////////////////////////////////////////////////////
    inline boost::uint64_t hrtimer_ticks()
    {
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        return (boost::uint64_t)now.QuadPart;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <class C, class D>
    inline bool CAS (volatile C * addr, D old, D nw, boost::mpl::true_)
    {
        return BOOST_INTERLOCKED_COMPARE_EXCHANGE(addr, nw, old) == old;
    }

#if defined(_M_IA64) || defined(_WIN64)
    template <class C, class D>
    inline bool CAS (volatile C * addr, D old, D nw, boost::mpl::false_)
    {
        return _InterlockedCompareExchange64((LONG64 volatile*)addr, (LONG64)nw, (LONG64)old) == old;
    }
#endif

    template <class C, class D>
    inline bool CAS (volatile C * addr, D old, D nw)
    {
        return CAS (addr, old, nw, boost::mpl::bool_<sizeof(C) == sizeof(long)>());
    }

    ///////////////////////////////////////////////////////////////////////////
    template <class C, class D, class E>
    inline bool CAS2(volatile C * addr, D old1, E old2, D new1, E new2)
    {
#if defined(_M_IA64) || defined(_WIN64)
// early AMD processors do not support the cmpxchg16b instruction
#if defined(BOOST_LOCKFREE_HAS_CMPXCHG16B)
#if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
#warning "CAS2: using CAS2_windows64, may crash on old AMD systems"
#endif

        return CAS2_windows64(reinterpret_cast<volatile __int64*>(addr), 
            (__int64)old1, (__int64)old2, (__int64)new1, (__int64)new2);

#else
#if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
#warning "CAS2: blocking cas emulation"
#endif

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
#endif
#else
#if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
#warning "CAS2: 32Bit hand coded asm"
#endif

        bool ok;
        __asm {
            mov eax, [old1]
            mov edx, [old2]
            mov ebx, [new1]
            mov ecx, [new2]
            mov edi, [addr]
            lock cmpxchg8b [edi]
            setz [ok]
        }
        return ok;
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    template <class C, class D>
    inline D interlocked_compare_exchange(volatile C * addr, D old, D nw)
    {
        return BOOST_INTERLOCKED_COMPARE_EXCHANGE(addr, nw, old);
    }

#if (_MSC_VER >= 1400) && !defined(UNDER_CE)

#if _MSC_VER==1400
    extern "C" unsigned char _interlockedbittestandset(long *a,long b);
    extern "C" unsigned char _interlockedbittestandreset(long *a,long b);
#else
    extern "C" unsigned char _interlockedbittestandset(volatile long *a,long b);
    extern "C" unsigned char _interlockedbittestandreset(volatile long *a,long b);
#endif

#pragma intrinsic(_interlockedbittestandset)
#pragma intrinsic(_interlockedbittestandreset)

    inline bool interlocked_bit_test_and_set(long* x, long bit)
    {
        return _interlockedbittestandset(x, bit) != 0;
    }

    inline bool interlocked_bit_test_and_reset(long* x, long bit)
    {
        return _interlockedbittestandreset(x, bit) != 0;
    }

#elif defined(BOOST_INTEL_WIN) && defined(_M_IX86)

    inline bool interlocked_bit_test_and_set(long* x, long bit)
    {
        __asm {
            mov eax, bit;
            mov edx, x;
            lock bts [edx], eax;
            setc al;
        };
    }

    inline bool interlocked_bit_test_and_reset(long* x, long bit)
    {
        __asm {
            mov eax, bit;
            mov edx, x;
            lock btr [edx], eax;
            setc al;
        };
    }

#else

    inline bool interlocked_bit_test_and_set(long* x, long bit)
    {
        long const value = 1 << bit;
        long old = *x;
        do {
            long const current = interlocked_compare_exchange(x, old, old | value);
            if (current == old)
            {
                break;
            }
            old = current;
        } while(true);
        return (old & value) != 0;
    }

    inline bool interlocked_bit_test_and_reset(long* x, long bit)
    {
        long const value = 1 << bit;
        long old = *x;
        do {
            long const current = interlocked_compare_exchange(x, old, old & ~value);
            if (current == old)
            {
                break;
            }
            old = current;
        } while(true);
        return (old & value) != 0;
    }

#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    inline T interlocked_decrement(T* value)
    {
        return BOOST_INTERLOCKED_DECREMENT(value);
    }

    template <typename T>
    inline T interlocked_increment(T* value)
    {
        return BOOST_INTERLOCKED_INCREMENT(value);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    inline T interlocked_exchange_sub(T* value, T sub)
    {
        return BOOST_INTERLOCKED_EXCHANGE_ADD(value, -sub);
    }

    template <typename T>
    inline T interlocked_exchange_add(T* value, T add)
    {
        return BOOST_INTERLOCKED_EXCHANGE_ADD(value, add);
    }

}}

#endif
