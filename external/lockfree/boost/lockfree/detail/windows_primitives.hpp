//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(BOOST_LOCKFREE_DETAIL_WINDOWS_PRIMITIVES_JUL_11_2008_0407PM)
#define BOOST_LOCKFREE_DETAIL_WINDOWS_PRIMITIVES_JUL_11_2008_0407PM

#include <boost/config.hpp>

#if !defined(BOOST_MSVC)
#error "do not include this file on non-MSVC platforms"
#endif

#include <windows.h>

#if BOOST_MSVC < 1400
extern "C" void __cdecl _ReadWriteBarrier();
extern "C" LONG __cdecl _InterlockedCompareExchange(LONG volatile*, LONG Exchange, LONG Comp);
#else
#include <intrin.h>
#endif

#pragma intrinsic(_ReadWriteBarrier)
#pragma intrinsic(_InterlockedCompareExchange)

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
    template <class C, class D>
    inline bool CAS (volatile C * addr, D old, D nw)
    {
        return _InterlockedCompareExchange(addr, nw, old) == old;
    }

    template <class C, class D>
    inline bool CAS (volatile C ** addr, D* old, D* nw)
    {
        return _InterlockedCompareExchange64(addr, nw, old) == old;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <class C, class D, class E>
    inline bool CAS2(volatile C * addr, D old1, E old2, D new1, E new2)
    {
#if defined(_M_IA64) || defined(_WIN64)
// early AMD processors do not support the cmpxchg16b instruction
#if defined(BOOST_LOCKFREE_HAS_CMPXCHG16B)

        return CAS2_windows64(reinterpret_cast<volatile __int64*>(addr), 
            (__int64)old1, (__int64)old2, (__int64)new1, (__int64)new2);

#else
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

}}

#endif
