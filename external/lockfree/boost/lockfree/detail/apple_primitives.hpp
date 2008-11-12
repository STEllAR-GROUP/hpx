//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(BOOST_LOCKFREE_DETAIL_APPLE_PRIMITIVES_JUL_11_2008_0408PM)
#define BOOST_LOCKFREE_DETAIL_APPLE_PRIMITIVES_JUL_11_2008_0408PM

#if !defined(__APPLE__)
#error "do not include this file on non-Apple platforms"
#endif

#include <libkern/OSAtomic.h>
#include <bits/atomicity.h>

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
    template <class C, class D>
    inline bool CAS(volatile C * addr, D old, D nw)
    {
        return OSAtomicCompareAndSwap32((int32_t) old, (int32_t)nw, (int32_t*)addr);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <class C, class D, class E>
    inline bool CAS2(volatile C * addr, D old1, E old2, D new1, E new2)
    {
    // FIXME: for 32 bit processes on Intel hardware we can add a specialization 
    // using OSAtomicCompareAndSwap64

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


}}

#endif
