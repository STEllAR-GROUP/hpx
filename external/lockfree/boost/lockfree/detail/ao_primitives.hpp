//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(BOOST_LOCKFREE_DETAIL_AO_PRIMITIVES_JUL_11_2008_0423PM)
#define BOOST_LOCKFREE_DETAIL_AO_PRIMITIVES_JUL_11_2008_0423PM

#define AO_REQUIRE_CAS
#define AO_USE_PENTIUM4_INSTRS

extern "C" {
    #include <libatomic_ops/src/atomic_ops.h>
}

#define BOOST_LOCKFREE_DCAS_ALIGNMENT
#define BOOST_LOCKFREE_CACHELINE_ALIGNMENT
#define BOOST_LOCKFREE_CACHELINE_ALIGNMENT_PREFIX

namespace boost { lockfree
{
    ///////////////////////////////////////////////////////////////////////////
    inline void memory_barrier()
    {
#if defined(AO_HAVE_nop_full)
        AO_nop_full();
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
        return 0;     // no timings are recorded
    }

    ///////////////////////////////////////////////////////////////////////////
    template <class C, class D>
    inline bool CAS (volatile C * addr, D old, D nw)
    {
#if defined(AO_HAVE_compare_and_swap_full)
        return AO_compare_and_swap_full(reinterpret_cast<volatile AO_t*>(addr),
            reinterpret_cast<AO_t>(old), reinterpret_cast<AO_t>(nw));
#else
#warning ("blocking cas emulation")
        boost::detail::lightweight_mutex::scoped_lock lock(detail::get_CAS_mutex());
        if (*addr == old)
        {
            *addr = nw;
            return true;
        }
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    template <class C, class D, class E>
    inline bool CAS2(volatile C * addr, D old1, E old2, D new1, E new2)
    {
#if defined(AO_HAVE_double_compare_and_swap_full)
        if (sizeof(D) != sizeof(AO_t) || sizeof(E) != sizeof(AO_t)) {
            BOOST_ASSERT(false);
            return false;
        }

        return AO_compare_double_and_swap_double_full(
            reinterpret_cast<volatile AO_double_t*>(addr),
            static_cast<AO_t>(old2), reinterpret_cast<AO_t>(old1),
            static_cast<AO_t>(new2), reinterpret_cast<AO_t>(new1));
#else
#warning ("blocking CAS2 emulation")
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
    }

    ///////////////////////////////////////////////////////////////////////////
    template <class C, class D>
    inline D interlocked_compare_exchange(volatile C * addr, D old, D nw)
    {
        if (AO_compare_and_swap_full(reinterpret_cast<volatile AO_t*>(addr),
            reinterpret_cast<AO_t>(old), reinterpret_cast<AO_t>(nw)))
        {
            return old;
        }
        return *addr;
    }

    template <typename T>
    inline bool interlocked_bit_test_and_set(T* x, T bit)
    {
        T const value = 1u << bit;
        T old = *x;
        do {
            T const current = interlocked_compare_exchange(x, old, T(old | value));
            if (current == old)
            {
                break;
            }
            old = current;
        } while(true);
        return (old & value) != 0;
    }

    template <typename T>
    inline bool interlocked_bit_test_and_reset(T* x, T bit)
    {
        T const value = 1u << bit;
        T old = *x;
        do {
            T const current = interlocked_compare_exchange(x, old, T(old & ~value));
            if (current == old)
            {
                break;
            }
            old = current;
        } while(true);
        return (old & value) != 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    inline T interlocked_decrement(T* value)
    {
        for(;;)
        {
            T oldv = *value;
            if(likely(CAS(value, oldv, oldv-1)))
                return oldv;
        }
    }

    template <typename T>
    inline T interlocked_increment(T* value)
    {
        for(;;)
        {
            T oldv = *value;
            if(likely(CAS(value, oldv, oldv+1)))
                return oldv;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    inline T interlocked_exchange_sub(T* value, T sub)
    {
        return AO_fetch_and_add_full(value, -sub);
    }

    template <typename T>
    inline T interlocked_exchange_add(T* value, T add)
    {
        return AO_fetch_and_add_full(value, add);
    }

}}

#endif
