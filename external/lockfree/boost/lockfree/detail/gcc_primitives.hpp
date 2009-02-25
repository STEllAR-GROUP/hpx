//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(BOOST_LOCKFREE_DETAIL_GCC_PRIMITIVES_JUL_11_2008_0408PM)
#define BOOST_LOCKFREE_DETAIL_GCC_PRIMITIVES_JUL_11_2008_0408PM

#if !defined(__GNUC__)
#error "do not include this file on non-gcc platforms"
#endif

#define BOOST_LOCKFREE_CACHELINE_ALIGNMENT __attribute__((aligned(64)))
#define BOOST_LOCKFREE_CACHELINE_ALIGNMENT_PREFIX 

#ifdef __i386__
    #define BOOST_LOCKFREE_DCAS_ALIGNMENT
#elif defined(__ppc__)
    #define BOOST_LOCKFREE_DCAS_ALIGNMENT
#elif defined(__x86_64__)
//     #if !(defined (__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16) || defined (__nocona__))
//         #define BOOST_LOCKFREE_PTR_COMPRESSION 1
//     #endif
    #define BOOST_LOCKFREE_DCAS_ALIGNMENT __attribute__((aligned(16)))
#endif

namespace boost { namespace lockfree
{
    ///////////////////////////////////////////////////////////////////////////
    inline void memory_barrier()
    {
#if defined(__GNUC__) && ( (__GNUC__ > 4) || ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 1)) )
        __sync_synchronize();
#else
#warning "no memory barrier implemented for this platform"
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    inline void spin(unsigned char i)
    {
//#if defined(__i386__) || defined(__x86_64__)
//       asm volatile(
//            "l%=: pause;\n\t"
//            "dec %0;\n\t"
//            "jnz l%=;\n\t"
//            : /**/
//            : "r" (i)
//            : "cc"
//        );
//#else
        // do nothing
//#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    inline boost::uint64_t hrtimer_ticks()
    {
#if  defined(__i386__) || defined(__i686__) || defined(__pentiumpro__) || defined(__nocona__ ) 
        boost::uint32_t _lo, _hi;
        __asm__ __volatile__ (
              "movl %%ebx,%%esi\n"
              "cpuid\n"
              "rdtsc\n"
              "movl %%esi,%%ebx\n"
            : "=a" (_lo), "=d" (_hi)
            :
            : "%esi", "%ecx"
        );
        return ((boost::uint64_t)_hi << 32) | _lo;
#elif defined(__x86_64__)
        boost::uint32_t _lo, _hi;
        __asm__ __volatile__ (
              "mov %%rbx,%%rsi\n"
              "cpuid\n"
              "rdtsc\n"
              "mov %%rsi,%%rbx\n"
            : "=a" (_lo), "=d" (_hi)
            :
            : "%rsi", "%rcx"
        );
        return ((boost::uint64_t)_hi << 32) | _lo;
#endif
        return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <class C, class D>
    inline bool CAS(volatile C * addr, D old, D nw)
    {
#if defined(__GNUC__) && ( (__GNUC__ > 4) || ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 1)) )
#if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
#warning "CAS: using __sync_bool_compare_and_swap"
#endif
        return __sync_bool_compare_and_swap(addr, old, nw);
#else
#if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
#warning "CAS: blocking cas emulation"
#endif

        boost::detail::lightweight_mutex::scoped_lock lock(detail::get_CAS_mutex());
        if (*addr == old)
        {
            *addr = nw;
            return true;
        }
        return false;
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    template <class C, class D, class E>
    inline bool CAS2(volatile C * addr, D old1, E old2, D new1, E new2)
    {
#if ((__GNUC__ >  4) || ( (__GNUC__ == 4) && (__GNUC_MINOR__ >= 2) ) ) && defined(__i386__) && \
        (defined(__i686__) || defined(__pentiumpro__) || defined(__nocona__ ) || \
        defined (__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8))

#if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
#warning "CAS2: using __sync_bool_compare_and_swap_8"
#endif

        struct packed_c
        {
            D d;
            E e;
        };

        union cu
        {
            packed_c c;
            long long l;
        };

        cu old;
        old.c.d = old1;
        old.c.e = old2;

        cu nw;
        nw.c.d = new1;
        nw.c.e = new2;

        return __sync_bool_compare_and_swap_8(
            reinterpret_cast<volatile long long*>(addr), old.l, nw.l);

#elif defined(__i686__) || defined(__pentiumpro__)  || defined(__nocona__ )

        char result;
# ifndef __PIC__
# if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
# warning "CAS2: using 32Bit hand coded asm (__PIC__)"
# endif
        __asm__ __volatile__("lock; cmpxchg8b %0; setz %1"
                             : "=m"(*addr), "=q"(result)
                             : "m"(*addr), "d" (old1), "a" (old2),
                               "c" (new1), "b" (new2) : "memory");
# else
# if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
# warning "CAS2: using 32Bit hand coded asm"
# endif
        __asm__ __volatile__("push %%ebx; movl %6,%%ebx; lock; cmpxchg8b %0; setz %1; pop %%ebx"
                             : "=m"(*addr), "=q"(result)
                             : "m"(*addr), "d" (old1), "a" (old2),
                               "c" (new1), "m" (new2) : "memory");
# endif
        return result != 0;

#elif defined(__x86_64__)

# if ( __GCC_HAVE_SYNC_COMPARE_AND_SWAP_16 ) || \
    ( (__GNUC__ >  4) || ( (__GNUC__ == 4) && (__GNUC_MINOR__ >= 2) ) && defined(__nocona__ ) )

# if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
# warning "CAS2: using __sync_bool_compare_and_swap_16"
# endif

        struct packed_c
        {
            long d;
            long e;
        };

        typedef int TItype __attribute__ ((mode (TI)));

        BOOST_STATIC_ASSERT(sizeof(packed_c) == sizeof(TItype));

        union cu
        {
            packed_c c;
            TItype l;
        };

        cu old;
        old.c.d = (long)old1;
        old.c.e = (long)old2;

        cu nw;
        nw.c.d = (long)new1;
        nw.c.e = (long)new2;

        return __sync_bool_compare_and_swap_16(
            reinterpret_cast<volatile TItype*>(addr), old.l, nw.l);

# elif defined(__nocona__ ) 

# if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
# warning "CAS2: 64Bit system: handcoded asm, will crash on early amd processors"
# endif

    // 64Bit system: handcoded asm, will crash on early amd processors 
        char result;
        __asm__ __volatile__("lock; cmpxchg16b %0; setz %1"
                             : "=m"(*addr), "=q"(result)
                             : "m"(*addr), "d" (old2), "a" (old1),
                               "c" (new2), "b" (new1) : "memory");
        return result != 0;

# else

# if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
# warning "CAS2: blocking CAS2 emulation"
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

// this is the only case where we need to use compressed pointers on 64Bit systems
#define BOOST_LOCKFREE_PTR_COMPRESSION 1

# endif

#else

# if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
# warning "CAS2: 32Bit system: handcoded asm"
# endif
    // 32Bit system: handcoded asm 
        char result;
        __asm__ __volatile__("lock; cmpxchg8b %0; setz %1"
                             : "=m"(*addr), "=q"(result)
                             : "m"(*addr), "d" (old2), "a" (old1),
                               "c" (new2), "b" (new1) : "memory");
        return result != 0;

#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    template <class C, class D>
    inline D interlocked_compare_exchange(volatile C * addr, D old, D nw)
    {
#if defined(__GNUC__) && ( (__GNUC__ > 4) || ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 1)) )
#if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
#warning "interlocked_compare_exchange: using __sync_val_compare_and_swap"
#endif
        return __sync_val_compare_and_swap(addr, old, nw);
#else
#if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
#warning "interlocked_compare_exchange: blocking emulation"
#endif

        boost::detail::lightweight_mutex::scoped_lock lock(detail::get_ilce_mutex());
        D retval = *addr;
        if (*addr == old)
        {
            *addr = nw;
        }
        return retval;
#endif
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
#if defined(__GNUC__) && ( (__GNUC__ > 4) || ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 1)) )
#if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
#warning "interlocked_decrement: using __sync_fetch_and_sub"
#endif
        return __sync_fetch_and_sub(value, 1);
#else
        for(;;)
        {
            T oldv = *value;
            if(likely(CAS(value, oldv, oldv-1)))
                return oldv;
        }
#endif
    }

    template <typename T>
    inline T interlocked_increment(T* value)
    {
#if defined(__GNUC__) && ( (__GNUC__ > 4) || ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 1)) )
#if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
#warning "interlocked_increment: using __sync_fetch_and_sub"
#endif
        return __sync_fetch_and_add(value, 1);
#else
        for(;;)
        {
            T oldv = *value;
            if(likely(CAS(value, oldv, oldv+1)))
                return oldv;
        }
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    inline T interlocked_exchange_sub(T* value, T sub)
    {
#if defined(__GNUC__) && ( (__GNUC__ > 4) || ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 1)) )
#if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
#warning "interlocked_exchange_sub: using __sync_fetch_and_sub"
#endif
        return __sync_fetch_and_add(value, sub);
#else
        for(;;)
        {
            T oldv = *value;
            if(likely(CAS(value, oldv, oldv-sub)))
                return oldv;
        }
#endif
    }

    template <typename T>
    inline T interlocked_exchange_add(T* value, T add)
    {
#if defined(__GNUC__) && ( (__GNUC__ > 4) || ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 1)) )
#if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
#warning "interlocked_exchange_add: using __sync_fetch_and_add"
#endif
        return __sync_fetch_and_add(value, add);
#else
        for(;;)
        {
            T oldv = *value;
            if(likely(CAS(value, oldv, oldv+add)))
                return oldv;
        }
#endif
    }

}}

#endif
