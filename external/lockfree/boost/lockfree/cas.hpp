//  Copyright (C) 2007, 2008 Tim Blechmann & Thomas Grill
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//  Disclaimer: Not a Boost library.

#ifndef BOOST_LOCKFREE_CAS_HPP_INCLUDED
#define BOOST_LOCKFREE_CAS_HPP_INCLUDED

#include <boost/noncopyable.hpp>
#include <boost/thread/once.hpp>
#include <boost/call_traits.hpp>
#include <boost/aligned_storage.hpp>
#include <boost/type_traits/add_pointer.hpp>
#include <boost/type_traits/alignment_of.hpp>

#include <boost/lockfree/prefix.hpp>
#include <boost/detail/lightweight_mutex.hpp>

#if defined(_MSC_VER) && (_MSC_VER >= 1300)
#include <intrin.h>
#pragma intrinsic(_ReadWriteBarrier)
#pragma intrinsic(_InterlockedCompareExchange)
#endif

namespace boost
{
namespace lockfree
{

inline void memory_barrier()
{
#if defined(__GNUC__) && ( (__GNUC__ > 4) || ((__GNUC__ >= 4) && (__GNUC_MINOR__ >= 1)) )
    __sync_synchronize();
#elif defined(_MSC_VER) && (_MSC_VER >= 1300)
    _ReadWriteBarrier();
#elif defined(__APPLE__)
    OSMemoryBarrier();
#elif defined(AO_HAVE_nop_full)
    AO_nop_full();
#else
#   warning "no memory barrier implemented for this platform"
#endif
}

template <class C, class D>
inline bool CAS(volatile C * addr, D old, D nw)
{
#if defined(__GNUC__) && ( (__GNUC__ > 4) || ((__GNUC__ >= 4) && (__GNUC_MINOR__ >= 1)) )
    return __sync_bool_compare_and_swap(addr, old, nw);
#elif defined(_MSC_VER)
    return _InterlockedCompareExchange(addr,old,nw) == old;
#elif defined(_WIN32)
    return InterlockedCompareExchange(addr,old,nw) == old;
#elif defined(__APPLE__)
    return OSAtomicCompareAndSwap32((int32_t) old, (int32_t)nw, (int32_t*)addr);
#elif defined(AO_HAVE_compare_and_swap_full)
    return AO_compare_and_swap_full(reinterpret_cast<volatile AO_t*>(addr),
                                    reinterpret_cast<AO_t>(old),
                                    reinterpret_cast<AO_t>(nw));
#else
#warning ("blocking cas emulation")

    boost::detail::lightweight_mutex guard;
    boost::detail::lightweight_mutex::scoped_lock lock(guard);

    if (*addr == old)
    {
        *addr = nw;
        return true;
    }
#endif
}

#if defined(_MSC_VER) && (defined(_M_IA64) || defined(_WIN64))
#if defined(HAS_CMPXCHG16B)
extern "C" bool CAS2_windows64(volatile __int64* addr, 
    __int64 old1, __int64 old2, __int64 new1, __int64 new2) throw();
#else

namespace detail
{
    template <class T, class Tag>
    struct static_ : boost::noncopyable
    {
        typedef T value_type;
        typedef typename boost::call_traits<T>::reference reference;
        typedef typename boost::call_traits<T>::const_reference const_reference;

    private:
        struct destructor
        {
            ~destructor()
            {
                static_::get_address()->~value_type();
            }
        };

        struct default_ctor
        {
            static void construct()
            {
                ::new (static_::get_address()) value_type();
                static destructor d;
            }
        };
        
    public:
        static_(Tag = Tag())
        {
            boost::call_once(&default_ctor::construct, constructed_);
        }

        operator reference()
        {
            return this->get();
        }

        operator const_reference() const
        {
            return this->get();
        }

        reference get()
        {
            return *this->get_address();
        }

        const_reference get() const
        {
            return *this->get_address();
        }

    private:
        typedef typename boost::add_pointer<value_type>::type pointer;

        static pointer get_address()
        {
            return static_cast<pointer>(data_.address());
        }

        typedef boost::aligned_storage<sizeof(value_type),
            boost::alignment_of<value_type>::value> storage_type;

        static storage_type data_;
        static boost::once_flag constructed_;
    };

    template <class T, class Tag>
    typename static_<T, Tag>::storage_type static_<T, Tag>::data_;

    template <class T, class Tag>
    boost::once_flag static_<T, Tag>::constructed_ = BOOST_ONCE_INIT;
}

struct lw_mutex_tag {};

inline boost::detail::lightweight_mutex& get_mutex()
{
    static detail::static_<boost::detail::lightweight_mutex, lw_mutex_tag> mtx;
    return mtx;
}

#endif
#endif

template <class C, class D, class E>
inline bool CAS2(volatile C * addr, D old1, E old2, D new1, E new2)
{
#if defined(__GNUC__) && ((__GNUC__ >  4) || ( (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 2) ) ) && defined(__i386__) && \
    (defined(__i686__) || defined(__pentiumpro__) || defined(__nocona__ ) || \
     defined (__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8))

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

    return __sync_bool_compare_and_swap_8(reinterpret_cast<volatile long long*>(addr),
                                          old.l,
                                          nw.l);
#elif defined(_MSC_VER)
#if defined(_M_IA64) || defined(_WIN64)
// early AMD processors do not support the cmpxchg16b instruction
#if defined(HAS_CMPXCHG16B)

    return CAS2_windows64(reinterpret_cast<volatile __int64*>(addr), 
        (__int64)old1, (__int64)old2, (__int64)new1, (__int64)new2);

#else

    struct packed_c
    {
        D d;
        E e;
    };

    volatile packed_c * packed_addr = reinterpret_cast<volatile packed_c*>(addr);

    boost::detail::lightweight_mutex::scoped_lock lock(get_mutex());

    if (packed_addr->d == old1 &&
        packed_addr->e == old2)
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
        mov eax,[old1]
        mov edx,[old2]
        mov ebx,[new1]
        mov ecx,[new2]
        mov edi,[addr]
        lock cmpxchg8b [edi]
        setz [ok]
    }
    return ok;
#endif
#elif defined(__GNUC__) && (defined(__i686__) || defined(__pentiumpro__)  || defined(__nocona__ ))
    char result;
#ifndef __PIC__
    __asm__ __volatile__("lock; cmpxchg8b %0; setz %1"
                         : "=m"(*addr), "=q"(result)
                         : "m"(*addr), "d" (old1), "a" (old2),
                           "c" (new1), "b" (new2) : "memory");
#else
    __asm__ __volatile__("push %%ebx; movl %6,%%ebx; lock; cmpxchg8b %0; setz %1; pop %%ebx"
                         : "=m"(*addr), "=q"(result)
                         : "m"(*addr), "d" (old1), "a" (old2),
                           "c" (new1), "m" (new2) : "memory");
#endif
    return result != 0;
#elif defined(AO_HAVE_double_compare_and_swap_full)
    if (sizeof(D) != sizeof(AO_t) || sizeof(E) != sizeof(AO_t)) {
        assert(false);
        return false;
    }

    return AO_compare_double_and_swap_double_full(
        reinterpret_cast<volatile AO_double_t*>(addr),
        static_cast<AO_t>(old2),
        reinterpret_cast<AO_t>(old1),
        static_cast<AO_t>(new2),
        reinterpret_cast<AO_t>(new1)
        );

#elif defined(__GNUC__) && defined(__x86_64__) &&                       \
    ( __GCC_HAVE_SYNC_COMPARE_AND_SWAP_16 ) ||                          \
    ( (__GNUC__ >  4) || ( (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 2) ) && defined(__nocona__ ))

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

    return __sync_bool_compare_and_swap_16(reinterpret_cast<volatile TItype*>(addr),
                                           old.l,
                                           nw.l);

#elif defined(__GNUC__) && defined(__x86_64__)
    /* handcoded asm, will crash on early amd processors */
    char result;
    __asm__ __volatile__("lock; cmpxchg16b %0; setz %1"
                         : "=m"(*addr), "=q"(result)
                         : "m"(*addr), "d" (old2), "a" (old1),
                           "c" (new2), "b" (new1) : "memory");
    return result != 0;
#else
#warning ("blocking CAS2 emulation")
    struct packed_c
    {
        D d;
        E e;
    };

    volatile packed_c * packed_addr = reinterpret_cast<volatile packed_c*>(addr);

    static boost::detail::lightweight_mutex guard;
    boost::detail::lightweight_mutex::scoped_lock lock(guard);

    if (packed_addr->d == old1 &&
        packed_addr->e == old2)
    {
        packed_addr->d = new1;
        packed_addr->e = new2;
        return true;
    }
    return false;
#endif
}

} /* namespace lockfree */
} /* namespace boost */

#endif /* BOOST_LOCKFREE_CAS_HPP_INCLUDED */
