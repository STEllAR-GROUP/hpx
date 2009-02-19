//  tagged pointer, for aba prevention
//
//  Copyright (C) 2008 Tim Blechmann, based on code by Cory Nelson
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//  Disclaimer: Not a Boost library.

#ifndef BOOST_LOCKFREE_TAGGED_PTR_PTRCOMPRESSION_HPP_INCLUDED
#define BOOST_LOCKFREE_TAGGED_PTR_PTRCOMPRESSION_HPP_INCLUDED

#include <boost/lockfree/cas.hpp>
#include <boost/lockfree/branch_hints.hpp>

#include <cstddef>              /* for std::size_t */

#include <boost/cstdint.hpp>
#include <boost/assert.hpp>

# if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
#warning "using tagged_ptr_ptrcompression"
#endif

namespace boost
{
namespace lockfree
{

#if defined(__x86_64__) || defined(_M_IA64) || defined(_WIN64)

template <class T>
class BOOST_LOCKFREE_DCAS_ALIGNMENT tagged_ptr
{
    typedef boost::uint64_t compressed_ptr_t;
    typedef boost::uint16_t tag_t;
    typedef boost::uint16_t flag_t;

private:
    union cast_unit
    {
        compressed_ptr_t value;
        tag_t tag[4];
    };

    static const int flag_index = 0;
    static const int tag_index = 3;
    static const compressed_ptr_t ptr_mask = (1LL << 48)-2;
    static const flag_t flag_mask = 1;

    static T* extract_ptr(compressed_ptr_t const & i)
    {
        return (T*)(i & ptr_mask);
    }

    static bool extract_flag(compressed_ptr_t const & i)
    {
        return (i & flag_mask) ? true : false;
    }

    static tag_t extract_tag(compressed_ptr_t const & i)
    {
        cast_unit cu;
        cu.value = i;
        return cu.tag[tag_index];
    }

    static compressed_ptr_t pack_ptr(T * ptr, int tag)
    {
        cast_unit ret;
        ret.value = compressed_ptr_t(ptr);
        BOOST_ASSERT(0 == (~ptr_mask & ret.value));
        ret.tag[tag_index] = tag;
        return ret.value;
    }

    static compressed_ptr_t pack_ptr(T * ptr, int tag, bool flag)
    {
        cast_unit ret;
        ret.value = compressed_ptr_t(ptr);
        BOOST_ASSERT(0 == (~ptr_mask & ret.value));
        ret.tag[tag_index] = tag;
        if (flag)
            ret.tag[flag_index] |= flag_mask;
        return ret.value;
    }

public:
    /** uninitialized constructor */
    tagged_ptr(void)//: ptr(0), tag(0)
    {}

    /** copy constructor */
    tagged_ptr(tagged_ptr const & p)//: ptr(0), tag(0)
    {
        set(p);
    }

    explicit tagged_ptr(T * p, tag_t t = 0)
      : ptr(pack_ptr(p, t))
    {}

    explicit tagged_ptr(T * p, tag_t t, bool f)
      : ptr(pack_ptr(p, t, f))
    {}

    /** atomic set operations */
    /* @{ */
    void operator= (tagged_ptr const & p)
    {
        atomic_set(p);
    }

    void atomic_set(tagged_ptr const & p)
    {
        set(p);
    }

    void atomic_set(T * p, tag_t t)
    {
        ptr = pack_ptr(p, t);
    }
    /* @} */

    friend tagged_ptr make_unique(tagged_ptr p)
    {
        for (;;)
        {
            tagged_ptr old;
            old.set(p);

            if(likely(p.CAS(old, p.ptr)))
                return p;
        }
    }

    /** unsafe set operation */
    /* @{ */
    void set(tagged_ptr const & p)
    {
        ptr = p.ptr;
    }

    void set(T * p, tag_t t)
    {
        ptr = pack_ptr(p, t);
    }

    void set(T * p, tag_t t, bool f)
    {
        ptr = pack_ptr(p, t, f);
    }
    /* @} */

    /** comparing semantics */
    /* @{ */
    friend bool operator== (tagged_ptr const& rhs, tagged_ptr const& lhs)
    {
        return lhs.ptr == rhs.ptr;
    }
    friend bool operator!= (tagged_ptr const& rhs, tagged_ptr const& lhs)
    {
        return !(lhs == rhs);
    }
    friend bool operator== (tagged_ptr volatile const& rhs, tagged_ptr const& lhs)
    {
        return lhs.ptr == rhs.ptr;
    }
    friend bool operator!= (tagged_ptr volatile const& rhs, tagged_ptr const& lhs)
    {
        return !(lhs == rhs);
    }
    /* @} */

    /** pointer access */
    /* @{ */
    T * get_ptr() const
    {
        return extract_ptr(ptr);
    }

    void set_ptr(T * p)
    {
        tag_t tag = get_tag();
        bool flag = get_flag();
        ptr = pack_ptr(p, tag, flag);
    }
    /* @} */

    /** tag access */
    /* @{ */
    tag_t get_tag() const
    {
        return extract_tag(ptr);
    }

    void set_tag(tag_t t)
    {
        T * p = get_ptr();
        bool flag = get_flag();
        ptr = pack_ptr(p, t, flag);
    }
    /* @} */

    /** flag access */
    /* @{ */
    bool get_flag() const
    {
        return extract_flag(ptr);
    }

    void set_flag(bool flag = true)
    {
        T * p = get_ptr();
        tag_t tag = get_tag();
        ptr = pack_ptr(p, t, flag);
    }
    /* @} */

    /** compare and swap  */
    /* @{ */
private:
    bool CAS(compressed_ptr_t const & oldval, compressed_ptr_t const & newval)
    {
        return boost::lockfree::CAS(&(this->ptr), oldval, newval);
    }

public:
    bool CAS(tagged_ptr const & oldval, T * newptr)
    {
        compressed_ptr_t new_compressed_ptr = pack_ptr(newptr, extract_tag(oldval.ptr)+1);
        return CAS(oldval.ptr, new_compressed_ptr);
    }

    bool CAS(tagged_ptr const & oldval, T * newptr, tag_t t)
    {
        compressed_ptr_t new_compressed_ptr = pack_ptr(newptr, t);
        return boost::lockfree::CAS(&(this->ptr), oldval.ptr, new_compressed_ptr);
    }
    /* @} */

    /** smart pointer support  */
    /* @{ */
    T & operator*() const
    {
        return *get_ptr();
    }

    T * operator->() const
    {
        return get_ptr();
    }

    operator bool(void) const
    {
        return bool (0 != get_ptr());
    }
    /* @} */

protected:
    compressed_ptr_t ptr;
};

#else
#error unsupported platform
#endif

} /* namespace lockfree */
} /* namespace boost */

#endif /* BOOST_LOCKFREE_TAGGED_PTR_PTRCOMPRESSION_HPP_INCLUDED */
