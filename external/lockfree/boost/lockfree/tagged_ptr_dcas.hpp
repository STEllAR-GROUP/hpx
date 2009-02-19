//  tagged pointer, for aba prevention
//
//  Copyright (C) 2008 Tim Blechmann
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//  Disclaimer: Not a Boost library.

#ifndef BOOST_LOCKFREE_TAGGED_PTR_DCAS_HPP_INCLUDED
#define BOOST_LOCKFREE_TAGGED_PTR_DCAS_HPP_INCLUDED

#include <boost/lockfree/cas.hpp>
#include <boost/lockfree/branch_hints.hpp>

#include <cstddef>              /* for std::size_t */

# if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
#warning "using tagged_ptr_dcas"
#endif

namespace boost
{
namespace lockfree
{

template <class T>
class BOOST_LOCKFREE_DCAS_ALIGNMENT tagged_ptr
{
private:
    typedef boost::uint16_t flag_t;
    static const flag_t flag_mask = 1;

    static T* pack_ptr(T * ptr, bool flag)
    {
        BOOST_ASSERT(0 == (std::size_t(ptr) & flag_mask));
        return flag ? (T*)(std::size_t(ptr) | flag_mask) : ptr;
    }

    static T* extract_ptr(T* p)
    {
        return (T*)(std::size_t(p) & ~flag_mask);
    }

    static bool extract_flag(T const* i)
    {
        return (std::size_t(i) & flag_mask) ? true : false;
    }

public:
    typedef std::size_t tag_t;

    /** uninitialized constructor */
    tagged_ptr(void) 
      : ptr(0), tag(0)
    {}

    /** copy constructor */
    tagged_ptr(tagged_ptr const & p)//: ptr(0), tag(0)
    {
        set(p);
    }

    explicit tagged_ptr(T * p, tag_t t = 0)
      : ptr(p), tag(t)
    {}

    explicit tagged_ptr(T * p, tag_t t, bool f)
      : ptr(pack_ptr(p, f)), tag(t)
    {}

    /** atomic set operations */
    /* @{ */
    void operator= (tagged_ptr const & p)
    {
        atomic_set(p);
    }

    void atomic_set(tagged_ptr const & p)
    {
        for (;;)
        {
            tagged_ptr old;
            old.set(*this);

            if(likely(CAS(old, p.ptr, p.tag)))
                return;
        }
    }

    void atomic_set(T * p, tag_t t)
    {
        for (;;)
        {
            tagged_ptr old;
            old.set(*this);

            if(likely(CAS(old, p, t)))
                return;
        }
    }

    void atomic_set(T * p, tag_t t, bool f)
    {
        for (;;)
        {
            tagged_ptr old;
            old.set(*this);

            if(likely(CAS(old, pack_ptr(p, f), t)))
                return;
        }
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
        tag = p.tag;
    }

    void set(T * p, tag_t t)
    {
        ptr = p;
        tag = t;
    }

    void set(T * p, tag_t t, bool f)
    {
        ptr = pack_ptr(p, f);
        tag = t;
    }
    /* @} */

    /** comparing semantics */
    /* @{ */
    friend bool operator== (tagged_ptr const& rhs, tagged_ptr const& lhs)
    {
        return lhs.ptr == rhs.ptr && lhs.tag == rhs.tag;
    }
    friend bool operator!= (tagged_ptr const& rhs, tagged_ptr const& lhs)
    {
        return !(lhs == rhs);
    }
    friend bool operator== (tagged_ptr volatile const& rhs, tagged_ptr const& lhs)
    {
        return lhs.ptr == rhs.ptr && lhs.tag == rhs.tag;
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
        ptr = pack_ptr(p, false);
    }

    void set_ptr(T * p, bool f)
    {
        ptr = pack_ptr(p, f);
    }
    /* @} */

    /** tag access */
    /* @{ */
    tag_t get_tag() const
    {
        return tag;
    }

    void set_tag(tag_t t)
    {
        tag = t;
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
        ptr = pack_ptr(p, flag);
    }
    /* @} */

    /** compare and swap  */
    /* @{ */
    bool CAS(tagged_ptr const & oldval, T * newptr)
    {
        return boost::lockfree::CAS2(this, oldval.ptr, oldval.tag, newptr, oldval.tag + 1);
    }

    bool CAS(tagged_ptr const & oldval, T * newptr, tag_t t)
    {
        return boost::lockfree::CAS2(this, oldval.ptr, oldval.tag, newptr, t);
    }
    /* @} */

    /** smart pointer support  */
    /* @{ */
    T & operator*() const
    {
        return *ptr;
    }

    T * operator->() const
    {
        return ptr;
    }

    operator bool(void) const
    {
        return bool(NULL != ptr);
    }
    /* @} */

protected:
    T * ptr;
    tag_t tag;
};

} /* namespace lockfree */
} /* namespace boost */

#endif /* BOOST_LOCKFREE_TAGGED_PTR_DCAS_HPP_INCLUDED */
