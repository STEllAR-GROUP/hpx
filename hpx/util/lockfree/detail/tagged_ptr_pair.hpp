//  tagged pointer pair, for aba prevention (intended for use with 128bit
//  atomics)
//
//  Copyright (C) 2008-2011 Tim Blechmann
//  Copyright (C) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//  Disclaimer: Not a Boost library.

#ifndef BOOST_LOCKFREE_5C0B91C3_F812_4DA3_A9B7_7F39440555EB
#define BOOST_LOCKFREE_5C0B91C3_F812_4DA3_A9B7_7F39440555EB

#include <hpx/config.hpp>

#include <boost/lockfree/detail/prefix.hpp>

#include <cstddef> // for std::size_t
#include <cstdint>

namespace boost { namespace lockfree
{

#if defined(BOOST_ATOMIC_HAVE_GNU_128BIT_INTEGERS)
template <class Left, class Right>
struct BOOST_LOCKFREE_DCAS_ALIGNMENT tagged_ptr_pair
{
    typedef __uint128_t compressed_ptr_pair_t;
    typedef std::uint64_t compressed_ptr_t;
    typedef std::uint16_t tag_t;

    union BOOST_LOCKFREE_DCAS_ALIGNMENT cast_unit
    {
        compressed_ptr_pair_t value;
        compressed_ptr_t ptrs[2];
        tag_t tags[8];
    };

    HPX_STATIC_CONSTEXPR std::size_t left_tag_index = 3;
    HPX_STATIC_CONSTEXPR std::size_t right_tag_index = 7;
    HPX_STATIC_CONSTEXPR std::size_t left_ptr_index = 0;
    HPX_STATIC_CONSTEXPR std::size_t right_ptr_index = 1;
    HPX_STATIC_CONSTEXPR compressed_ptr_t ptr_mask = 0xffffffffffff;

    static Left* extract_left_ptr(volatile compressed_ptr_pair_t const& i)
    {
        cast_unit cu;
        cu.value = i;
        return reinterpret_cast<Left*>(cu.ptrs[left_ptr_index] & ptr_mask);
    }

    static Right* extract_right_ptr(volatile compressed_ptr_pair_t const& i)
    {
        cast_unit cu;
        cu.value = i;
        return reinterpret_cast<Right*>(cu.ptrs[right_ptr_index] & ptr_mask);
    }

    static tag_t extract_left_tag(volatile compressed_ptr_pair_t const& i)
    {
        cast_unit cu;
        cu.value = i;
        return cu.tags[left_tag_index];
    }

    static tag_t
    extract_right_tag(volatile compressed_ptr_pair_t const& i)
    {
        cast_unit cu;
        cu.value = i;
        return cu.tags[right_tag_index];
    }

    template <typename IntegralL, typename IntegralR>
    static compressed_ptr_pair_t
    pack_ptr_pair(Left* lptr, Right* rptr, IntegralL ltag, IntegralR rtag)
    {
        cast_unit ret;
        ret.ptrs[left_ptr_index] = reinterpret_cast<compressed_ptr_t>(lptr);
        ret.ptrs[right_ptr_index] = reinterpret_cast<compressed_ptr_t>(rptr);
        ret.tags[left_tag_index] = static_cast<tag_t>(ltag);
        ret.tags[right_tag_index] = static_cast<tag_t>(rtag);
        return ret.value;
    }

    /** uninitialized constructor */
    tagged_ptr_pair(): pair_(0) {}

    template <typename IntegralL>
    tagged_ptr_pair(Left* lptr, Right* rptr, IntegralL ltag):
        pair_(pack_ptr_pair(lptr, rptr, ltag, 0)) {}

    template <typename IntegralL, typename IntegralR>
    tagged_ptr_pair(Left* lptr, Right* rptr, IntegralL ltag, IntegralR rtag):
        pair_(pack_ptr_pair(lptr, rptr, ltag, rtag)) {}

    /** copy constructors */
    tagged_ptr_pair(tagged_ptr_pair const& p): pair_(p.pair_) {}

    tagged_ptr_pair(Left* lptr, Right* rptr):
        pair_(pack_ptr_pair(lptr, rptr, 0, 0)) {}

    /** unsafe set operations */
    /* @{ */
    void operator= (tagged_ptr_pair const& p)
    { pair_ = p.pair_; }

    void set(Left* lptr, Right* rptr)
    { pair_ = pack_ptr_pair(lptr, rptr, 0, 0); }

    void reset(Left* lptr, Right* rptr)
    { set(lptr, rptr, 0, 0); }

    template <typename IntegralL>
    void set(Left* lptr, Right* rptr, IntegralL ltag)
    { pair_ = pack_ptr_pair(lptr, rptr, ltag, 0); }

    template <typename IntegralL, typename IntegralR>
    void set(Left* lptr, Right* rptr, IntegralL ltag, IntegralR rtag)
    { pair_ = pack_ptr_pair(lptr, rptr, ltag, rtag); }

    template <typename IntegralL>
    void reset(Left* lptr, Right* rptr, IntegralL ltag)
    { set(lptr, rptr, ltag, 0); }

    template <typename IntegralL, typename IntegralR>
    void reset(Left* lptr, Right* rptr, IntegralL ltag, IntegralR rtag)
    { set(lptr, rptr, ltag, rtag); }
    /* @} */

    /** comparing semantics */
    /* @{ */
    bool operator== (volatile tagged_ptr_pair const& p) const
    { return (pair_ == p.pair_); }

    bool operator!= (volatile tagged_ptr_pair const& p) const
    { return !operator==(p); }
    /* @} */

    /** pointer access */
    /* @{ */
    Left* get_left_ptr() const volatile
    { return extract_left_ptr(pair_); }

    Right* get_right_ptr() const volatile
    { return extract_right_ptr(pair_); }

    void set_left_ptr(Left* lptr) volatile
    {
        Right* rptr = get_right_ptr();
        tag_t ltag = get_left_tag();
        tag_t rtag = get_right_tag();
        pair_ = pack_ptr_pair(lptr, rptr, ltag, rtag);
    }

    void set_right_ptr(Right* rptr) volatile
    {
        Left* lptr = get_left_ptr();
        tag_t ltag = get_left_tag();
        tag_t rtag = get_right_tag();
        pair_ = pack_ptr_pair(lptr, rptr, ltag, rtag);
    }
    /* @} */

    /** tag access */
    /* @{ */
    tag_t get_left_tag() const volatile
    { return extract_left_tag(pair_); }

    tag_t get_right_tag() const volatile
    { return extract_right_tag(pair_); }

    template <typename Integral>
    void set_left_tag(Integral ltag) volatile
    {
        Left* lptr = get_left_ptr();
        Right* rptr = get_right_ptr();
        tag_t rtag = get_right_tag();
        pair_ = pack_ptr_pair(lptr, rptr, ltag, rtag);
    }

    template <typename Integral>
    void set_right_tag(Integral rtag) volatile
    {
        Left* lptr = get_left_ptr();
        Right* rptr = get_right_ptr();
        tag_t ltag = get_left_tag();
        pair_ = pack_ptr_pair(lptr, rptr, ltag, rtag);
    }
    /* @} */

    /** smart pointer support  */
    /* @{ */
    operator bool() const
    { return (get_left_ptr() != 0) && (get_right_ptr() != 0); }
    /* @} */

  private:
    compressed_ptr_pair_t pair_;
};
#else
struct BOOST_LOCKFREE_DCAS_ALIGNMENT uint128_type
{
    std::uint64_t left;
    std::uint64_t right;

    bool operator==(volatile uint128_type const& rhs) const
    { return (left == rhs.left) && (right == rhs.right); }

    bool operator!=(volatile uint128_type const& rhs) const
    { return !(*this == rhs); }
};

template <class Left, class Right>
struct BOOST_LOCKFREE_DCAS_ALIGNMENT tagged_ptr_pair
{
    typedef uint128_type compressed_ptr_pair_t;
    typedef std::uint64_t compressed_ptr_t;
    typedef std::uint16_t tag_t;

    union BOOST_LOCKFREE_DCAS_ALIGNMENT cast_unit
    {
        compressed_ptr_pair_t value;
        tag_t tags[8];
    };

    HPX_STATIC_CONSTEXPR std::size_t left_tag_index = 3;
    HPX_STATIC_CONSTEXPR std::size_t right_tag_index = 7;
    HPX_STATIC_CONSTEXPR compressed_ptr_t ptr_mask = 0xffffffffffff;

    static Left* extract_left_ptr(volatile compressed_ptr_pair_t const& i)
    { return reinterpret_cast<Left*>(i.left & ptr_mask); }

    static Right* extract_right_ptr(volatile compressed_ptr_pair_t const& i)
    { return reinterpret_cast<Right*>(i.right & ptr_mask); }

    static tag_t extract_left_tag(volatile compressed_ptr_pair_t const& i)
    {
        cast_unit cu;
        cu.value.left = i.left;
        cu.value.right = i.right;
        return cu.tags[left_tag_index];
    }

    static tag_t
    extract_right_tag(volatile compressed_ptr_pair_t const& i)
    {
        cast_unit cu;
        cu.value.left = i.left;
        cu.value.right = i.right;
        return cu.tags[right_tag_index];
    }

    template <typename IntegralL, typename IntegralR>
    static void
    pack_ptr_pair(compressed_ptr_pair_t& pair,
                  Left* lptr, Right* rptr, IntegralL ltag, IntegralR rtag)
    {
        cast_unit ret;
        ret.value.left = reinterpret_cast<compressed_ptr_t>(lptr);
        ret.value.right = reinterpret_cast<compressed_ptr_t>(rptr);
        ret.tags[left_tag_index] = static_cast<tag_t>(ltag);
        ret.tags[right_tag_index] = static_cast<tag_t>(rtag);
        pair = ret.value;
    }

    /** uninitialized constructor */
    tagged_ptr_pair() { pair_.left = 0; pair_.right = 0; }

    template <typename IntegralL>
    tagged_ptr_pair(Left* lptr, Right* rptr, IntegralL ltag)
    { pack_ptr_pair(pair_, lptr, rptr, ltag, 0); }

    template <typename IntegralL, typename IntegralR>
    tagged_ptr_pair(Left* lptr, Right* rptr, IntegralL ltag, IntegralR rtag)
    { pack_ptr_pair(pair_, lptr, rptr, ltag, rtag); }

    /** copy constructors */
    tagged_ptr_pair(tagged_ptr_pair const& p)
    {
        pair_.left = p.pair_.left;
        pair_.right = p.pair_.right;
    }

    tagged_ptr_pair(Left* lptr, Right* rptr)
    { pack_ptr_pair(pair_, lptr, rptr, 0, 0); }

    /** unsafe set operations */
    /* @{ */
    tagged_ptr_pair& operator= (tagged_ptr_pair const& p)
    {
        pair_.left = p.pair_.left;
        pair_.right = p.pair_.right;
        return *this;
    }

    void set(Left* lptr, Right* rptr)
    { pack_ptr_pair(pair_, lptr, rptr, 0, 0); }

    void reset(Left* lptr, Right* rptr)
    { set(lptr, rptr, 0, 0); }

    template <typename IntegralL>
    void set(Left* lptr, Right* rptr, IntegralL ltag)
    { pack_ptr_pair(pair_, lptr, rptr, ltag, 0); }

    template <typename IntegralL, typename IntegralR>
    void set(Left* lptr, Right* rptr, IntegralL ltag, IntegralR rtag)
    { pack_ptr_pair(pair_, lptr, rptr, ltag, rtag); }

    template <typename IntegralL>
    void reset(Left* lptr, Right* rptr, IntegralL ltag)
    { set(lptr, rptr, ltag, 0); }

    template <typename IntegralL, typename IntegralR>
    void reset(Left* lptr, Right* rptr, IntegralL ltag, IntegralR rtag)
    { set(lptr, rptr, ltag, rtag); }
    /* @} */

    /** comparing semantics */
    /* @{ */
    bool operator== (volatile tagged_ptr_pair const& p) const
    { return (pair_ == p.pair_); }

    bool operator!= (volatile tagged_ptr_pair const& p) const
    { return !operator==(p); }
    /* @} */

    /** pointer access */
    /* @{ */
    Left* get_left_ptr() const volatile
    { return extract_left_ptr(pair_); }

    Right* get_right_ptr() const volatile
    { return extract_right_ptr(pair_); }

    void set_left_ptr(Left* lptr) volatile
    {
        Right* rptr = get_right_ptr();
        tag_t ltag = get_left_tag();
        tag_t rtag = get_right_tag();
        pack_ptr_pair(pair_, lptr, rptr, ltag, rtag);
    }

    void set_right_ptr(Right* rptr) volatile
    {
        Left* lptr = get_left_ptr();
        tag_t ltag = get_left_tag();
        tag_t rtag = get_right_tag();
        pack_ptr_pair(pair_, lptr, rptr, ltag, rtag);
    }
    /* @} */

    /** tag access */
    /* @{ */
    tag_t get_left_tag() const volatile
    { return extract_left_tag(pair_); }

    tag_t get_right_tag() const volatile
    { return extract_right_tag(pair_); }

    template <typename Integral>
    void set_left_tag(Integral ltag) volatile
    {
        Left* lptr = get_left_ptr();
        Right* rptr = get_right_ptr();
        tag_t rtag = get_right_tag();
        pack_ptr_pair(pair_, lptr, rptr, ltag, rtag);
    }

    template <typename Integral>
    void set_right_tag(Integral rtag) volatile
    {
        Left* lptr = get_left_ptr();
        Right* rptr = get_right_ptr();
        tag_t ltag = get_left_tag();
        pack_ptr_pair(pair_, lptr, rptr, ltag, rtag);
    }
    /* @} */

    /** smart pointer support  */
    /* @{ */
    operator bool() const
    { return (get_left_ptr() != 0) && (get_right_ptr() != 0); }
    /* @} */

  private:
    compressed_ptr_pair_t pair_;
};
#endif

}}

#endif // BOOST_LOCKFREE_5C0B91C3_F812_4DA3_A9B7_7F39440555EB

