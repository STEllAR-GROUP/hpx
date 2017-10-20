////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2012-2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#ifndef HPX_RUNTIME_THREADS_CPU_MASK_HPP
#define HPX_RUNTIME_THREADS_CPU_MASK_HPP

#include <hpx/config.hpp>
#include <hpx/util/assert.hpp>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <string>

#if defined(HPX_HAVE_MORE_THAN_64_THREADS) || (defined(HPX_HAVE_MAX_CPU_COUNT) \
            && HPX_HAVE_MAX_CPU_COUNT > 64)
#  if defined(HPX_HAVE_MAX_CPU_COUNT)
#    include <bitset>
#  else
#    include <boost/dynamic_bitset.hpp>
#  endif
#endif

namespace hpx { namespace threads
{
    /// \cond NOINTERNAL
#if !defined(HPX_HAVE_MORE_THAN_64_THREADS) || (defined(HPX_HAVE_MAX_CPU_COUNT) \
             && HPX_HAVE_MAX_CPU_COUNT <= 64)
    typedef std::uint64_t mask_type;
    typedef std::uint64_t mask_cref_type;

    inline std::uint64_t bits(std::size_t idx)
    {
       HPX_ASSERT(idx < CHAR_BIT * sizeof(mask_type));
       return std::uint64_t(1) << idx;
    }

    inline bool any(mask_cref_type mask)
    {
        return mask != 0;
    }

    inline mask_type not_(mask_cref_type mask)
    {
        return ~mask;
    }

    inline bool test(mask_cref_type mask, std::size_t idx)
    {
        HPX_ASSERT(idx < CHAR_BIT * sizeof(mask_type));
        return (bits(idx) & mask) != 0;
    }

    inline void set(mask_type& mask, std::size_t idx)
    {
        HPX_ASSERT(idx < CHAR_BIT * sizeof(mask_type));
        mask |= bits(idx);
    }

    inline void unset(mask_type& mask, std::size_t idx)
    {
        HPX_ASSERT(idx < CHAR_BIT * sizeof(mask_type));
        mask &= not_(bits(idx));
    }

    inline std::size_t mask_size(mask_cref_type mask)
    {
        return CHAR_BIT * sizeof(mask_type);
    }

    inline void resize(mask_type& mask, std::size_t s)
    {
        HPX_ASSERT(s <= CHAR_BIT * sizeof(mask_type));
    }

    inline std::size_t find_first(mask_cref_type mask)
    {
        if (mask) {
            std::size_t c = 0;    // Will count mask's trailing zero bits.

            // Set mask's trailing 0s to 1s and zero rest.
            mask = (mask ^ (mask - 1)) >> 1;
            for (/**/; mask; ++c)
                mask >>= 1;

            return c;
        }
        return ~std::size_t(0);
    }

    inline bool equal(mask_cref_type lhs, mask_cref_type rhs, std::size_t = 0)
    {
        return lhs == rhs;
    }

    // return true if at least one of the masks has a bit set
    inline bool bit_or(mask_cref_type lhs, mask_cref_type rhs, std::size_t = 0)
    {
        return (lhs | rhs) != 0;
    }

    // return true if at least one bit is set in both masks
    inline bool bit_and(mask_cref_type lhs, mask_cref_type rhs, std::size_t = 0)
    {
        return (lhs & rhs) != 0;
    }

    // returns the number of bits set
    // taken from https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
    inline std::size_t count(mask_cref_type mask)
    {
        std::size_t c; // c accumulates the total bits set in v
        for (c = 0; mask; c++)
        {
              mask &= mask - 1; // clear the least significant bit set
        }
        return c;
    }

    inline void reset(mask_type& mask)
    {
        mask = 0ull;
    }

#define HPX_CPU_MASK_PREFIX "0x"

#else
# if defined(HPX_HAVE_MAX_CPU_COUNT)
    typedef std::bitset<HPX_HAVE_MAX_CPU_COUNT> mask_type;
    typedef std::bitset<HPX_HAVE_MAX_CPU_COUNT> const& mask_cref_type;
# else
    typedef boost::dynamic_bitset<std::uint64_t> mask_type;
    typedef boost::dynamic_bitset<std::uint64_t> const& mask_cref_type;
# endif

    inline bool any(mask_cref_type mask)
    {
        return mask.any();
    }

    inline mask_type not_(mask_cref_type mask)
    {
        return ~mask;
    }

    inline bool test(mask_cref_type mask, std::size_t idx)
    {
        return mask.test(idx);
    }

    inline void set(mask_type& mask, std::size_t idx)
    {
        mask.set(idx);
    }

    inline std::size_t mask_size(mask_cref_type mask)
    {
        return mask.size();
    }

    inline void resize(mask_type& mask, std::size_t s)
    {
# if defined(HPX_HAVE_MAX_CPU_COUNT)
        HPX_ASSERT(s <= mask.size());
# else
        return mask.resize(s);
# endif
    }

    inline std::size_t find_first(mask_cref_type mask)
    {
# if defined(HPX_HAVE_MAX_CPU_COUNT)
        if (mask.any())
        {
            for (std::size_t i = 0; i != HPX_HAVE_MAX_CPU_COUNT; ++i)
            {
                if (mask[i])
                    return i;
            }
        }
        return ~std::size_t(0);
# else
        return mask.find_first();
# endif
    }

# if defined(HPX_HAVE_MAX_CPU_COUNT)
#define HPX_CPU_MASK_PREFIX "0b"
#else
#define HPX_CPU_MASK_PREFIX "0x"
#endif

    inline bool equal(mask_cref_type lhs, mask_cref_type rhs, std::size_t = 0)
    {
        return lhs == rhs;
    }

    // return true if at least one of the masks has a bit set
    inline bool bit_or(mask_cref_type lhs, mask_cref_type rhs, std::size_t = 0)
    {
        return (lhs | rhs).any();
    }

    // return true if at least one bit is set in both masks
    inline bool bit_and(mask_cref_type lhs, mask_cref_type rhs, std::size_t = 0)
    {
        return (lhs & rhs).any();
    }

    // returns the number of bits set
    inline std::size_t count(mask_cref_type mask)
    {
        return mask.count();
    }

    inline void reset(mask_type& mask)
    {
        mask.reset();
    }

#endif

    HPX_API_EXPORT std::string to_string(mask_cref_type);
    /// \endcond
}}

#endif /*HPX_RUNTIME_THREADS_CPU_MASK_HPP*/
