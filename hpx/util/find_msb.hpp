//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_FIND_MSB_JUN_23_2008_1004PM)
#define HPX_UTIL_FIND_MSB_JUN_23_2008_1004PM

#include <boost/cstdint.hpp>
#include <boost/mpl/integral_c.hpp>

namespace hpx { namespace util
{
    namespace detail
    {
        static char const log_table_256[] = { 
            0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
        };

        template <typename Bytes>
        struct find_msb;

        template <>
        struct find_msb<boost::mpl::int_<2> >
        {
            static int call(boost::uint16_t v)
            {
                // 16Bit size_t
                std::size_t t = v >> 8;
                return (t ? 8 + log_table_256[t] : log_table_256[v]);
            }
        };

        template <>
        struct find_msb<boost::mpl::int_<4> >
        {
            static int call(boost::uint32_t v)
            {
                // 32Bit size_t
                std::size_t t = v >> 16;
                if (t != 0) 
                    return 16 + find_msb<boost::mpl::int_<2> >::call(t);
                return find_msb<boost::mpl::int_<2> >::call(v);
            }
        };

        template <>
        struct find_msb<boost::mpl::int_<8> >
        {
            static int call(boost::uint64_t v)
            {
                // 64Bit size_t
                std::size_t t = v >> 32;
                if (t != 0)
                    return 32 + find_msb<boost::mpl::int_<4> >::call(t);
                return find_msb<boost::mpl::int_<4> >::call(v);
            }
        };
    }

    /// \brief  Find the position of the most significant bit in the given 
    ///         integer value. This is useful for determining the smallest 
    ///         number dividable by two which is still greater than a given 
    ///         number.
    inline int find_msb(std::size_t v)
    {
        if (1 == v)     // most common case (and most expensive to calculate)
            return 0;
        return detail::find_msb<boost::mpl::int_<sizeof(std::size_t)> >::call(v);
    }

    ///
    inline std::size_t find_msb_value(std::size_t value)
    {
        int i = find_msb(value);
        std::size_t msb = (0x01 << i);
        if (0 != (value & ~msb)) {
            // the most significant bit is not the only bit set
            return msb << 1;
        }
        return msb;
    }
}}

#endif
