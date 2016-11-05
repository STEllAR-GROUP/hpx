//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2016 Andreas Schaefer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DATAPAR_LIBFLATARRAY_COUNT_BITS_SEP_22_2016_0220PM)
#define HPX_PARALLEL_DATAPAR_LIBFLATARRAY_COUNT_BITS_SEP_22_2016_0220PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_LIBFLATARRAY)
#include <cstddef>

#include <libflatarray/flat_array.hpp>

namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        HPX_HOST_DEVICE HPX_FORCEINLINE
        unsigned char get_num_bits(unsigned char mask)
        {
            static unsigned char const numbits_in_byte[] =
            {
            //  0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            // 10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f
                1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
            // 20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f
                1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
            // 30 31 32 33 34 35 36 37 38 39 3a 3b 3c 3d 3e 3f
                2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            // 40 41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f
                1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
            // 50 51 52 53 54 55 56 57 58 59 5a 5b 5c 5d 5e 5f
                2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            // 60 61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f
                2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            // 70 71 72 73 74 75 76 77 78 79 7a 7b 7c 7d 7e 7f
                3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            // 80 81 82 83 84 85 86 87 88 89 8a 8b 8c 8d 8e 8f
                1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
            // 90 91 92 93 94 95 96 97 98 99 9a 9b 9c 9d 9e 9f
                2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            // a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 aa ab ac ad ae af
                2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            // b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 ba bb bc bd be bf
                3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            // c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 ca cb cc cd ce cf
                2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            // d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 da db dc dd de df
                3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            // e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 ea eb ec ed ee ef
                3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            // f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 fa fb fc fd fe ff
                4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
            };

            return numbits_in_byte[mask];
        }

        template <typename T>
        HPX_HOST_DEVICE HPX_FORCEINLINE std::size_t get_num_bits(T mask)
        {
            unsigned char count = 0;
            for (int i = 0; mask != 0 && i != sizeof(mask); ++i)
            {
                count += get_num_bits(static_cast<unsigned char>(mask & 0xff));
                mask >>= 8;
            }
            return count;
        }
    }

    HPX_HOST_DEVICE HPX_FORCEINLINE
    std::size_t count_bits(unsigned char mask)
    {
        return detail::get_num_bits(mask);
    }

    HPX_HOST_DEVICE HPX_FORCEINLINE
    std::size_t count_bits(unsigned short mask)
    {
        return detail::get_num_bits(mask);
    }

    HPX_HOST_DEVICE HPX_FORCEINLINE
    std::size_t count_bits(unsigned mask)
    {
        return detail::get_num_bits(mask);
    }
}}}

#endif
#endif

