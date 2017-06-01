//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/components/partitioned_vector/detail/make_index_sequence.hpp

#ifndef HPX_PARTITIONED_VECTOR_DETAIL_MAKE_INDEX_SEQUENCE_HPP
#define HPX_PARTITIONED_VECTOR_DETAIL_MAKE_INDEX_SEQUENCE_HPP

namespace hpx {  namespace detail {

    // Unrolled recursive version of make_index_sequence

    template< class T, T... I>
    class integer_sequence
    {};

    template< std::size_t N,
        std::size_t Start = 0,
        class previous_sequence = integer_sequence<std::size_t>,
        bool = (N > 8)>
    struct make_index_sequence;

    template<std::size_t Start, std::size_t... I>
    struct make_index_sequence<0, Start,
        integer_sequence<std::size_t, I...>, false>
    {
        using type = integer_sequence<std::size_t, I...>;
    };

    template<std::size_t Start, std::size_t... I>
    struct make_index_sequence<1, Start,
        integer_sequence<std::size_t, I...>, false>
    {
        using type = integer_sequence<std::size_t, I..., Start>;
    };

    template<std::size_t Start, std::size_t... I>
    struct make_index_sequence<2, Start,
        integer_sequence<std::size_t, I...>, false>
    {
        using type = integer_sequence<std::size_t, I..., Start, Start+1>;
    };

    template<std::size_t Start, std::size_t... I>
    struct make_index_sequence<3, Start,
        integer_sequence<std::size_t, I...>, false>
    {
        using type
            = integer_sequence<std::size_t, I..., Start, Start+1, Start+2>;
    };

    template<std::size_t Start, std::size_t... I>
    struct make_index_sequence<4, Start,
        integer_sequence<std::size_t, I...>, false>
    {
        using type
            = integer_sequence<std::size_t, I..., Start, Start+1, Start+2,
                Start+3>;
    };

    template<std::size_t Start, std::size_t... I>
    struct make_index_sequence<5, Start,
        integer_sequence<std::size_t, I...>, false>
    {
        using type
            = integer_sequence<std::size_t, I..., Start, Start+1, Start+2,
                Start+3, Start+4>;
    };

    template<std::size_t Start, std::size_t... I>
    struct make_index_sequence<6, Start,
        integer_sequence<std::size_t, I...>, false>
    {
        using type
            = integer_sequence<std::size_t, I..., Start, Start+1, Start+2,
                Start+3, Start+4, Start+5>;
    };

    template<std::size_t Start, std::size_t... I>
    struct make_index_sequence<7, Start,
        integer_sequence<std::size_t, I...>, false>
    {
        using type
            = integer_sequence<std::size_t, I..., Start, Start+1, Start+2,
                Start+3, Start+4, Start+5, Start+6>;
    };

    template<std::size_t Start, std::size_t... I>
    struct make_index_sequence<8, Start,
        integer_sequence<std::size_t, I...>, false>
    {
        using type
            = integer_sequence<std::size_t, I..., Start, Start+1,Start+2,
                Start+3, Start+4, Start+5, Start+6, Start+7>;
    };

    template<std::size_t Start, std::size_t N, std::size_t... I>
    struct make_index_sequence<N, Start,
        integer_sequence<std::size_t, I...>, true>
    {
        using type
            = typename make_index_sequence<N-8, Start+8,
                integer_sequence<std::size_t, I..., Start, Start+1, Start+2,
                    Start+3, Start+4, Start+5, Start+6, Start+7>>::type;
    };

}}

#endif
