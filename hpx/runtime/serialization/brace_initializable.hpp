//  Copyright (c) 2019 Jan Melech
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_BRACE_INITIALIZABLE_HPP
#define HPX_SERIALIZATION_BRACE_INITIALIZABLE_HPP

#include <hpx/config/automatic_struct_serialization.hpp>
#include <hpx/runtime/serialization/brace_initializable_fwd.hpp>
#include <hpx/traits/brace_initializable_traits.hpp>

#include <tuple>

namespace hpx { namespace serialization
{
#if defined(HPX_SUPPORT_AUTOMATIC_STRUCT_SERIALIZATION)
    template <typename Archive, typename T, size_t ... I>
    void serialize_tuple(Archive& ar,
                         T& t,
                         std::index_sequence<I...>)
    {
        ((ar & std::get<I>(t)), ...);
    }

    template <typename Archive, typename ... T>
    void serialize_tuple(Archive& ar, const std::tuple<T...>& t)
    {
        serialize_tuple(ar, t, std::make_index_sequence<sizeof...(T)>{});
    }

    template <typename T, typename Archive>
    void serialize_struct(Archive& archive, T& t, hpx::traits::size<0>)
    {
    }

    template <typename T, typename Archive>
    void serialize_struct(Archive& archive, T& t, hpx::traits::size<1>)
    {
        auto& [p1] = t;
        serialize_tuple(archive, std::forward_as_tuple(p1));
    }

    template <typename T, typename Archive>
    void serialize_struct(Archive& archive, T& t, hpx::traits::size<2>)
    {
        auto& [p1, p2] = t;
        serialize_tuple(archive, std::forward_as_tuple(p1, p2));
    }

    template <typename T, typename Archive>
    void serialize_struct(Archive& archive, T& t, hpx::traits::size<3>)
    {
        auto& [p1, p2, p3] = t;
        serialize_tuple(archive, std::forward_as_tuple(p1, p2, p3));
    }

    template <typename T, typename Archive>
    void serialize_struct(Archive& archive, T& t, hpx::traits::size<4>)
    {
        auto& [p1, p2, p3, p4] = t;
        serialize_tuple(archive, std::forward_as_tuple(p1, p2, p3, p4));
    }

    template <typename T, typename Archive>
    void serialize_struct(Archive& archive, T& t, hpx::traits::size<5>)
    {
        auto& [p1, p2, p3, p4, p5] = t;
        serialize_tuple(archive, std::forward_as_tuple(p1, p2, p3, p4, p5));
    }

    template <typename T, typename Archive>
    void serialize_struct(Archive& archive, T& t, hpx::traits::size<6>)
    {
        auto& [p1, p2, p3, p4, p5, p6] = t;
        serialize_tuple(archive, std::forward_as_tuple(p1, p2, p3, p4, p5,
            p6));
    }

    template <typename T, typename Archive>
    void serialize_struct(Archive& archive, T& t, hpx::traits::size<7>)
    {
        auto& [p1, p2, p3, p4, p5, p6, p7] = t;
        serialize_tuple(archive, std::forward_as_tuple(p1, p2, p3, p4, p5,
            p6, p7));
    }

    template <typename T, typename Archive>
    void serialize_struct(Archive& archive, T& t, hpx::traits::size<8>)
    {
        auto& [p1, p2, p3, p4, p5, p6, p7, p8] = t;
        serialize_tuple(archive, std::forward_as_tuple(p1, p2, p3, p4, p5,
            p6, p7, p8));
    }

    template <typename T, typename Archive>
    void serialize_struct(Archive& archive, T& t, hpx::traits::size<9>)
    {
        auto& [p1, p2, p3, p4, p5, p6, p7, p8, p9] = t;
        serialize_tuple(archive, std::forward_as_tuple(p1, p2, p3, p4, p5,
            p6, p7, p8, p9));
    }

    template <typename T, typename Archive>
    void serialize_struct(Archive& archive, T& t, hpx::traits::size<10>)
    {
        auto& [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10] = t;
        serialize_tuple(archive, std::forward_as_tuple(p1, p2, p3, p4, p5,
            p6, p7, p8, p9, p10));
    }

    template <typename T, typename Archive>
    void serialize_struct(Archive& archive, T& t, hpx::traits::size<11>)
    {
        auto& [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11] = t;
        serialize_tuple(archive, std::forward_as_tuple(p1, p2, p3, p4, p5,
            p6, p7, p8, p9, p10, p11));
    }

    template <typename T, typename Archive>
    void serialize_struct(Archive& archive, T& t, hpx::traits::size<12>)
    {
        auto& [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12] = t;
        serialize_tuple(archive, std::forward_as_tuple(p1, p2, p3, p4, p5,
            p6, p7, p8, p9, p10, p11, p12));
    }

    template <typename T, typename Archive>
    void serialize_struct(Archive& archive, T& t, hpx::traits::size<13>)
    {
        auto& [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13] = t;
        serialize_tuple(archive, std::forward_as_tuple(p1, p2, p3, p4, p5,
            p6, p7, p8, p9, p10, p11, p12, p13));
    }

    template <typename T, typename Archive>
    void serialize_struct(Archive& archive, T& t, hpx::traits::size<14>)
    {
        auto& [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
            p14] = t;
        serialize_tuple(archive, std::forward_as_tuple(p1, p2, p3, p4, p5,
            p6, p7, p8, p9, p10, p11, p12, p13, p14));
    }

    template <typename T, typename Archive>
    void serialize_struct(Archive& archive, T& t, hpx::traits::size<15>)
    {
        auto& [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
            p14, p15] = t;
        serialize_tuple(archive, std::forward_as_tuple(p1, p2, p3, p4, p5,
            p6, p7, p8, p9, p10, p11, p12, p13, p14, p15));
    }

    template <typename Archive, typename T>
    void serialize_struct(Archive& ar, T& t, const unsigned int)
    {
        serialize_struct(ar, t, hpx::traits::arity<T>());
    }
#endif
}}

#endif
