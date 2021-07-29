//  Copyright (c) 2019 Jan Melech
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <hpx/datastructures/tuple.hpp>
#include <hpx/serialization/brace_initializable_fwd.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/traits/brace_initializable_traits.hpp>
#include <hpx/serialization/tuple.hpp>

namespace hpx { namespace serialization {

    template <typename Archive, typename T>
    void serialize_struct(Archive& archive, T& t, const unsigned int version,
        hpx::traits::detail::size<0>)
    {
        serialize(archive, t, version);
    }

    template <typename Archive, typename T>
    void serialize_struct(Archive& archive, T& t, const unsigned int version,
        hpx::traits::detail::size<1>)
    {
        auto& [p1] = t;
        auto&& data = hpx::forward_as_tuple(p1);
        serialize(archive, data, version);
    }

    template <typename Archive, typename T>
    void serialize_struct(Archive& archive, T& t, const unsigned int version,
        hpx::traits::detail::size<2>)
    {
        auto& [p1, p2] = t;
        auto&& data = hpx::forward_as_tuple(p1, p2);
        serialize(archive, data, version);
    }

    template <typename Archive, typename T>
    void serialize_struct(Archive& archive, T& t, const unsigned int version,
        hpx::traits::detail::size<3>)
    {
        auto& [p1, p2, p3] = t;
        auto&& data = hpx::forward_as_tuple(p1, p2, p3);
        serialize(archive, data, version);
    }

    template <typename Archive, typename T>
    void serialize_struct(Archive& archive, T& t, const unsigned int version,
        hpx::traits::detail::size<4>)
    {
        auto& [p1, p2, p3, p4] = t;
        auto&& data = hpx::forward_as_tuple(p1, p2, p3, p4);
        serialize(archive, data, version);
    }

    template <typename Archive, typename T>
    void serialize_struct(Archive& archive, T& t, const unsigned int version,
        hpx::traits::detail::size<5>)
    {
        auto& [p1, p2, p3, p4, p5] = t;
        auto&& data = hpx::forward_as_tuple(p1, p2, p3, p4, p5);
        serialize(archive, data, version);
    }

    template <typename Archive, typename T>
    void serialize_struct(Archive& archive, T& t, const unsigned int version,
        hpx::traits::detail::size<6>)
    {
        auto& [p1, p2, p3, p4, p5, p6] = t;
        auto&& data = hpx::forward_as_tuple(p1, p2, p3, p4, p5, p6);
        serialize(archive, data, version);
    }

    template <typename Archive, typename T>
    void serialize_struct(Archive& archive, T& t, const unsigned int version,
        hpx::traits::detail::size<7>)
    {
        auto& [p1, p2, p3, p4, p5, p6, p7] = t;
        auto&& data = hpx::forward_as_tuple(p1, p2, p3, p4, p5, p6, p7);
        serialize(archive, data, version);
    }

    template <typename Archive, typename T>
    void serialize_struct(Archive& archive, T& t, const unsigned int version,
        hpx::traits::detail::size<8>)
    {
        auto& [p1, p2, p3, p4, p5, p6, p7, p8] = t;
        auto&& data = hpx::forward_as_tuple(p1, p2, p3, p4, p5, p6, p7, p8);
        serialize(archive, data, version);
    }

    template <typename Archive, typename T>
    void serialize_struct(Archive& archive, T& t, const unsigned int version,
        hpx::traits::detail::size<9>)
    {
        auto& [p1, p2, p3, p4, p5, p6, p7, p8, p9] = t;
        auto&& data = hpx::forward_as_tuple(p1, p2, p3, p4, p5, p6, p7, p8, p9);
        serialize(archive, data, version);
    }

    template <typename Archive, typename T>
    void serialize_struct(Archive& archive, T& t, const unsigned int version,
        hpx::traits::detail::size<10>)
    {
        auto& [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10] = t;
        auto&& data =
            hpx::forward_as_tuple(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);
        serialize(archive, data, version);
    }

    template <typename Archive, typename T>
    void serialize_struct(Archive& archive, T& t, const unsigned int version,
        hpx::traits::detail::size<11>)
    {
        auto& [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11] = t;
        auto&& data =
            hpx::forward_as_tuple(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11);
        serialize(archive, data, version);
    }

    template <typename Archive, typename T>
    void serialize_struct(Archive& archive, T& t, const unsigned int version,
        hpx::traits::detail::size<12>)
    {
        auto& [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12] = t;
        auto&& data = hpx::forward_as_tuple(
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12);
        serialize(archive, data, version);
    }

    template <typename Archive, typename T>
    void serialize_struct(Archive& archive, T& t, const unsigned int version,
        hpx::traits::detail::size<13>)
    {
        auto& [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13] = t;
        auto&& data = hpx::forward_as_tuple(
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);
        serialize(archive, data, version);
    }

    template <typename Archive, typename T>
    void serialize_struct(Archive& archive, T& t, const unsigned int version,
        hpx::traits::detail::size<14>)
    {
        auto& [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14] = t;
        auto&& data = hpx::forward_as_tuple(
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14);
        serialize(archive, data, version);
    }

    template <typename Archive, typename T>
    void serialize_struct(Archive& archive, T& t, const unsigned int version,
        hpx::traits::detail::size<15>)
    {
        auto& [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14,
            p15] = t;
        auto&& data = hpx::forward_as_tuple(
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15);
        serialize(archive, data, version);
    }

    template <typename Archive, typename T>
    void serialize_struct(Archive& ar, T& t, const unsigned int version)
    {
        serialize_struct(ar, t, version, hpx::traits::detail::arity<T>());
    }
}}    // namespace hpx::serialization
