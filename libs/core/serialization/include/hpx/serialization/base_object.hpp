//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file base_object.hpp

#pragma once

#include <hpx/serialization/access.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/traits/polymorphic_traits.hpp>

#include <type_traits>

namespace hpx::serialization {

    template <typename Derived, typename Base, typename Enable = void>
    struct base_object_type
    {
        explicit constexpr base_object_type(Derived& d) noexcept
          : d_(d)
        {
        }
        Derived& d_;

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            access::serialize(ar,
                static_cast<Base&>(const_cast<std::decay_t<Derived>&>(d_)), 0);
        }
    };

    // we need another specialization to explicitly specify non-virtual calls of
    // virtual functions in intrusively serialized base classes.
    template <typename Derived, typename Base>
    struct base_object_type<Derived, Base,
        std::enable_if_t<hpx::traits::is_intrusive_polymorphic_v<Derived>>>
    {
        explicit constexpr base_object_type(Derived& d) noexcept
          : d_(d)
        {
        }
        Derived& d_;

        template <typename Archive>
        void save(Archive& ar, unsigned) const
        {
            access::save_base_object(ar, static_cast<Base const&>(d_), 0);
        }

        template <typename Archive>
        void load(Archive& ar, unsigned)
        {
            access::load_base_object(ar, static_cast<Base&>(d_), 0);
        }

        HPX_SERIALIZATION_SPLIT_MEMBER();
    };

    template <typename Base, typename Derived>
    constexpr base_object_type<Derived, Base> base_object(Derived& d) noexcept
    {
        return base_object_type<Derived, Base>(d);
    }

    // allow our base_object_type to be serialized as prvalue compiler should
    // support good ADL implementation but it is rather for all hpx
    // serialization library
    template <typename D, typename B>
    HPX_FORCEINLINE output_archive& operator<<(
        output_archive& ar, base_object_type<D, B> t)
    {
        ar.save(t);
        return ar;
    }

    template <typename D, typename B>
    HPX_FORCEINLINE input_archive& operator>>(
        input_archive& ar, base_object_type<D, B> t)
    {
        ar.load(t);
        return ar;
    }

    template <typename D, typename B>
    HPX_FORCEINLINE output_archive& operator&(    //-V524
        output_archive& ar, base_object_type<D, B> t)
    {
        ar.save(t);
        return ar;
    }

    template <typename D, typename B>
    HPX_FORCEINLINE input_archive& operator&(    //-V524
        input_archive& ar, base_object_type<D, B> t)
    {
        ar.load(t);
        return ar;
    }
}    // namespace hpx::serialization
