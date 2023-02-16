//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2014-2015 Anton Bikineev
//  Copyright (c) 2022-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/brace_initializable_fwd.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/traits/brace_initializable_traits.hpp>
#include <hpx/serialization/traits/polymorphic_traits.hpp>

#include <string>
#include <type_traits>
#include <utility>

namespace hpx::serialization {

    namespace detail {

        template <typename T>
        HPX_FORCEINLINE void serialize_force_adl(
            output_archive& ar, T const& t, unsigned)
        {
            serialize(ar, const_cast<T&>(t), 0);
        }

        template <class T>
        HPX_FORCEINLINE void serialize_force_adl(
            input_archive& ar, T& t, unsigned)
        {
            serialize(ar, t, 0);
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // This trait must live outside of 'class access' below as otherwise MSVC
    // will find the serialize() function in 'class access' as a dependent class
    // (which is an MS extension)
    template <typename T>
    class has_serialize_adl
    {
        template <typename T1>
        static std::false_type test(...);

        template <typename T1,
            typename = decltype(
                serialize(std::declval<hpx::serialization::output_archive&>(),
                    std::declval<std::remove_const_t<T1>&>(), 0u))>
        static std::true_type test(int);

    public:
        static constexpr bool value = decltype(test<T>(0))::value;
    };

    template <typename T>
    inline constexpr bool has_serialize_adl_v = has_serialize_adl<T>::value;

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class has_struct_serialization
    {
        template <typename T1>
        static std::false_type test(...);

        template <typename T1,
            typename = decltype(serialize_struct(
                std::declval<hpx::serialization::output_archive&>(),
                std::declval<std::remove_const_t<T1>&>(), 0u,
                hpx::traits::detail::arity<T1>()))>
        static std::true_type test(int);

    public:
        static constexpr bool value = decltype(test<T>(0))::value;
    };

    template <typename T>
    inline constexpr bool has_struct_serialization_v =
        has_struct_serialization<T>::value;

    ///////////////////////////////////////////////////////////////////////////
    class access
    {
    public:
        template <typename T>
        class has_serialize
        {
            template <typename T1>
            static std::false_type test(...);

            // the following expression sfinae trick appears to work on
            // clang-3.4, gcc-4.9, icc-16, msvc-2017 (at least) note that this
            // detection would have been much easier to implement if there
            // hadn't been an issue with gcc:
            // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=82478
            template <typename T1,
                typename = decltype(
                    std::declval<std::remove_const_t<T1>&>().serialize(
                        std::declval<hpx::serialization::output_archive&>(),
                        0u))>
            static std::true_type test(int);

        public:
            static constexpr bool value = decltype(test<T>(0))::value;
        };

        template <typename T>
        static constexpr bool has_serialize_v = has_serialize<T>::value;

    public:
        template <typename Archive, typename T>
        static void serialize(Archive& ar, T& t, unsigned)
        {
            if constexpr (hpx::traits::is_intrusive_polymorphic_v<T>)
            {
                // intrusive_polymorphic: the following template function is
                // viable to call the right overloaded function according to T
                // constness and to prevent calling templated version of
                // serialize function
                t.serialize(ar, 0);
            }
            else if constexpr (has_serialize_v<T>)
            {
                // intrusive_usual: cast it to let it be run for templated
                // member functions
                const_cast<std::decay_t<T>&>(t).serialize(ar, 0);
            }
            else if constexpr (!std::is_empty_v<T>)
            {
                // non_intrusive
                if constexpr (has_serialize_adl_v<T>)
                {
                    // this additional indirection level is needed to force ADL
                    // on the second phase of template lookup. call of serialize
                    // function directly from base_object finds only
                    // serialize-member function and doesn't perform ADL
                    detail::serialize_force_adl(ar, t, 0);
                }
                else if constexpr (has_struct_serialization_v<T>)
                {
                    // This is automatic serialization for types that are simple
                    // (brace-initializable) structs, what that means every
                    // struct's field has to be serializable and public.
                    serialize_struct(ar, t, 0);
                }
                else
                {
                    static_assert(
                        has_serialize_adl_v<T> || has_struct_serialization_v<T>,
                        "No serialization method found");
                }
            }
        }

        template <typename Archive, typename T>
        HPX_FORCEINLINE static void save_base_object(
            Archive& ar, T const& t, unsigned)
        {
            // explicitly specify virtual function of base class to avoid
            // infinite recursion
            t.T::save(ar, 0);
        }

        template <typename Archive, typename T>
        HPX_FORCEINLINE static void load_base_object(
            Archive& ar, T& t, unsigned)
        {
            // explicitly specify virtual function of base class to avoid
            // infinite recursion
            t.T::load(ar, 0);
        }

        template <typename T>
        [[nodiscard]] HPX_FORCEINLINE static std::string get_name(T const* t)
        {
            return t->hpx_serialization_get_name();
        }
    };
}    // namespace hpx::serialization
