//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/access.hpp>
#include <hpx/serialization/basic_archive.hpp>
#include <hpx/serialization/detail/extra_archive_data.hpp>
#include <hpx/serialization/detail/non_default_constructible.hpp>
#include <hpx/serialization/detail/polymorphic_id_factory.hpp>
#include <hpx/serialization/detail/polymorphic_intrusive_factory.hpp>
#include <hpx/serialization/detail/polymorphic_nonintrusive_factory.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/string.hpp>
#include <hpx/serialization/traits/polymorphic_traits.hpp>
#include <hpx/type_support/identity.hpp>
#include <hpx/type_support/lazy_conditional.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace serialization {

    ////////////////////////////////////////////////////////////////////////////
    namespace detail {

        struct ptr_helper;

        // we need top use shared_ptr as util::any requires for the held type
        // to be copy-constructible
        using ptr_helper_ptr = std::unique_ptr<ptr_helper>;

        using input_pointer_tracker = std::map<std::uint64_t, ptr_helper_ptr>;
        using output_pointer_tracker = std::map<void const*, std::uint64_t>;

        // This is explicitly instantiated to ensure that the id is stable across
        // shared libraries. MSVC and gcc/clang require different handling of
        // exported explicitly instantiated templates.
#if defined(HPX_MSVC)
        extern template struct extra_archive_data_id_helper<
            input_pointer_tracker>;
        extern template struct extra_archive_data_id_helper<
            output_pointer_tracker>;
#else
        extern template struct HPX_CORE_EXPORT
            extra_archive_data_id_helper<input_pointer_tracker>;
        extern template struct HPX_CORE_EXPORT
            extra_archive_data_id_helper<output_pointer_tracker>;
#endif
    }    // namespace detail

    ////////////////////////////////////////////////////////////////////////////
    HPX_CORE_EXPORT void register_pointer(
        input_archive& ar, std::uint64_t pos, detail::ptr_helper_ptr helper);

    HPX_CORE_EXPORT detail::ptr_helper& tracked_pointer(
        input_archive& ar, std::uint64_t pos);

    HPX_CORE_EXPORT std::uint64_t track_pointer(
        output_archive& ar, void const* pos);

    ////////////////////////////////////////////////////////////////////////////
    namespace detail {

        template <typename Pointer>
        struct erase_ptr_helper : ptr_helper
        {
            using referred_type = typename Pointer::element_type;

            erase_ptr_helper(Pointer&& t, Pointer& ptr)
              : t_(std::move(t))
            {
                ptr = t_;
            }

            Pointer t_;
        };

        template <typename Pointer>
        class pointer_input_dispatcher
        {
            using referred_type = typename Pointer::element_type;

            struct intrusive_polymorphic
            {
                static Pointer call(input_archive& ar)
                {
                    std::string name;
                    ar >> name;

                    Pointer t(polymorphic_intrusive_factory::instance()
                                  .create<referred_type>(name));
                    ar >> *t;
                    return t;
                }
            };

            struct polymorphic_with_id
            {
                static Pointer call(input_archive& ar)
                {
#if !defined(HPX_DEBUG)
                    std::uint32_t id;
                    ar >> id;

                    Pointer t(
                        polymorphic_id_factory::create<referred_type>(id));
                    ar >> *t;
                    return t;
#else
                    std::uint32_t id;
                    std::string name;
                    ar >> name;
                    ar >> id;

                    Pointer t(polymorphic_id_factory::create<referred_type>(
                        id, &name));
                    ar >> *t;
                    return t;
#endif
                }
            };

            struct nonintrusive_polymorphic
            {
                static Pointer call(input_archive& ar)
                {
                    return Pointer(polymorphic_nonintrusive_factory::instance()
                                       .load<referred_type>(ar));
                }
            };

            struct usual
            {
                static Pointer call(input_archive& ar)
                {
                    Pointer t(constructor_selector<referred_type>::create(ar));
                    return t;
                }
            };

        public:
            using type = typename util::lazy_conditional<
                hpx::traits::is_serialized_with_id<referred_type>::value,
                hpx::util::identity<polymorphic_with_id>,
                std::conditional<
                    hpx::traits::is_intrusive_polymorphic<referred_type>::value,
                    intrusive_polymorphic,
                    typename std::conditional<
                        hpx::traits::is_nonintrusive_polymorphic<
                            referred_type>::value,
                        nonintrusive_polymorphic, usual>::type>>::type;
        };

        template <typename Pointer>
        class pointer_output_dispatcher
        {
            using referred_type = typename Pointer::element_type;

            struct intrusive_polymorphic
            {
                static void call(output_archive& ar, Pointer const& ptr)
                {
                    const std::string name = access::get_name(ptr.get());
                    ar << name;
                    ar << *ptr;
                }
            };

            struct polymorphic_with_id
            {
                static void call(output_archive& ar, Pointer const& ptr)
                {
#if !defined(HPX_DEBUG)
                    const std::uint32_t id = polymorphic_id_factory::get_id(
                        access::get_name(ptr.get()));
                    ar << id;
                    ar << *ptr;
#else
                    std::string const name(access::get_name(ptr.get()));
                    const std::uint32_t id =
                        polymorphic_id_factory::get_id(name);
                    ar << name;
                    ar << id;
                    ar << *ptr;
#endif
                }
            };

            struct usual
            {
                static void call(output_archive& ar, Pointer const& ptr)
                {
                    call(ar, *ptr);
                }

                template <typename T>
                static typename std::enable_if<
                    !std::is_constructible<T>::value>::type
                call(output_archive& ar, T const& val)
                {
                    save_construct_data(ar, &val, 0);
                    ar << val;
                }

                template <typename T>
                static typename std::enable_if<
                    std::is_constructible<T>::value>::type
                call(output_archive& ar, T const& val)
                {
                    ar << val;
                }
            };

        public:
            using type = typename std::conditional<
                hpx::traits::is_serialized_with_id<referred_type>::value,
                polymorphic_with_id,
                typename std::conditional<
                    hpx::traits::is_intrusive_polymorphic<referred_type>::value,
                    intrusive_polymorphic, usual>::type>::type;
        };

        // forwarded serialize pointer functions
        template <typename Pointer>
        HPX_FORCEINLINE void serialize_pointer_tracked(
            output_archive& ar, Pointer const& ptr)
        {
            bool valid = static_cast<bool>(ptr);
            ar << valid;
            if (valid)
            {
                std::uint64_t cur_pos = current_pos(ar);
                std::uint64_t pos = track_pointer(ar, ptr.get());
                ar << pos;
                if (pos == std::uint64_t(-1))
                {
                    ar << cur_pos;
                    detail::pointer_output_dispatcher<Pointer>::type::call(
                        ar, ptr);
                }
            }
        }

        template <typename Pointer>
        HPX_FORCEINLINE void serialize_pointer_tracked(
            input_archive& ar, Pointer& ptr)
        {
            bool valid = false;
            ar >> valid;
            if (valid)
            {
                std::uint64_t pos = 0;
                ar >> pos;
                if (pos == std::uint64_t(-1))
                {
                    pos = 0;
                    ar >> pos;
                    Pointer temp =
                        detail::pointer_input_dispatcher<Pointer>::type::call(
                            ar);
                    register_pointer(ar, pos,
                        ptr_helper_ptr(new detail::erase_ptr_helper<Pointer>(
                            std::move(temp), ptr)));
                }
                else
                {
                    auto& helper =
                        static_cast<detail::erase_ptr_helper<Pointer>&>(
                            tracked_pointer(ar, pos));
                    ptr = helper.t_;
                }
            }
        }

        template <typename Pointer>
        HPX_FORCEINLINE void serialize_pointer_untracked(
            output_archive& ar, Pointer const& ptr)
        {
            bool valid = static_cast<bool>(ptr);
            ar << valid;
            if (valid)
            {
                detail::pointer_output_dispatcher<Pointer>::type::call(ar, ptr);
            }
        }

        template <typename Pointer>
        HPX_FORCEINLINE void serialize_pointer_untracked(
            input_archive& ar, Pointer& ptr)
        {
            bool valid = false;
            ar >> valid;
            if (valid)
            {
                ptr = detail::pointer_input_dispatcher<Pointer>::type::call(ar);
            }
        }

    }    // namespace detail
}}       // namespace hpx::serialization
