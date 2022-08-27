//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2022 Hartmut Kaiser
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

namespace hpx::serialization {

    ////////////////////////////////////////////////////////////////////////////
    namespace detail {

        struct ptr_helper;

        // we need top use shared_ptr as util::any requires for the held type
        // to be copy-constructible
        using ptr_helper_ptr = std::unique_ptr<ptr_helper>;

        using input_pointer_tracker = std::map<std::uint64_t, ptr_helper_ptr>;
        using output_pointer_tracker = std::map<void const*, std::uint64_t>;

        // This is explicitly instantiated to ensure that the id is stable across
        // shared libraries.
        template <>
        struct extra_archive_data_helper<input_pointer_tracker>
        {
            HPX_CORE_EXPORT static extra_archive_data_id_type id() noexcept;
            static constexpr void reset(input_pointer_tracker*) noexcept {}
        };

        template <>
        struct extra_archive_data_helper<output_pointer_tracker>
        {
            HPX_CORE_EXPORT static extra_archive_data_id_type id() noexcept;
            HPX_CORE_EXPORT static void reset(output_pointer_tracker* data);
        };
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

            erase_ptr_helper(Pointer&& t, Pointer& ptr) noexcept
              : t_(HPX_MOVE(t))
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
                    Pointer t(
                        constructor_selector_ptr<referred_type>::create(ar));
                    return t;
                }
            };

        public:
            using type = util::lazy_conditional_t<
                hpx::traits::is_serialized_with_id_v<referred_type>,
                hpx::util::identity<polymorphic_with_id>,
                std::conditional<
                    hpx::traits::is_intrusive_polymorphic_v<referred_type>,
                    intrusive_polymorphic,
                    std::conditional_t<
                        hpx::traits::is_nonintrusive_polymorphic_v<
                            referred_type>,
                        nonintrusive_polymorphic, usual>>>;
        };

        template <typename Pointer>
        class pointer_output_dispatcher
        {
            using referred_type = typename Pointer::element_type;

            struct intrusive_polymorphic
            {
                static void call(output_archive& ar, Pointer const& ptr)
                {
                    std::string const name = access::get_name(ptr.get());
                    ar << name;
                    ar << *ptr;
                }
            };

            struct polymorphic_with_id
            {
                static void call(output_archive& ar, Pointer const& ptr)
                {
#if !defined(HPX_DEBUG)
                    std::uint32_t const id = polymorphic_id_factory::get_id(
                        access::get_name(ptr.get()));
                    ar << id;
                    ar << *ptr;
#else
                    std::string const name(access::get_name(ptr.get()));
                    std::uint32_t const id =
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
                    using element_type = typename Pointer::element_type;
                    if constexpr (std::is_constructible_v<element_type>)
                    {
                        ar << *ptr;
                    }
                    else
                    {
                        save_construct_data(ar, ptr.get(), 0);
                        ar << *ptr;
                    }
                }
            };

        public:
            using type = std::conditional_t<
                hpx::traits::is_serialized_with_id_v<referred_type>,
                polymorphic_with_id,
                std::conditional_t<
                    hpx::traits::is_intrusive_polymorphic_v<referred_type>,
                    intrusive_polymorphic, usual>>;
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
                            HPX_MOVE(temp), ptr)));
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
}    // namespace hpx::serialization
