//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2022-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/preprocessor/stringize.hpp>
#include <hpx/serialization/detail/polymorphic_intrusive_factory.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/traits/polymorphic_traits.hpp>
#include <hpx/type_support/static.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::serialization::detail {

    class id_registry
    {
    public:
        HPX_NON_COPYABLE(id_registry);

    public:
        using ctor_t = void* (*) ();
        using typename_to_ctor_t = std::map<std::string, ctor_t>;
        using typename_to_id_t = std::map<std::string, std::uint32_t>;
        using cache_t = std::vector<ctor_t>;

        static constexpr std::uint32_t invalid_id = ~0u;

        HPX_CORE_EXPORT void register_factory_function(
            std::string const& type_name, ctor_t ctor);

        HPX_CORE_EXPORT void register_typename(
            std::string const& type_name, std::uint32_t id);

        HPX_CORE_EXPORT void fill_missing_typenames();

        [[nodiscard]] HPX_CORE_EXPORT std::uint32_t try_get_id(
            std::string const& type_name) const;

        [[nodiscard]] std::uint32_t get_max_registered_id() const noexcept
        {
            return max_id;
        }

        [[nodiscard]] HPX_CORE_EXPORT std::vector<std::string>
        get_unassigned_typenames() const;

        HPX_CORE_EXPORT static id_registry& instance();

    private:
        id_registry() noexcept
          : max_id(0u)
        {
        }

        friend struct ::hpx::util::static_<id_registry>;
        friend class polymorphic_id_factory;

        HPX_CORE_EXPORT void cache_id(std::uint32_t id, ctor_t ctor);

        std::uint32_t max_id;
        typename_to_ctor_t typename_to_ctor;
        typename_to_id_t typename_to_id;
        cache_t cache;
    };

    class polymorphic_id_factory
    {
    public:
        HPX_NON_COPYABLE(polymorphic_id_factory);

    private:
        using ctor_t = id_registry::ctor_t;
        using typename_to_ctor_t = id_registry::typename_to_ctor_t;
        using typename_to_id_t = id_registry::typename_to_id_t;
        using cache_t = id_registry::cache_t;

    public:
        template <class T>
        [[nodiscard]] static T* create(
            std::uint32_t id, std::string const* name = nullptr)
        {
            cache_t const& vec = id_registry::instance().cache;

            if (id >= vec.size())    //-V104
            {
                std::string msg(
                    "Unknown type descriptor " + std::to_string(id));
#if defined(HPX_DEBUG)
                if (name != nullptr)
                {
                    msg += ", for typename " + *name + "\n";
                    msg += collect_registered_typenames();
                }
#else
                HPX_UNUSED(name);
#endif
                HPX_THROW_EXCEPTION(hpx::error::serialization_error,
                    "polymorphic_id_factory::create", msg);
            }

            ctor_t const ctor = vec[static_cast<std::size_t>(id)];
            HPX_ASSERT(ctor != nullptr);    //-V108
            return static_cast<T*>(ctor());
        }

        [[nodiscard]] HPX_CORE_EXPORT static std::uint32_t get_id(
            std::string const& type_name);

    private:
        polymorphic_id_factory() = default;

        HPX_CORE_EXPORT static polymorphic_id_factory& instance();

        [[nodiscard]] HPX_CORE_EXPORT static std::string
        collect_registered_typenames();

        friend struct hpx::util::static_<polymorphic_id_factory>;
    };

    template <typename T>
    struct register_class_name<T,
        std::enable_if_t<traits::is_serialized_with_id_v<T>>>
    {
        register_class_name()
        {
            id_registry::instance().register_factory_function(
                T::hpx_serialization_get_name_impl(), &factory_function);
        }

        [[nodiscard]] static void* factory_function()
        {
            return new T;
        }

        register_class_name& instantiate()
        {
            return *this;
        }

        static register_class_name instance;
    };

    template <class T>
    register_class_name<T, std::enable_if_t<traits::is_serialized_with_id_v<T>>>
        register_class_name<T,
            std::enable_if_t<traits::is_serialized_with_id_v<T>>>::instance;

    template <std::uint32_t desc>
    [[nodiscard]] std::string get_constant_entry_name();

    template <std::uint32_t Id>
    struct add_constant_entry
    {
        add_constant_entry()
        {
            id_registry::instance().register_typename(
                get_constant_entry_name<Id>(), Id);
        }

        static add_constant_entry instance;
    };

    template <std::uint32_t Id>
    add_constant_entry<Id> add_constant_entry<Id>::instance;

}    // namespace hpx::serialization::detail

#include <hpx/config/warnings_suffix.hpp>

#define HPX_SERIALIZATION_ADD_CONSTANT_ENTRY(String, Id)                       \
    namespace hpx::serialization::detail {                                     \
        template <>                                                            \
        std::string get_constant_entry_name</**/ Id>()                         \
        {                                                                      \
            return HPX_PP_STRINGIZE(String);                                   \
        }                                                                      \
        template add_constant_entry<Id> add_constant_entry</**/ Id>::instance; \
    }                                                                          \
    /**/
