//  Copyright (c) 2014 Anton Bikineev
//  Copyright (c) 2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/debugging.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/expand.hpp>
#include <hpx/preprocessor/nargs.hpp>
#include <hpx/preprocessor/stringize.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <functional>
#include <string>
#include <unordered_map>

namespace hpx::serialization::detail {

    class polymorphic_intrusive_factory
    {
    public:
        HPX_NON_COPYABLE(polymorphic_intrusive_factory);

    private:
        using ctor_type = void* (*) ();
        using ctor_map_type =
            std::unordered_map<std::string, ctor_type, std::hash<std::string>>;

    public:
        polymorphic_intrusive_factory() = default;

        HPX_CORE_EXPORT static polymorphic_intrusive_factory& instance();

        HPX_CORE_EXPORT void register_class(
            std::string const& name, ctor_type fun);

        [[nodiscard]] HPX_CORE_EXPORT void* create(
            std::string const& name) const;

        template <typename T>
        [[nodiscard]] T* create(std::string const& name) const
        {
            return static_cast<T*>(create(name));
        }

    private:
        ctor_map_type map_;
    };

    template <typename T, typename Enable = void>
    struct register_class_name
    {
        register_class_name()
        {
            polymorphic_intrusive_factory::instance().register_class(
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

    template <typename T, typename Enable>
    register_class_name<T, Enable> register_class_name<T, Enable>::instance;
}    // namespace hpx::serialization::detail

///////////////////////////////////////////////////////////////////////////////
#define HPX_SERIALIZATION_ADD_GET_NAME_MEMBERS_WITH_NAME(                      \
    Class, Name, Override)                                                     \
    template <typename, typename>                                              \
    friend struct ::hpx::serialization::detail::register_class_name;           \
                                                                               \
    static std::string hpx_serialization_get_name_impl()                       \
    {                                                                          \
        hpx::serialization::detail::register_class_name<Class>::instance       \
            .instantiate();                                                    \
        return Name;                                                           \
    }                                                                          \
    virtual std::string hpx_serialization_get_name() const Override            \
    {                                                                          \
        return Class::hpx_serialization_get_name_impl();                       \
    }                                                                          \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS(Class, Override)               \
    virtual void load(hpx::serialization::input_archive& ar, unsigned n)       \
        Override                                                               \
    {                                                                          \
        serialize<hpx::serialization::input_archive>(ar, n);                   \
    }                                                                          \
    virtual void save(hpx::serialization::output_archive& ar, unsigned n)      \
        const Override                                                         \
    {                                                                          \
        const_cast<Class*>(this)                                               \
            ->serialize<hpx::serialization::output_archive>(ar, n);            \
    }

#define HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS_SPLITTED(Override)             \
    virtual void load(hpx::serialization::input_archive& ar, unsigned n)       \
        Override                                                               \
    {                                                                          \
        load<hpx::serialization::input_archive>(ar, n);                        \
    }                                                                          \
    virtual void save(hpx::serialization::output_archive& ar, unsigned n)      \
        const Override                                                         \
    {                                                                          \
        save<hpx::serialization::output_archive>(ar, n);                       \
    }                                                                          \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_SPLITTED(...)                  \
    HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_SPLITTED_(__VA_ARGS__)             \
    /**/

#define HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_SPLITTED_(...)                 \
    HPX_PP_EXPAND(                                                             \
        HPX_PP_CAT(HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_SPLITTED_,          \
            HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))                           \
    /**/

#define HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_SPLITTED_3(                    \
    Class, Name, Override)                                                     \
    HPX_SERIALIZATION_ADD_GET_NAME_MEMBERS_WITH_NAME(Class, Name, Override);   \
    HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS_SPLITTED(Override)                 \
    /**/

#define HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_SPLITTED_2(Class, Name)        \
    HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_SPLITTED_3(Class, Name, /**/)      \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME(...)                           \
    HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_(__VA_ARGS__)                      \
    /**/

#define HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_(...)                          \
    HPX_PP_EXPAND(HPX_PP_CAT(HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_,         \
        HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))                               \
    /**/

#define HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_3(Class, Name, Override)       \
    HPX_SERIALIZATION_ADD_GET_NAME_MEMBERS_WITH_NAME(Class, Name, Override);   \
    HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS(Class, Override)                   \
    HPX_SERIALIZATION_SPLIT_MEMBER()                                           \
    /**/

#define HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_2(Class, Name)                 \
    HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_3(Class, Name, /**/)               \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_SERIALIZATION_POLYMORPHIC_ABSTRACT(Class)                          \
    virtual std::string hpx_serialization_get_name() const = 0;                \
    HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS(Class, /**/)                       \
    HPX_SERIALIZATION_SPLIT_MEMBER()                                           \
    /**/

#define HPX_SERIALIZATION_POLYMORPHIC_ABSTRACT_SPLITTED(Class)                 \
    virtual std::string hpx_serialization_get_name() const = 0;                \
    HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS_SPLITTED(/**/)                     \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_SERIALIZATION_POLYMORPHIC(...)                                     \
    HPX_SERIALIZATION_POLYMORPHIC_(__VA_ARGS__)                                \
    /**/

#define HPX_SERIALIZATION_POLYMORPHIC_(...)                                    \
    HPX_PP_EXPAND(HPX_PP_CAT(HPX_SERIALIZATION_POLYMORPHIC_,                   \
        HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))                               \
    /**/

#define HPX_SERIALIZATION_POLYMORPHIC_2(Class, Override)                       \
    HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_3(                                 \
        Class, HPX_PP_STRINGIZE(Class), Override)                              \
    /**/

#define HPX_SERIALIZATION_POLYMORPHIC_1(Class)                                 \
    HPX_SERIALIZATION_POLYMORPHIC_2(Class, /**/)                               \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_SERIALIZATION_POLYMORPHIC_SPLITTED(...)                            \
    HPX_SERIALIZATION_POLYMORPHIC_SPLITTED_(__VA_ARGS__)                       \
    /**/

#define HPX_SERIALIZATION_POLYMORPHIC_SPLITTED_(...)                           \
    HPX_PP_EXPAND(HPX_PP_CAT(HPX_SERIALIZATION_POLYMORPHIC_SPLITTED_,          \
        HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))                               \
    /**/

#define HPX_SERIALIZATION_POLYMORPHIC_SPLITTED_2(Class, Override)              \
    HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_SPLITTED_3(                        \
        Class, HPX_PP_STRINGIZE(Class), Override)                              \
    /**/

#define HPX_SERIALIZATION_POLYMORPHIC_SPLITTED_1(Class)                        \
    HPX_SERIALIZATION_POLYMORPHIC_SPLITTED_2(Class, /**/)                      \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE(...)                            \
    HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_(__VA_ARGS__)                       \
    /**/

#define HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_(...)                           \
    HPX_PP_EXPAND(HPX_PP_CAT(HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_,          \
        HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))                               \
    /**/

#define HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_2(Class, Override)              \
    HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_3(                                 \
        Class, hpx::util::debug::type_id<Class>::typeid_.type_id(), Override)  \
    /**/

#define HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_1(Class)                        \
    HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_2(Class, /**/)                      \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_SPLITTED(...)                   \
    HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_SPLITTED_(__VA_ARGS__)              \
    /**/

#define HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_SPLITTED_(...)                  \
    HPX_PP_EXPAND(HPX_PP_CAT(HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_SPLITTED_, \
        HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))                               \
    /**/

#define HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_SPLITTED_2(Class, Override)     \
    HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_SPLITTED_3(                        \
        Class, hpx::util::debug::type_id<T>::typeid_.type_id(), Override)      \
    /**/

#define HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_SPLITTED_1(Class)               \
    HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_SPLITTED_2(Class, /**/)             \
    /**/
