//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2015 Andreas Schaefer
//  Copyright (c) 2022-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/debugging.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/preprocessor/stringize.hpp>
#include <hpx/preprocessor/strip_parens.hpp>
#include <hpx/serialization/detail/non_default_constructible.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/traits/needs_automatic_registration.hpp>
#include <hpx/serialization/traits/polymorphic_traits.hpp>
#include <hpx/type_support/static.hpp>

#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::serialization::detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct get_serialization_name
#ifdef HPX_DISABLE_AUTOMATIC_SERIALIZATION_REGISTRATION
        ;
#else
    {
        [[nodiscard]] char const* operator()() const noexcept
        {
            // If you encounter this assert while compiling code, that means
            // that you have a HPX_REGISTER_ACTION macro somewhere in a source
            // file, but the header in which the action is defined misses a
            // HPX_REGISTER_ACTION_DECLARATION
            static_assert(traits::needs_automatic_registration_v<T>,
                "HPX_REGISTER_ACTION_DECLARATION missing");
            return util::debug::type_id<T>::typeid_.type_id();
        }
    };
#endif

    struct function_bunch_type
    {
        using save_function_type = void (*)(output_archive&, void const* base);
        using load_function_type = void (*)(input_archive&, void* base);
        using create_function_type = void* (*) (input_archive&);

        save_function_type save_function;
        load_function_type load_function;
        create_function_type create_function;
    };

    template <typename T>
    class constructor_selector_ptr
    {
    public:
        [[nodiscard]] static T* create(input_archive& ar)
        {
            std::unique_ptr<T> t;

            // create new object
            if constexpr (std::is_default_constructible_v<T>)
            {
                t.reset(new T);
            }
            else
            {
                using storage_type =
                    std::aligned_storage_t<sizeof(T), alignof(T)>;

                t.reset(reinterpret_cast<T*>(new storage_type));    //-V572
                load_construct_data(ar, t.get(), 0);
            }

            // de-serialize new object
            if constexpr (hpx::traits::is_nonintrusive_polymorphic_v<T>)
            {
                serialize(ar, *t, 0);
            }
            else
            {
                ar >> *t;
            }

            return t.release();
        }
    };

    class polymorphic_nonintrusive_factory
    {
    public:
        HPX_NON_COPYABLE(polymorphic_nonintrusive_factory);

    public:
        using serializer_map_type = std::unordered_map<std::string,
            function_bunch_type, std::hash<std::string>>;
        using serializer_typeinfo_map_type = std::unordered_map<std::string,
            std::string, std::hash<std::string>>;

        HPX_CORE_EXPORT static polymorphic_nonintrusive_factory& instance();

        void register_class(std::type_info const& typeinfo,
            std::string const& class_name, function_bunch_type const& bunch)
        {
            if (!typeinfo.name() && std::string(typeinfo.name()).empty())
            {
                HPX_THROW_EXCEPTION(hpx::error::serialization_error,
                    "polymorphic_nonintrusive_factory::register_class",
                    "Cannot register a factory with an empty type name");
            }
            if (class_name.empty())
            {
                HPX_THROW_EXCEPTION(hpx::error::serialization_error,
                    "polymorphic_nonintrusive_factory::register_class",
                    "Cannot register a factory with an empty name");
            }
            auto const it = map_.find(class_name);
            auto const jt = typeinfo_map_.find(typeinfo.name());

            if (it == map_.end())
                map_[class_name] = bunch;
            if (jt == typeinfo_map_.end())
                typeinfo_map_[typeinfo.name()] = class_name;
        }

        // the following templates are defined in *.ipp file
        template <typename T>
        void save(output_archive& ar, T const& t);

        template <typename T>
        void load(input_archive& ar, T& t);

        // use raw pointer to construct either shared_ptr or intrusive_ptr from
        // it
        template <typename T>
        [[nodiscard]] T* load(input_archive& ar);

    private:
        polymorphic_nonintrusive_factory() = default;

        friend struct hpx::util::static_<polymorphic_nonintrusive_factory>;

        serializer_map_type map_;
        serializer_typeinfo_map_type typeinfo_map_;
    };

    template <typename Derived>
    struct register_class
    {
        static void save(output_archive& ar, void const* base)
        {
            serialize(ar, *static_cast<Derived*>(const_cast<void*>(base)), 0);
        }

        static void load(input_archive& ar, void* base)
        {
            serialize(ar, *static_cast<Derived*>(base), 0);
        }

        // this function is needed for pointer type serialization
        [[nodiscard]] static void* create(input_archive& ar)
        {
            return constructor_selector_ptr<Derived>::create(ar);
        }

        register_class()
        {
            static constexpr function_bunch_type bunch = {
                &register_class<Derived>::save, &register_class<Derived>::load,
                &register_class<Derived>::create};

            // It's safe to call typeid here. The typeid(t) return value is only
            // used for local lookup to the portable string that goes over the
            // wire
            polymorphic_nonintrusive_factory::instance().register_class(
                typeid(Derived), get_serialization_name<Derived>()(), bunch);
        }

        static register_class instance;
    };

    template <class T>
    register_class<T> register_class<T>::instance;

}    // namespace hpx::serialization::detail

#include <hpx/config/warnings_suffix.hpp>

#define HPX_SERIALIZATION_REGISTER_CLASS_DECLARATION(Class)                    \
    namespace hpx::serialization::detail {                                     \
        template <>                                                            \
        struct HPX_ALWAYS_EXPORT get_serialization_name</**/ Class>;           \
    }                                                                          \
    namespace hpx::traits {                                                    \
        template <>                                                            \
        struct needs_automatic_registration</**/ action> : std::false_type     \
        {                                                                      \
        };                                                                     \
    }                                                                          \
    HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC(Class)                                 \
/**/
#define HPX_SERIALIZATION_REGISTER_CLASS_NAME(Class, Name)                     \
    namespace hpx::serialization::detail {                                     \
        template <>                                                            \
        struct HPX_ALWAYS_EXPORT get_serialization_name</**/ Class>            \
        {                                                                      \
            constexpr char const* operator()() const noexcept                  \
            {                                                                  \
                return Name;                                                   \
            }                                                                  \
        };                                                                     \
    }                                                                          \
    template hpx::serialization::detail::register_class</**/ Class>            \
        hpx::serialization::detail::register_class</**/ Class>::instance;      \
/**/
#define HPX_SERIALIZATION_REGISTER_CLASS_NAME_TEMPLATE(                        \
    Parameters, Template, Name)                                                \
    namespace hpx::serialization::detail {                                     \
        HPX_PP_STRIP_PARENS(Parameters)                                        \
        struct HPX_ALWAYS_EXPORT                                               \
            get_serialization_name<HPX_PP_STRIP_PARENS(Template)>              \
        {                                                                      \
            constexpr char const* operator()() const noexcept                  \
            {                                                                  \
                return Name;                                                   \
            }                                                                  \
        };                                                                     \
    }                                                                          \
/**/
#define HPX_SERIALIZATION_REGISTER_CLASS(Class)                                \
    HPX_SERIALIZATION_REGISTER_CLASS_NAME(Class, HPX_PP_STRINGIZE(Class))      \
/**/
#define HPX_SERIALIZATION_REGISTER_CLASS_TEMPLATE(Parameters, Template)        \
    HPX_SERIALIZATION_REGISTER_CLASS_NAME_TEMPLATE(Parameters, Template,       \
        hpx::util::debug::type_id<HPX_PP_STRIP_PARENS(Template)>::typeid_      \
            .type_id())                                                        \
    HPX_PP_STRIP_PARENS(Parameters)                                            \
    hpx::serialization::detail::register_class<HPX_PP_STRIP_PARENS(Template)>  \
        HPX_PP_STRIP_PARENS(Template)::hpx_register_class_instance;            \
/**/
#define HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_SEMIINTRUSIVE(Template)         \
    static hpx::serialization::detail::register_class<Template>                \
        hpx_register_class_instance;                                           \
                                                                               \
    virtual hpx::serialization::detail::register_class</**/ Template>&         \
    hpx_get_register_class_instance(                                           \
        hpx::serialization::detail::register_class</**/ Template>*) const      \
    {                                                                          \
        return hpx_register_class_instance;                                    \
    }                                                                          \
/**/
#define HPX_SERIALIZATION_WITH_CUSTOM_CONSTRUCTOR(Class, Func)                 \
    namespace hpx::serialization::detail {                                     \
        template <>                                                            \
        class constructor_selector_ptr<HPX_PP_STRIP_PARENS(Class)>             \
        {                                                                      \
        public:                                                                \
            static Class* create(input_archive& ar)                            \
            {                                                                  \
                return Func(ar);                                               \
            }                                                                  \
        };                                                                     \
    }                                                                          \
/**/
#define HPX_SERIALIZATION_WITH_CUSTOM_CONSTRUCTOR_TEMPLATE(                    \
    Parameters, Template, Func)                                                \
    namespace hpx::serialization::detail {                                     \
        HPX_PP_STRIP_PARENS(Parameters)                                        \
        class constructor_selector_ptr<HPX_PP_STRIP_PARENS(Template)>          \
        {                                                                      \
        public:                                                                \
            static HPX_PP_STRIP_PARENS(Template) * create(input_archive& ar)   \
            {                                                                  \
                return Func(                                                   \
                    ar, static_cast<HPX_PP_STRIP_PARENS(Template)*>(0));       \
            }                                                                  \
        };                                                                     \
    }                                                                          \
/**/
