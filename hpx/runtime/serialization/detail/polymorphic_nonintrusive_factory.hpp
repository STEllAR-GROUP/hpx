//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2015 Andreas Schaefer
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_POLYMORPHIC_NONINTRUSIVE_FACTORY_HPP
#define HPX_SERIALIZATION_POLYMORPHIC_NONINTRUSIVE_FACTORY_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/traits/needs_automatic_registration.hpp>
#include <hpx/traits/polymorphic_traits.hpp>
#include <hpx/util/demangle_helper.hpp>
#include <hpx/util/jenkins_hash.hpp>
#include <hpx/util/static.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/type_traits/is_abstract.hpp>

#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace serialization { namespace detail
{
    template <typename T>
    struct get_serialization_name
#ifdef HPX_DISABLE_AUTOMATIC_SERIALIZATION_REGISTRATION
    ;
#else
    {
        const char *operator()()
        {
            /// If you encounter this assert while compiling code, that means that
            /// you have a HPX_REGISTER_ACTION macro somewhere in a source file,
            /// but the header in which the action is defined misses a
            /// HPX_REGISTER_ACTION_DECLARATION
            static_assert(
                traits::needs_automatic_registration<T>::value,
                "HPX_REGISTER_ACTION_DECLARATION missing");
            return util::type_id<T>::typeid_.type_id();
        }
    };
#endif

    struct function_bunch_type
    {
        typedef void (*save_function_type) (output_archive& , const void* base);
        typedef void (*load_function_type) (input_archive& , void* base);
        typedef void* (*create_function_type) (input_archive&);

        save_function_type save_function;
        load_function_type load_function;
        create_function_type create_function;
    };

    template <class T>
    class constructor_selector
    {
    public:
        static T *create(input_archive& ar)
        {
            T *t = new T;
            try {
                load_polymorphic(t, ar, hpx::traits::is_nonintrusive_polymorphic<T>());
            } catch (...) {
                delete t;
                throw;
            }
            return t;
        }

    private:
        static void load_polymorphic(T *t, input_archive& ar, boost::mpl::true_)
        {
            serialize(ar, *t, 0);
        }

        static void load_polymorphic(T *t, input_archive& ar, boost::mpl::false_)
        {
            ar >> *t;
        }
    };

    class polymorphic_nonintrusive_factory
    {
        HPX_NON_COPYABLE(polymorphic_nonintrusive_factory);

    public:
        typedef std::unordered_map<std::string,
                  function_bunch_type, hpx::util::jenkins_hash> serializer_map_type;
        typedef std::unordered_map<std::string,
                  std::string, hpx::util::jenkins_hash> serializer_typeinfo_map_type;

        HPX_EXPORT static polymorphic_nonintrusive_factory& instance();

        void register_class(const std::type_info& typeinfo,
            const std::string& class_name,
            const function_bunch_type& bunch)
        {
            if(!typeinfo.name() && std::string(typeinfo.name()).empty())
            {
                HPX_THROW_EXCEPTION(serialization_error
                  , "polymorphic_nonintrusive_factory::register_class"
                  , "Cannot register a factory with an empty type name");
            }
            if(class_name.empty())
            {
                HPX_THROW_EXCEPTION(serialization_error
                  , "polymorphic_nonintrusive_factory::register_class"
                  , "Cannot register a factory with an empty name");
            }
            auto it = map_.find(class_name);
            auto jt = typeinfo_map_.find(typeinfo.name());

            if(it == map_.end())
                map_.emplace(class_name, bunch);
            if(jt == typeinfo_map_.end())
                typeinfo_map_.emplace(typeinfo.name(), class_name);
        }

        // the following templates are defined in *.ipp file
        template <class T>
        void save(output_archive& ar, const T& t);

        template <class T>
        void load(input_archive& ar, T& t);

        // use raw pointer to construct either
        // shared_ptr or intrusive_ptr from it
        template <class T>
        T* load(input_archive& ar);

    private:
        polymorphic_nonintrusive_factory()
        {
        }

        friend struct hpx::util::static_<polymorphic_nonintrusive_factory>;

        serializer_map_type map_;
        serializer_typeinfo_map_type typeinfo_map_;
    };

    template <class Derived>
    struct register_class
    {
        static void save(output_archive& ar, const void* base)
        {
            serialize(ar, *static_cast<Derived*>(const_cast<void*>(base)), 0);
        }

        static void load(input_archive& ar, void* base)
        {
            serialize(ar, *static_cast<Derived*>(base), 0);
        }

        // this function is needed for pointer type serialization
        static void* create(input_archive& ar)
        {
            return constructor_selector<Derived>::create(ar);
        }

        register_class()
        {
            function_bunch_type bunch = {
                &register_class<Derived>::save,
                &register_class<Derived>::load,
                &register_class<Derived>::create
            };

           // It's safe to call typeid here. The typeid(t) return value is
           // only used for local lookup to the portable string that goes over the
           // wire
            polymorphic_nonintrusive_factory::instance().
                register_class(
                    typeid(Derived),
                    get_serialization_name<Derived>()(),
                    bunch
                );
        }

        static register_class instance;
    };

    template <class T>
    register_class<T> register_class<T>::instance;

}}}

#include <hpx/config/warnings_suffix.hpp>

#define HPX_SERIALIZATION_REGISTER_CLASS_DECLARATION(Class)                   \
    namespace hpx { namespace serialization { namespace detail {              \
        template <>                                                           \
        struct HPX_ALWAYS_EXPORT get_serialization_name<Class>;               \
    }}}                                                                       \
    namespace hpx { namespace traits {                                        \
        template <>                                                           \
        struct needs_automatic_registration<action>                           \
          : boost::mpl::false_                                                \
        {};                                                                   \
    }}                                                                        \
    HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC(Class);                               \
/**/
#define HPX_SERIALIZATION_REGISTER_CLASS_NAME(Class, Name)                    \
    namespace hpx { namespace serialization { namespace detail {              \
        template <>                                                           \
        struct HPX_ALWAYS_EXPORT get_serialization_name<Class>                \
        {                                                                     \
            char const* operator()()                                          \
            {                                                                 \
                return Name;                                                  \
            }                                                                 \
        };                                                                    \
    }}}                                                                       \
    template hpx::serialization::detail::register_class<Class>                \
        hpx::serialization::detail::register_class<Class>::instance;          \
/**/
#define HPX_SERIALIZATION_REGISTER_CLASS_NAME_TEMPLATE(                       \
        Parameters, Template, Name)                                           \
    namespace hpx { namespace serialization { namespace detail {              \
        HPX_UTIL_STRIP(Parameters)                                            \
        struct HPX_ALWAYS_EXPORT get_serialization_name<HPX_UTIL_STRIP(       \
            Template)>                                                        \
        {                                                                     \
            char const* operator()()                                          \
            {                                                                 \
                return Name;                                                  \
            }                                                                 \
        };                                                                    \
    }}}                                                                       \
/**/
#define HPX_SERIALIZATION_REGISTER_CLASS(Class)                               \
    HPX_SERIALIZATION_REGISTER_CLASS_NAME(Class, BOOST_PP_STRINGIZE(Class))   \
/**/
#define HPX_SERIALIZATION_REGISTER_CLASS_TEMPLATE(Parameters, Template)       \
    HPX_SERIALIZATION_REGISTER_CLASS_NAME_TEMPLATE(                           \
        Parameters, Template,                                                 \
        hpx::util::type_id<HPX_UTIL_STRIP(Template) >::typeid_.type_id())     \
    HPX_UTIL_STRIP(Parameters) hpx::serialization::detail::register_class<    \
        HPX_UTIL_STRIP(Template)>                                             \
        HPX_UTIL_STRIP(Template)::hpx_register_class_instance;                \
/**/
#define HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_SEMIINTRUSIVE(Template)        \
    static hpx::serialization::detail::register_class<Template>               \
    hpx_register_class_instance;                                              \
                                                                              \
    virtual hpx::serialization::detail::register_class<Template>&             \
    hpx_get_register_class_instance(                                          \
        hpx::serialization::detail::register_class<Template>*) const          \
    {                                                                         \
        return hpx_register_class_instance;                                   \
    }                                                                         \
/**/
#define HPX_SERIALIZATION_WITH_CUSTOM_CONSTRUCTOR(Class, Func)                \
    namespace hpx { namespace serialization { namespace detail {              \
    template<>                                                                \
    class constructor_selector<HPX_UTIL_STRIP(Class)>                         \
    {                                                                         \
    public:                                                                   \
        static Class *create(input_archive& ar)                               \
        {                                                                     \
            return Func(ar);                                                  \
        }                                                                     \
    };                                                                        \
    }}}                                                                       \
/**/
#define HPX_SERIALIZATION_WITH_CUSTOM_CONSTRUCTOR_TEMPLATE(                   \
    Parameters, Template, Func)                                               \
    namespace hpx { namespace serialization { namespace detail {              \
    HPX_UTIL_STRIP(Parameters)                                                \
    class constructor_selector<HPX_UTIL_STRIP(Template)>                      \
    {                                                                         \
    public:                                                                   \
        static HPX_UTIL_STRIP(Template) *create(input_archive& ar)            \
        {                                                                     \
            return Func(ar, static_cast<HPX_UTIL_STRIP(Template)*>(0));       \
        }                                                                     \
    };                                                                        \
    }}}                                                                       \
/**/
#endif
