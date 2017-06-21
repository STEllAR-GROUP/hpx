//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_POLYMORPHIC_ID_FACTORY_HPP
#define HPX_SERIALIZATION_POLYMORPHIC_ID_FACTORY_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/detail/polymorphic_intrusive_factory.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/traits/polymorphic_traits.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/static.hpp>

#include <boost/preprocessor/stringize.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace serialization {

    namespace detail
    {
        class id_registry
        {
        public:
            HPX_NON_COPYABLE(id_registry);

        public:
            typedef void* (*ctor_t) ();
            typedef std::map<std::string, ctor_t> typename_to_ctor_t;
            typedef std::map<std::string, std::uint32_t> typename_to_id_t;
            typedef std::vector<ctor_t> cache_t;

            HPX_STATIC_CONSTEXPR std::uint32_t invalid_id = ~0u;

            HPX_EXPORT void register_factory_function(
                const std::string& type_name, ctor_t ctor);

            HPX_EXPORT void register_typename(
                const std::string& type_name, std::uint32_t id);

            HPX_EXPORT void fill_missing_typenames();

            HPX_EXPORT std::uint32_t try_get_id(
                const std::string& type_name) const;

            std::uint32_t get_max_registered_id() const
            {
                return max_id;
            }

            HPX_EXPORT std::vector<std::string> get_unassigned_typenames() const;

            HPX_EXPORT static id_registry& instance();

        private:
            id_registry() : max_id(0u) {}

            friend struct ::hpx::util::static_<id_registry>;
            friend class polymorphic_id_factory;

            HPX_EXPORT void cache_id(std::uint32_t id, ctor_t ctor);

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
            typedef id_registry::ctor_t ctor_t;
            typedef id_registry::typename_to_ctor_t typename_to_ctor_t;
            typedef id_registry::typename_to_id_t typename_to_id_t;
            typedef id_registry::cache_t cache_t;

        public:
            template <class T>
            static T* create(std::uint32_t id, std::string const* name = nullptr)
            {
                const cache_t& vec = id_registry::instance().cache;

                if (id >= vec.size()) //-V104
                {
                    std::string msg(
                        "Unknown type descriptor " + std::to_string(id));
#if defined(HPX_DEBUG)
                    if (name != nullptr)
                    {
                        msg += ", for typename " + *name + "\n";
                        msg += collect_registered_typenames();
                    }
#endif
                    HPX_THROW_EXCEPTION(serialization_error
                      , "polymorphic_id_factory::create", msg);
                }

                ctor_t ctor = vec[id]; //-V108
                HPX_ASSERT(ctor != nullptr);
                return static_cast<T*>(ctor());
            }

            HPX_EXPORT static std::uint32_t get_id(
                const std::string& type_name);

        private:
            polymorphic_id_factory() {}

            HPX_EXPORT static polymorphic_id_factory& instance();
            HPX_EXPORT static std::string collect_registered_typenames();

            friend struct hpx::util::static_<polymorphic_id_factory>;
        };

        template <class T>
        struct register_class_name<T, typename std::enable_if<
            traits::is_serialized_with_id<T>::value>::type>
        {
            register_class_name()
            {
                id_registry::instance().register_factory_function(
                    T::hpx_serialization_get_name_impl(),
                    &factory_function);
            }

            static void* factory_function()
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
        register_class_name<T, typename std::enable_if<
            traits::is_serialized_with_id<T>::value>::type>
                register_class_name<T, typename std::enable_if<
                    traits::is_serialized_with_id<T>::value>::type>::instance;

        template <std::uint32_t desc>
        std::string get_constant_entry_name();

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

    } // detail

}}

#include <hpx/config/warnings_suffix.hpp>

#define HPX_SERIALIZATION_ADD_CONSTANT_ENTRY(String, Id)                       \
    namespace hpx { namespace serialization { namespace detail {               \
        template <> std::string get_constant_entry_name<Id>()                  \
        {                                                                      \
            return BOOST_PP_STRINGIZE(String);                                 \
        }                                                                      \
        template add_constant_entry<Id>                                        \
            add_constant_entry<Id>::instance;                                  \
    }}}                                                                        \
/**/

#endif
