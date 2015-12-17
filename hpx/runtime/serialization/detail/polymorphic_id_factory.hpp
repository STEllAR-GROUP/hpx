//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_POLYMORPHIC_ID_FACTORY_HPP
#define HPX_SERIALIZATION_POLYMORPHIC_ID_FACTORY_HPP

#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/runtime/serialization/detail/polymorphic_intrusive_factory.hpp>
#include <hpx/traits/polymorphic_traits.hpp>
#include <hpx/util/static.hpp>
#include <hpx/util/safe_lexical_cast.hpp>
#include <hpx/util/move.hpp>
#include <hpx/exception.hpp>

#include <boost/preprocessor/stringize.hpp>
#include <boost/unordered_map.hpp>
#include <boost/atomic.hpp>
#include <boost/mpl/bool.hpp>

#include <map>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace serialization {

    namespace detail
    {
        class id_registry : boost::noncopyable
        {
        public:
            typedef void* (*ctor_t) ();
            typedef std::map<std::string, ctor_t> typename_to_ctor_t;
            typedef std::map<std::string, boost::uint32_t> typename_to_id_t;
            typedef std::vector<ctor_t> cache_t;

            HPX_STATIC_CONSTEXPR boost::uint32_t invalid_id = ~0u;

            void register_factory_function(const std::string& type_name,
                ctor_t ctor)
            {
#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 408000
                typename_to_ctor.emplace(type_name, ctor);
#else
                typename_to_ctor.insert(
                    typename_to_ctor_t::value_type(type_name, ctor)
                );
#endif
                // populate cache
                typename_to_id_t::const_iterator it =
                    typename_to_id.find(type_name);
                if (it != typename_to_id.end())
                    cache_id(it->second, ctor);
            }

            void register_typename(const std::string& type_name,
                boost::uint32_t id)
            {
#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 408000
                typename_to_id.emplace(type_name, id);
#else
                typename_to_id.insert(
                    typename_to_id_t::value_type(type_name, id)
                );
#endif
                // populate cache
                typename_to_ctor_t::const_iterator it =
                    typename_to_ctor.find(type_name);
                if (it != typename_to_ctor.end())
                    cache_id(id, it->second);

                if (id > max_id) max_id = id;
            }

            boost::uint32_t try_get_id(const std::string& type_name) const
            {
                typename_to_id_t::const_iterator it =
                    typename_to_id.find(type_name);
                if (it == typename_to_id.end())
                    return invalid_id;

                return it->second;
            }

            boost::uint32_t get_max_registered_id() const
            {
                return max_id;
            }

            std::vector<std::string> get_unassigned_typenames() const
            {
                typedef typename_to_ctor_t::value_type value_type;

                std::vector<std::string> result;

                // O(Nlog(M)) ?
                for (const value_type& v : typename_to_ctor)
                    if (!typename_to_id.count(v.first))
                        result.push_back(v.first);

                return result;
            }

            HPX_EXPORT static id_registry& instance();

        private:
            id_registry() : max_id(0u) {}

            friend struct ::hpx::util::static_<id_registry>;
            friend class polymorphic_id_factory;

            void cache_id(boost::uint32_t id, ctor_t ctor)
            {
                if (id >= cache.size()) //-V104
                    cache.resize(id + 1, NULL); //-V106
                cache[id] = ctor; //-V108
            }

            boost::uint32_t max_id;
            typename_to_ctor_t typename_to_ctor;
            typename_to_id_t typename_to_id;
            cache_t cache;
        };

        class polymorphic_id_factory : boost::noncopyable
        {
            typedef id_registry::ctor_t ctor_t;
            typedef id_registry::typename_to_ctor_t typename_to_ctor_t;
            typedef id_registry::typename_to_id_t typename_to_id_t;
            typedef id_registry::cache_t cache_t;

        public:
            template <class T>
            static T* create(boost::uint32_t id)
            {
                const cache_t& vec = id_registry::instance().cache;

                if (id > vec.size()) //-V104
                    HPX_THROW_EXCEPTION(serialization_error
                      , "polymorphic_id_factory::create"
                      , "Unknown type descriptor " +
                            util::safe_lexical_cast<std::string>(id));

                ctor_t ctor = vec[id]; //-V108
                HPX_ASSERT(ctor != NULL);
                return static_cast<T*>(ctor());
            }

            static boost::uint32_t get_id(const std::string& type_name)
            {
                boost::uint32_t id = id_registry::instance().
                    try_get_id(type_name);

                if (id == id_registry::invalid_id)
                    HPX_THROW_EXCEPTION(serialization_error
                      , "polymorphic_id_factory::get_id"
                      , "Unknown typename: " + type_name);

                return id;
            }

        private:
            polymorphic_id_factory() {}

            HPX_EXPORT static polymorphic_id_factory& instance();

            friend struct hpx::util::static_<polymorphic_id_factory>;
        };

        template <class T>
        struct register_class_name<T, typename boost::enable_if<
            traits::is_serialized_with_id<T> >::type>
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
        register_class_name<T, typename boost::enable_if<
            traits::is_serialized_with_id<T> >::type>
                register_class_name<T, typename boost::enable_if<
                    traits::is_serialized_with_id<T> >::type>::instance;

        template <boost::uint32_t desc>
        std::string get_constant_entry_name();

        template <boost::uint32_t Id>
        struct add_constant_entry
        {
            add_constant_entry()
            {
                id_registry::instance().register_typename(
                        get_constant_entry_name<Id>(), Id);
            }

            static add_constant_entry instance;
        };

        template <boost::uint32_t Id>
        add_constant_entry<Id> add_constant_entry<Id>::instance;

    } // detail

}}

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
