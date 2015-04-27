//  Copyright (c) 2014 Anton Bikineev
//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_POLYMORPHIC_CENTRALIZED_FACTORY_HPP
#define HPX_SERIALIZATION_POLYMORPHIC_CENTRALIZED_FACTORY_HPP

#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/runtime/serialization/detail/polymorphic_intrusive_factory.hpp>
#include <hpx/traits/polymorphic_traits.hpp>
#include <hpx/util/static.hpp>
#include <hpx/util/safe_lexical_cast.hpp>
#include <hpx/exception.hpp>

#include <boost/preprocessor/stringize.hpp>
#include <boost/unordered_map.hpp>
#include <boost/atomic.hpp>
#include <boost/mpl/bool.hpp>

#include <map>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace serialization { namespace detail
{
    struct type_name_desc_type
    {
        BOOST_STATIC_CONSTEXPR boost::uint32_t invalid = ~0u;

        type_name_desc_type(std::string name,
                boost::uint32_t desc = invalid)
            : name(std::move(name)), desc(desc)
        {}

        std::string name;
        boost::uint32_t desc;
    };

    BOOST_FORCEINLINE
    static bool operator<(
            const hpx::serialization::detail::type_name_desc_type& first,
            const hpx::serialization::detail::type_name_desc_type& second)
    {
        return first.desc < second.desc;
    }

    class HPX_EXPORT polymorphic_centralized_factory: boost::noncopyable
    {
    public:
        typedef void* (*ctor_type) ();
        typedef std::multimap<type_name_desc_type, ctor_type> ctor_map_type;

        static void add_factory_function(const std::string& name, ctor_type fun)
        {
            if(name.empty())
            {
                HPX_THROW_EXCEPTION(serialization_error
                  , "polymorphic_intrusive_factory::register_class"
                  , "Cannot register a factory with an empty name");
            }
            auto& map = polymorphic_centralized_factory::instance().map_;
            map.emplace(type_name_desc_type(name), fun);
        }

        template <class T>
        static T* create(boost::uint32_t desc)
        {
            const auto& map = polymorphic_centralized_factory::instance().map_;
            auto it = map.find(type_name_desc_type("", desc));

            if (it != map.end())
                return static_cast<T*>(it->second());

            HPX_THROW_EXCEPTION(serialization_error
              , "polymorphic_intrusive_factory::create"
              , "Unknown type descriptor" + util::safe_lexical_cast<std::string>(desc));
        }

        static std::vector<std::string> get_registered_typenames()
        {
            typedef typename ctor_map_type::value_type value_type;
            const auto& map = polymorphic_centralized_factory::instance().map_;

            std::vector<std::string> result;
            result.reserve(map.size());

            std::for_each(map.begin(), map.end(),
                    [&](const value_type& s){
                        result.push_back(s.first.name);
                    });

            return result;
        }

        static void register_type_descriptor(
            const std::string& type_id, boost::uint32_t desc)
        {
            auto & map = polymorphic_centralized_factory::instance().map_;

            auto it = find_by_typename(type_id);
            if (it != map.end())
            {
                ctor_type ctor = std::move(it->second);
                map.erase(it);
                map.emplace(type_name_desc_type(type_id, desc), ctor);
                return;
            }

            HPX_THROW_EXCEPTION(serialization_error
              , "polymorphic_intrusive_factory::register_type_descriptor"
              , "Unknown typename");
        }

        static boost::uint32_t get_descriptor_by_typeid(const std::string& str)
        {
            const auto & map = polymorphic_centralized_factory::instance().map_;

            auto it = find_by_typename(str);
            if (it != map.end())
            {
                boost::uint32_t value = it->first.desc;
                HPX_ASSERT(value != type_name_desc_type::invalid);
                return value;
            }

            HPX_THROW_EXCEPTION(serialization_error
              , "polymorphic_intrusive_factory::get_descriptor_by_typeid"
              , "Unknown typename");
        }

    private:
        friend struct hpx::util::static_<polymorphic_centralized_factory>;

        static ctor_map_type::iterator find_by_typename(const std::string& name)
        {
            typedef typename ctor_map_type::value_type value_type;
            auto& map = polymorphic_centralized_factory::instance().map_;

            return std::find_if(map.begin(), map.end(),
                [&](const value_type& s){
                    return s.first.name == name;
                });
        }

        static polymorphic_centralized_factory& instance()
        {
            hpx::util::static_<polymorphic_centralized_factory> factory;
            return factory.get();
        }

        ctor_map_type map_;
    };

    template <class T>
    struct register_class_name<T, typename boost::enable_if<
        traits::does_require_centralization<T> >::type>
    {
        register_class_name()
        {
            T* t = 0; //dirty
            polymorphic_centralized_factory::add_factory_function(
                t->T::hpx_serialization_get_name(), //non-virtual call
                &factory_function
            );
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
        traits::does_require_centralization<T> >::type>
            register_class_name<T, typename boost::enable_if<
                traits::does_require_centralization<T> >::type>::instance;


    template <boost::uint32_t desc>
    std::string get_constant_entry_name();

    template <boost::uint32_t desc>
    struct add_constant_entry
    {
        add_constant_entry(const std::string& str)
        {
            polymorphic_centralized_factory::register_type_descriptor(str, desc);
        }

        static add_constant_entry instance;
    };

    template <boost::uint32_t desc>
    add_constant_entry<desc> add_constant_entry<desc>::instance(get_constant_entry_name<desc>());

}}}

// bikineev: usage?
#define HPX_SERIALIZATION_ADD_CONSTANT_ENTRY(String, Desc)                             \
    namespace hpx { namespace serialization { namespace detail {                       \
        template <> std::string get_constant_entry_name<Desc>() { return String; }     \
        template add_constant_entry<Desc> add_constant_entry<Desc>::instance;          \
    }}}                                                                                \
/**/

#endif
