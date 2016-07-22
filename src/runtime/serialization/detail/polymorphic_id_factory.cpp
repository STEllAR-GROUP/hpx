//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/detail/polymorphic_id_factory.hpp>

#include <map>
#include <string>
#include <vector>

#include <boost/cstdint.hpp>

namespace hpx { namespace serialization { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    id_registry& id_registry::instance()
    {
        util::static_<id_registry> inst;
        return inst.get();
    }

    void id_registry::cache_id(boost::uint32_t id, ctor_t ctor)
    {
        if (id >= cache.size()) //-V104
        {
            cache.resize(id + 1, nullptr); //-V106
            cache[id] = ctor; //-V108
        }
        else if (cache[id] == nullptr)
        {
            cache[id] = ctor; //-V108
        }
    }

    void id_registry::register_factory_function(
        const std::string& type_name, ctor_t ctor)
    {
        HPX_ASSERT(ctor != nullptr);

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

    void id_registry::register_typename(
        const std::string& type_name, boost::uint32_t id)
    {
        HPX_ASSERT(id != invalid_id);

        std::pair<typename_to_id_t::iterator, bool> p =
#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 408000
        typename_to_id.emplace(type_name, id);
#else
        typename_to_id.insert(
            typename_to_id_t::value_type(type_name, id)
        );
#endif
        if (!p.second)
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "polymorphic_id_factory::register_typename",
                "failed to insert " + type_name +
                " into typename_to_id_t registry");
            return;
        }

        // populate cache
        typename_to_ctor_t::const_iterator it =
            typename_to_ctor.find(type_name);
        if (it != typename_to_ctor.end())
            cache_id(id, it->second);

        if (id > max_id) max_id = id;
    }

    // This makes sure that the registries are consistent.
    void id_registry::fill_missing_typenames()
    {
        // Register all type-names and assign missing ids
        for (std::string const& str : get_unassigned_typenames())
            register_typename(str, ++max_id);

        // Go over all registered mappings from type-names to ids and
        // fill in missing id to constructor mappings.
        for (auto const& d : typename_to_id)
        {
            typename_to_ctor_t::const_iterator it =
                typename_to_ctor.find(d.first);
            if (it != typename_to_ctor.end())
                cache_id(d.second, it->second);
        }

        // Go over all registered mappings from type-names to
        // constructors and fill in missing id to constructor mappings.
        for (auto const& d : typename_to_ctor)
        {
            typename_to_id_t::const_iterator it =
                typename_to_id.find(d.first);
            HPX_ASSERT(it != typename_to_id.end());
            cache_id(it->second, d.second);
        }
    }

    boost::uint32_t id_registry::try_get_id(const std::string& type_name) const
    {
        typename_to_id_t::const_iterator it =
            typename_to_id.find(type_name);
        if (it == typename_to_id.end())
            return invalid_id;

        return it->second;
    }

    std::vector<std::string> id_registry::get_unassigned_typenames() const
    {
        typedef typename_to_ctor_t::value_type value_type;

        std::vector<std::string> result;

        // O(Nlog(M)) ?
        for (const value_type& v : typename_to_ctor)
            if (!typename_to_id.count(v.first))
                result.push_back(v.first);

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    polymorphic_id_factory& polymorphic_id_factory::instance()
    {
        hpx::util::static_<polymorphic_id_factory> factory;
        return factory.get();
    }

    boost::uint32_t polymorphic_id_factory::get_id(const std::string& type_name)
    {
        boost::uint32_t id = id_registry::instance().try_get_id(type_name);

        if (id == id_registry::invalid_id)
        {
            HPX_THROW_EXCEPTION(serialization_error
                , "polymorphic_id_factory::get_id"
                , "Unknown typename: " + type_name);
        }

        return id;
    }

    std::string polymorphic_id_factory::collect_registered_typenames()
    {
        std::string msg("known constructors:\n");

        for (auto const& desc : id_registry::instance().typename_to_ctor)
        {
            msg += desc.first + "\n";
        }

        msg += "\nknown typenames:\n";
        for (auto const& desc : id_registry::instance().typename_to_id)
        {
            msg += desc.first + " (";
            msg += std::to_string(desc.second) + ")\n";
        }

        return msg;
    }
}}}
