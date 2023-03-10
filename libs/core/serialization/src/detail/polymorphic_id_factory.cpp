//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/serialization/detail/polymorphic_id_factory.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace hpx::serialization::detail {

    ///////////////////////////////////////////////////////////////////////////
    id_registry& id_registry::instance()
    {
        util::static_<id_registry> inst;
        return inst.get();
    }

    void id_registry::cache_id(std::uint32_t id, ctor_t ctor)
    {
        if (id >= cache.size())    //-V104
        {
            cache.resize(static_cast<std::size_t>(id) + 1, nullptr);
            cache[id] = ctor;    //-V108
        }
        else if (cache[id] == nullptr)    //-V108
        {
            cache[id] = ctor;    //-V108
        }
    }

    void id_registry::register_factory_function(
        std::string const& type_name, ctor_t ctor)
    {
        HPX_ASSERT(ctor != nullptr);

        typename_to_ctor.emplace(type_name, ctor);

        // populate cache
        auto const it = typename_to_id.find(type_name);
        if (it != typename_to_id.end())
            cache_id(it->second, ctor);
    }

    void id_registry::register_typename(
        std::string const& type_name, std::uint32_t id)
    {
        HPX_ASSERT(id != invalid_id);

        std::pair<typename_to_id_t::iterator, bool> const p =
            typename_to_id.emplace(type_name, id);

        if (!p.second)
        {
            HPX_THROW_EXCEPTION(hpx::error::invalid_status,
                "polymorphic_id_factory::register_typename",
                "failed to insert {} into typename_to_id_t registry",
                type_name);
            return;
        }

        // populate cache
        auto const it = typename_to_ctor.find(type_name);
        if (it != typename_to_ctor.end())
            cache_id(id, it->second);

        if (id > max_id)
            max_id = id;
    }

    // This makes sure that the registries are consistent.
    void id_registry::fill_missing_typenames()
    {
        // Register all type-names and assign missing ids
        for (std::string const& str : get_unassigned_typenames())
            register_typename(str, ++max_id);

        // Go over all registered mappings from type-names to ids and
        // fill in missing id to constructor mappings.
        for (auto const& [fst, snd] : typename_to_id)
        {
            auto const it = typename_to_ctor.find(fst);
            if (it != typename_to_ctor.end())
                cache_id(snd, it->second);
        }

        // Go over all registered mappings from type-names to
        // constructors and fill in missing id to constructor mappings.
        for (auto const& [fst, snd] : typename_to_ctor)
        {
            typename_to_id_t::const_iterator it = typename_to_id.find(fst);
            HPX_ASSERT(it != typename_to_id.end());
            cache_id(it->second, snd);    //-V783
        }
    }

    std::uint32_t id_registry::try_get_id(std::string const& type_name) const
    {
        auto const it = typename_to_id.find(type_name);
        if (it == typename_to_id.end())
            return invalid_id;

        return it->second;
    }

    std::vector<std::string> id_registry::get_unassigned_typenames() const
    {
        std::vector<std::string> result;

        // O(N log(M)) ?
        for (auto const& [fst, snd] : typename_to_ctor)
        {
            if (!typename_to_id.count(fst))
            {
                result.push_back(fst);
            }
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    polymorphic_id_factory& polymorphic_id_factory::instance()
    {
        hpx::util::static_<polymorphic_id_factory> factory;
        return factory.get();
    }

    std::uint32_t polymorphic_id_factory::get_id(std::string const& type_name)
    {
        std::uint32_t const id = id_registry::instance().try_get_id(type_name);

        if (id == id_registry::invalid_id)
        {
            HPX_THROW_EXCEPTION(hpx::error::serialization_error,
                "polymorphic_id_factory::get_id", "Unknown typename: {}",
                type_name);
        }

        return id;
    }

    std::string polymorphic_id_factory::collect_registered_typenames()
    {
#if defined(HPX_DEBUG)
        std::string msg("known constructors:\n");

        for (auto const& [fst, snd] : id_registry::instance().typename_to_ctor)
        {
            msg += fst + "\n";
        }

        msg += "\nknown typenames:\n";
        for (auto const& [fst, snd] : id_registry::instance().typename_to_id)
        {
            msg += fst + " (";
            msg += std::to_string(snd) + ")\n";
        }

        return msg;
#else
        return std::string();
#endif
    }
}    // namespace hpx::serialization::detail
