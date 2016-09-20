//  Copyright (c)      2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/throw_exception.hpp>
#include <hpx/runtime/actions/detail/action_factory.hpp>
#include <hpx/util/assert.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace hpx { namespace actions { namespace detail
{
    action_registry::action_registry()
      : max_id_(0)
    {}

    void action_registry::register_factory(std::string const& type_name, ctor_t ctor)
    {
        HPX_ASSERT(ctor != nullptr);

        typename_to_ctor_.emplace(std::string(type_name), ctor);

        // populate cache
        typename_to_id_t::const_iterator it = typename_to_id_.find(type_name);
        if (it != typename_to_id_.end())
            cache_id(it->second, ctor);
    }

    void action_registry::register_typename(std::string const& type_name, std::uint32_t id)
    {
        HPX_ASSERT(id != invalid_id);

        std::pair<typename_to_id_t::iterator, bool> p =
            typename_to_id_.emplace(type_name, id);

        if (!p.second)
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "action_registry::register_typename",
                "failed to insert " + type_name +
                " into typename to id registry.");
        }

        // populate cache
        typename_to_ctor_t::const_iterator it =
            typename_to_ctor_.find(type_name);
        if (it != typename_to_ctor_.end())
            cache_id(id, it->second);

        if (id > max_id_) max_id_ = id;
    }

    // This makes sure that the registries are consistent.
    void action_registry::fill_missing_typenames()
    {
        // Register all type-names and ssign missing ids
        for (std::string const& str : get_unassigned_typenames())
            register_typename(str, ++max_id_);

        // Go over all registered mappings from type-names to ids and
        // fill in missing id to constructor mappings.
        for (auto const& d: typename_to_id_)
        {
            typename_to_ctor_t::const_iterator it =
                typename_to_ctor_.find(d.first);
            if (it != typename_to_ctor_.end())
                cache_id(d.second, it->second);
        }

        // Go over all registered mappings from type-names to ctors and
        // fill in missing id to constructor mappings.
        for (auto const& d: typename_to_ctor_)
        {
            typename_to_id_t::const_iterator it =
                typename_to_id_.find(d.first);
            HPX_ASSERT(it != typename_to_id_.end());
            cache_id(it->second, d.second);
        }
    }

    std::uint32_t action_registry::try_get_id(std::string const& type_name) const
    {
        typename_to_id_t::const_iterator it =
            typename_to_id_.find(type_name);
        if (it == typename_to_id_.end())
            return invalid_id;

        return it->second;
    }

    std::vector<std::string> action_registry::get_unassigned_typenames() const
    {
        typedef typename_to_ctor_t::value_type value_type;

        std::vector<std::string> result;

        for (const value_type& v: typename_to_ctor_)
            if (!typename_to_id_.count(v.first))
                result.push_back(v.first);

        return result;
    }

    std::uint32_t action_registry::get_id(std::string const& type_name)
    {
        std::uint32_t id = instance().try_get_id(type_name);

        if (id == invalid_id)
        {
            HPX_THROW_EXCEPTION(serialization_error,
                "action_registry::get_id",
                "Unknown typename: " + type_name + "\n" +
                instance().collect_registered_typenames());
        }

        return id;
    }

    base_action* action_registry::create(
        std::uint32_t id, bool with_continuation, std::string const* name)
    {
        action_registry& this_ = instance();

        if (id >= this_.cache_.size())
        {
            std::string msg(
                "Unknown type desciptor " + std::to_string(id));
#if defined(HPX_DEBUG)
            if (name != nullptr)
            {
                msg += ", for typename " + *name + "\n";
                msg += this_.collect_registered_typenames();
            }
#endif
            HPX_THROW_EXCEPTION(serialization_error,
                "action_registry::create", msg);
        }
        ctor_t ctor = this_.cache_[id];
        HPX_ASSERT(ctor != nullptr);
        return ctor(with_continuation);
    }

    action_registry& action_registry::instance()
    {
        static action_registry this_;
        return this_;
    }

    void action_registry::cache_id(std::uint32_t id, action_registry::ctor_t ctor)
    {
        if (id >= cache_.size())
        {
            cache_.resize(id + 1, nullptr);
            cache_[id] = nullptr;
            return;
        }

        if (cache_[id] == nullptr)
        {
            cache_[id] = ctor;
        }
    }

    std::string action_registry::collect_registered_typenames()
    {
#if defined(HPX_DEBUG)
        std::string msg("known constructors:\n");

        for (auto const& desc : typename_to_ctor_)
        {
            msg += desc.first + "\n";
        }

        msg += "\nknown typenames:\n";
        for (auto const& desc : typename_to_id_)
        {
            msg += desc.first + " (";
            msg += std::to_string(desc.second) + ")\n";
        }
        return msg;
#else
        return std::string();
#endif
    }

}}}
