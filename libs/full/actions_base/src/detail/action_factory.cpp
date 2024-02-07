//  Copyright (c)      2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/actions_base/detail/action_factory.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/errors.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hpx::actions::detail {

    action_registry::action_registry()
      : max_id_(0)
    {
    }

    action_registry::~action_registry() = default;

    void action_registry::register_factory(
        std::string const& type_name, ctor_t ctor, ctor_t ctor_cont)
    {
        HPX_ASSERT(ctor != nullptr && ctor_cont != nullptr);

        typename_to_ctor_.emplace(
            std::string(type_name), std::make_pair(ctor, ctor_cont));

        // populate cache
        if (auto const it = typename_to_id_.find(type_name);
            it != typename_to_id_.end())
        {
            cache_id(it->second, ctor, ctor_cont);
        }
    }

    void action_registry::register_typename(
        std::string const& type_name, std::uint32_t id)
    {
        HPX_ASSERT(id != invalid_id);

        if (auto const [it, inserted] = typename_to_id_.emplace(type_name, id);
            !inserted && it->second != 0)
        {
            HPX_THROW_EXCEPTION(hpx::error::invalid_status,
                "action_registry::register_typename",
                "failed to insert {} into typename to id registry.", type_name);
        }

        // populate cache
        if (auto const it = typename_to_ctor_.find(type_name);
            it != typename_to_ctor_.end())
        {
            cache_id(id, it->second.first, it->second.second);
        }

        if (id > max_id_)
        {
            max_id_ = id;
        }
    }

    // This makes sure that the registries are consistent.
    void action_registry::fill_missing_typenames()
    {
        // Register all type-names and assign missing ids
        for (std::string const& str : get_unassigned_typenames())
        {
            register_typename(str, ++max_id_);
        }

        // Go over all registered mappings from type-names to ids and
        // fill in missing id to constructor mappings.
        for (auto const& d : typename_to_id_)
        {
            if (auto const it = typename_to_ctor_.find(d.first);
                it != typename_to_ctor_.end())
            {
                cache_id(d.second, it->second.first, it->second.second);
            }
        }

        // Go over all registered mappings from type-names to ctors and
        // fill in missing id to constructor mappings.
        for (auto const& d : typename_to_ctor_)
        {
            auto const it = typename_to_id_.find(d.first);
            HPX_ASSERT(it != typename_to_id_.end());
            cache_id(it->second, d.second.first, d.second.second);
        }
    }

    std::uint32_t action_registry::try_get_id(
        std::string const& type_name) const
    {
        auto const it = typename_to_id_.find(type_name);
        if (it == typename_to_id_.end())
        {
            return invalid_id;
        }
        return it->second;
    }

    std::vector<std::string> action_registry::get_unassigned_typenames() const
    {
        std::vector<std::string> result;

        for (auto const& [k, _] : typename_to_ctor_)
        {
            if (!typename_to_id_.count(k))
            {
                result.push_back(k);
            }
        }

        return result;
    }

    std::uint32_t action_registry::get_id(std::string const& type_name)
    {
        std::uint32_t const id = instance().try_get_id(type_name);

        if (id == invalid_id)
        {
            HPX_THROW_EXCEPTION(hpx::error::serialization_error,
                "action_registry::get_id", "Unknown typename: {}\n{}",
                type_name, instance().collect_registered_typenames());
        }

        return id;
    }

    base_action* action_registry::create(std::uint32_t id,
        bool with_continuation, [[maybe_unused]] std::string const* name)
    {
        action_registry const& this_ = instance();

        if (id >= this_.cache_.size())
        {
            std::string msg(
                "Unknown type descriptor (unknown id)" + std::to_string(id));
#if defined(HPX_DEBUG)
            if (name != nullptr)
            {
                msg += ", for typename " + *name;
            }
            msg += this_.collect_registered_typenames();
#endif
            HPX_THROW_EXCEPTION(hpx::error::serialization_error,
                "action_registry::create", msg);
        }

        std::pair<ctor_t, ctor_t> const& ctors =
            this_.cache_[static_cast<std::size_t>(id)];
        if (ctors.first == nullptr || ctors.second == nullptr)    // -V108
        {
            std::string msg("Unknown type descriptor (undefined constructors)" +
                std::to_string(id));
#if defined(HPX_DEBUG)
            if (name != nullptr)
            {
                msg += ", for typename " + *name;
            }
            msg += this_.collect_registered_typenames();
#endif
            HPX_THROW_EXCEPTION(hpx::error::serialization_error,
                "action_registry::create", msg);
        }
        return !with_continuation ? ctors.first() : ctors.second();
    }

    action_registry& action_registry::instance()
    {
        static action_registry this_;
        return this_;
    }

    void action_registry::cache_id(std::uint32_t id,
        action_registry::ctor_t ctor, action_registry::ctor_t ctor_cont)
    {
        std::size_t const id_ = static_cast<std::size_t>(id);
        if (id_ >= cache_.size())
        {
            cache_.resize(id_ + 1, std::pair<ctor_t, ctor_t>(nullptr, nullptr));
            cache_[id_] = std::make_pair(ctor, ctor_cont);
            return;
        }

        if (cache_[id_].first == nullptr || cache_[id_].second == nullptr)
        {
            cache_[id_] = std::make_pair(ctor, ctor_cont);
        }
    }

    std::string action_registry::collect_registered_typenames() const
    {
#if defined(HPX_DEBUG)
        std::string msg("\nknown constructors:\n");

        for (auto const& [desc, _] : typename_to_ctor_)
        {
            msg += desc + "\n";
        }

        msg += "\nknown typenames:\n";
        for (auto const& [desc, id] : typename_to_id_)
        {
            msg += desc + " (";
            msg += std::to_string(id) + ")\n";
        }

        return msg;
#else
        return {};
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    std::uint32_t get_action_id_from_name(char const* action_name)
    {
        using hpx::actions::detail::action_registry;
        return action_registry::get_id(action_name);
    }
}    // namespace hpx::actions::detail

#endif
