//  Copyright (c) 2007-2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/static_reinit/reinitializable_static.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/thread_support/atomic_count.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    namespace detail
    {
        struct component_database
        {
            struct component_entry
            {
                component_entry()
                  : enabled_(false)
                  , instance_count_(0)
                  , deleter_(nullptr)
                {}

                // note, this done to be able to put the entry into the map.
                // a component_entry should not be moved otherwise, but this
                // saves us a dynamic allocation
                component_entry(component_entry &&)
                  : enabled_(false)
                  , instance_count_(0)
                  , deleter_(nullptr)
                {}

                bool enabled_;
                util::atomic_count instance_count_;
                component_deleter_type deleter_;
            };

        private:
            typedef hpx::lcos::local::spinlock mutex_type;
            typedef
                std::map<component_type, component_entry> map_type;

            static mutex_type& mtx()
            {
                static mutex_type mtx_;
                return mtx_;
            }

            static map_type& data()
            {
                static map_type map;
                return map;
            }

        public:
            static component_entry& get_entry(component_type type)
            {
                std::lock_guard<mutex_type> l(mtx());
                auto& d = data();

                auto it = d.find(type);
                if (it == d.end())
                {
                    it = d.emplace(type, component_entry()).first;
                }

                return it->second;
            }

            static bool enumerate_instance_counts(
                util::unique_function_nonser<bool(component_type)> const& f)
            {
                std::vector<component_type> types;

                {
                    std::lock_guard<mutex_type> l(mtx());
                    types.reserve(data().size());

                    for (auto const& e : data())
                    {
                        types.push_back(e.first);
                    }
                }

                for (component_type t : types)
                {
                    if (!f(t))
                        return false;
                }

                return true;
            }
        };
    }

    bool& enabled(component_type type)
    {
        return detail::component_database::get_entry(type).enabled_;
    }

    util::atomic_count& instance_count(component_type type)
    {
        return detail::component_database::get_entry(type).instance_count_;
    }

    component_deleter_type& deleter(component_type type)
    {
        return detail::component_database::get_entry(type).deleter_;
    }

    bool enumerate_instance_counts(
        util::unique_function_nonser<bool(component_type)> const& f)
    {
        return detail::component_database::enumerate_instance_counts(f);
    }

    namespace detail
    {
        // the entries in this array need to be in exactly the same sequence
        // as the values defined in the component_type enumerator
        char const* const names[] = {
            "component_runtime_support",               /*  0 */
            "component_plain_function",                /*  1 */
            "component_base_lco",                      /*  2 */
            "component_base_lco_with_value_unmanaged", /*  3 */
            "component_base_lco_with_value",           /*  4 */
            "component_latch",                         /*  5 (0x50005) */
            "component_barrier",                       /*  6 (0x60006) */
            "component_promise",                       /*  7 (0x70004) */

            "component_agas_locality_namespace",  /*  8 */
            "component_agas_primary_namespace",   /*  9 */
            "component_agas_component_namespace", /* 10 */
            "component_agas_symbol_namespace",    /* 11 */
        };
    }

    // Return the string representation for a given component type id
    std::string const get_component_type_name(std::int32_t type)
    {
        std::string result;

        if (type == component_invalid)
            result = "component_invalid";
        else if ((type < component_last) && (get_derived_type(type) == 0))
            result = components::detail::names[type];
        else if (get_derived_type(type) <
            component_last && (get_derived_type(type) != 0))
            result = components::detail::names[get_derived_type(type)];
        else
            result = "component";

        if (type == get_base_type(type) || component_invalid == type)
            result += "[" + std::to_string(type) + "]";
        else {
            result += "[" +
                std::to_string
                  (static_cast<int>(get_derived_type(type))) +
                "(" + std::to_string
                    (static_cast<int>(get_base_type(type))) + ")"
                "]";
        }
        return result;
    }

    namespace detail {
        component_type get_agas_component_type(const char* name,
            const char* base_name, component_type base_type, bool enabled)
        {
            component_type type = component_invalid;
            naming::resolver_client& agas_client = hpx::naming::get_agas_client();

            if (enabled)
            {
                naming::gid_type locality = agas_client.get_local_locality();
                type = agas_client.register_factory(locality, name);
                if (component_invalid == type) {
                    HPX_THROW_EXCEPTION(duplicate_component_id,
                        "get_agas_component_type",
                        std::string("the component name ") + name +
                        " is already in use");
                }
            }
            else
            {
                type = agas_client.get_component_id(name);
            }

            if (base_name)
            {
                // NOTE: This assumes that the derived component is loaded.
                if (base_type == component_invalid)
                {
                    base_type = agas_client.get_component_id(base_name);
                }
                type = derived_component_type(type, base_type);
            }

            return type;
        }
    }
}}

