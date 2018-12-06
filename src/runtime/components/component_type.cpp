//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/throw_exception.hpp>

#include <hpx/lcos/local/spinlock.hpp>

#include <hpx/util/atomic_count.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/reinitializable_static.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>

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

    namespace detail
    {
        // the entries in this array need to be in exactly the same sequence
        // as the values defined in the component_type enumerator
        char const* const names[] =
        {
            "component_runtime_support",                        /*  0 */
            "component_plain_function",                         /*  1 */
            "component_memory",                                 /*  2 */
            "component_base_lco",                               /*  3 */
            "component_base_lco_with_value_unmanaged",          /*  4 */
            "component_base_lco_with_value",                    /*  5 */
            "component_latch",                                  /*  6 (0x60006) */
            "component_barrier",                                /*  7 (0x70004) */
            "component_flex_barrier",                           /*  8 (0x80004) */
            "component_promise",                                /*  9 (0x90006) */

            "component_agas_locality_namespace",                /* 10 */
            "component_agas_primary_namespace",                 /* 11 */
            "component_agas_component_namespace",               /* 12 */
            "component_agas_symbol_namespace",                  /* 13 */
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
            naming::gid_type locality = agas_client.get_local_locality();

            if (enabled)
            {
                type = agas_client.register_factory(locality, name);
                if (component_invalid == type) {
                    HPX_THROW_EXCEPTION(duplicate_component_id,
                        "get_agas_component_type",
                        std::string("the component name ") + name +
                        " is already in use");
                }
            }
            else {
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

