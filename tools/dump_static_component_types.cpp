////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <limits.h>
#include <stdio.h>

#include <boost/cstdint.hpp>

#include <hpx/runtime/components/component_type.hpp>

using hpx::components::get_component_type_name;
using hpx::components::get_base_type;
using hpx::components::get_derived_type;
    
using hpx::components::component_invalid;
using hpx::components::component_runtime_support;
using hpx::components::component_memory;
using hpx::components::component_memory_block;
using hpx::components::component_base_lco;
using hpx::components::component_base_lco_with_value;
using hpx::components::component_future;
using hpx::components::component_value_adaptor;
using hpx::components::component_barrier;
using hpx::components::component_thread;
using hpx::components::component_dataflow_variable;
using hpx::components::component_thunk;
using hpx::components::component_dataflow_block;
using hpx::components::component_last;
using hpx::components::component_first_dynamic;

namespace {

template <typename T>
std::string binary(T x, T space = 4)
{
    std::string r("");

    for (T z = 1 << ((sizeof(T) * CHAR_BIT) - 1), i = 0; z > 0; z >>= 1, ++i)
    {
        if (i == space)
        {
            r += " ";
            i = 0;
        }

        r += (((x & z) == z) ? "1" : "0");
    }
    return r;
}

template <typename T>
void print_component(T type, char const* identifier)
{
    printf("identifier = %s\n"
           "name       = %s\n"
           "type       = %s\n"
           "base       = %s\n"
           "derived    = %s\n\n",
          identifier,
          get_component_type_name(type).c_str(),
          binary(boost::uint32_t(type)).c_str(),
          binary(boost::uint32_t(get_base_type(type))).c_str(),
          binary(boost::uint32_t(get_derived_type(type))).c_str());
}

}

#define HPX_PRINT_COMPONENT(c) print_component(c, #c)

int main()
{
    HPX_PRINT_COMPONENT(component_invalid);
    HPX_PRINT_COMPONENT(component_runtime_support);
    HPX_PRINT_COMPONENT(component_memory);
    HPX_PRINT_COMPONENT(component_memory_block);
    HPX_PRINT_COMPONENT(component_base_lco);
    HPX_PRINT_COMPONENT(component_base_lco_with_value);
    HPX_PRINT_COMPONENT(component_future);
    HPX_PRINT_COMPONENT(component_value_adaptor);
    HPX_PRINT_COMPONENT(component_barrier);
    HPX_PRINT_COMPONENT(component_thread);
    HPX_PRINT_COMPONENT(component_dataflow_variable);
    HPX_PRINT_COMPONENT(component_thunk);
    HPX_PRINT_COMPONENT(component_dataflow_block);
    HPX_PRINT_COMPONENT(component_last);
    HPX_PRINT_COMPONENT(component_first_dynamic);
}

