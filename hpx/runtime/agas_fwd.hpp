//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_AGAS_FWD_HPP
#define HPX_RUNTIME_AGAS_FWD_HPP

#include <hpx/config.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/runtime_configuration/agas_service_mode.hpp>

#include <cstdint>
#include <string>

namespace hpx { namespace components
{
    typedef std::int32_t component_type;
}}

namespace hpx { namespace agas
{
    typedef hpx::util::function<
        void(std::string const&, components::component_type)
    > iterate_types_function_type;

    struct HPX_EXPORT component_namespace;
    struct HPX_EXPORT locality_namespace;
    struct HPX_EXPORT primary_namespace;
    struct HPX_EXPORT symbol_namespace;
    namespace server
    {
        struct HPX_EXPORT component_namespace;
        struct HPX_EXPORT locality_namespace;
        struct HPX_EXPORT primary_namespace;
        struct HPX_EXPORT symbol_namespace;
    }
    struct HPX_API_EXPORT addressing_service;
}}

#endif /*HPX_RUNTIME_AGAS_FWD_HPP*/
