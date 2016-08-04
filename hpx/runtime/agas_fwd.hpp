//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_AGAS_FWD_HPP
#define HPX_RUNTIME_AGAS_FWD_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/util/function.hpp>

#include <string>

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

    enum service_mode
    {
        service_mode_invalid = -1,
        service_mode_bootstrap = 0,
        service_mode_hosted = 1
    };
}}

#endif /*HPX_RUNTIME_AGAS_FWD_HPP*/
