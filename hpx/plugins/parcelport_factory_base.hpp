//  Copyright (c)      2014 Thomas Heller
//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PLUGINS_PARCELPORT_FACTORY_BASE_HPP
#define HPX_PLUGINS_PARCELPORT_FACTORY_BASE_HPP

#include <hpx/config.hpp>
#include <hpx/util_fwd.hpp>
#include <hpx/util/runtime_configuration.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace hpx { namespace parcelset {
    class parcelport;
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a plugin_factory_base has to be used as a base class for all
    /// plugin factories.
    struct HPX_EXPORT parcelport_factory_base
    {
        virtual ~parcelport_factory_base() {}

        virtual void get_plugin_info(std::vector<std::string> & fillini) = 0;

        virtual void init(int *argc, char ***argv, util::command_line_handling &cfg) = 0;

        /// Create a new instance of a parcelport
        ///
        /// return Returns the newly created instance of the parcelport
        ///        supported by this factory
        virtual parcelset::parcelport* create(
            hpx::util::runtime_configuration const & cfg,
            util::function_nonser<void(std::size_t, char const*)> const& on_start_thread,
            util::function_nonser<void()> const& on_stop_thread) = 0;
    };
}}

#endif /*HPX_PLUGINS_PARCELPORT_FACTORY_BASE_HPP*/
