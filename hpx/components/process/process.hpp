// Copyright (c) 2016 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_PROCESS_MAR_11_2016_1255PM)
#define HPX_COMPONENTS_PROCESS_MAR_11_2016_1255PM

#include <hpx/config.hpp>
#include <hpx/runtime/components/new.hpp>
#include <hpx/runtime/naming/id_type.hpp>

#include <hpx/components/process/child.hpp>

#include <type_traits>

namespace hpx { namespace components { namespace process
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ... Ts>
    child execute(hpx::id_type const& id, Ts && ... ts)
    {
        return hpx::new_<child>(id, std::forward<Ts>(ts)...);
    }
}}}

#endif
