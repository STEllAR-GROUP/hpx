//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx_fwd.hpp

#if !defined(HPX_HPX_FWD_MAR_24_2008_1119AM)
#define HPX_HPX_FWD_MAR_24_2008_1119AM

#include <hpx/config.hpp>

#if defined(HPX_MSVC)
#  pragma message("The header hpx_fwd.hpp is deprecated")
#else
#  warning "The header hpx_fwd.hpp is deprecated"
#endif

#include <cstdlib>
#include <string>
#include <vector>

#include <boost/config.hpp>
#include <boost/version.hpp>

#if defined(HPX_WINDOWS)
#if !defined(WIN32)
#  define WIN32
#endif
#include <winsock2.h>
#include <windows.h>
#endif

#include <boost/intrusive_ptr.hpp>
#include <boost/cstdint.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/system/error_code.hpp>
#include <boost/type_traits/remove_reference.hpp>

#include <hpx/exception_fwd.hpp>
#include <hpx/lcos_fwd.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/util/function.hpp>
// ^ this has to come before the naming/id_type.hpp below
#include <hpx/util/unique_function.hpp>
#include <hpx/util/unused.hpp>
#include <hpx/runtime/agas_fwd.hpp>
#include <hpx/runtime/applier_fwd.hpp>
#include <hpx/runtime/components_fwd.hpp>
#include <hpx/runtime/find_here.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/runtime/runtime_mode.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/threads_fwd.hpp>
#include <hpx/runtime/threads/detail/combined_tagged_state.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>

#include <hpx/runtime/find_localities.hpp>
#include <hpx/runtime/get_colocation_id.hpp>
#include <hpx/runtime/get_num_localities.hpp>
#include <hpx/runtime/get_thread_name.hpp>
#include <hpx/runtime/shutdown_function.hpp>
#include <hpx/runtime/startup_function.hpp>

/// \namespace hpx
///
/// The namespace \a hpx is the main namespace of the HPX library. All classes
/// functions and variables are defined inside this namespace.
namespace hpx
{
    /// \cond NOINTERNAL

    namespace performance_counters
    {
        struct counter_info;
    }

    /// \endcond
}

// Including declarations of various API function declarations
#include <hpx/runtime/basename_registration.hpp>
#include <hpx/runtime/trigger_lco.hpp>
#include <hpx/runtime/get_locality_name.hpp>
#include <hpx/runtime/get_locality_id.hpp>
#include <hpx/runtime/get_config_entry.hpp>
#include <hpx/runtime/set_parcel_write_handler.hpp>
#include <hpx/runtime/get_os_thread_count.hpp>
#include <hpx/runtime/get_worker_thread_num.hpp>
#include <hpx/runtime/message_handler_fwd.hpp>

#include <hpx/lcos/async_fwd.hpp>
#include <hpx/lcos/async_callback_fwd.hpp>

#endif

