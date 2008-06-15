//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_BASE_LCO_JUN_12_2008_0852PM)
#define HPX_LCOS_BASE_LCO_JUN_12_2008_0852PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/components/component_type.hpp>
#include <hpx/components/action.hpp>
#include <hpx/runtime/threadmanager/px_thread.hpp>

namespace hpx { namespace lcos { namespace detail
{
    ///
    struct base_lco
    {
        // parcel action code: the action to be performed on the destination 
        // object 
        enum actions
        {
            set_event = 0,
            set_error = 1
        };

        /// Destructor, needs to be virtual to allow for clean destruction of
        /// derived objects
        virtual ~base_lco() {}

        ///
        virtual threadmanager::thread_state set_error_proc (
            threadmanager::px_thread_self&, applier::applier& appl,
            hpx::error code, std::string msg) = 0;
    };
    
    /// The base_lco class is the common base class for all LCO's. The 
    /// \a EventArgument template argument should be set to the type of the 
    /// argument expected for the set_event action
    template <typename EventArgument>
    struct base_lco_with_value : public base_lco
    {
        ///
        virtual threadmanager::thread_state set_event_proc (
            threadmanager::px_thread_self&, applier::applier& appl,
            EventArgument const& result) = 0;
    };
    
    /// The base_lco<void> specialization is used whenever the set_event action
    /// for a particular LCO doesn't carry an argument
    template<>
    struct base_lco_with_value<void> : public base_lco
    {
        ///
        virtual threadmanager::thread_state set_event_proc (
            threadmanager::px_thread_self&, applier::applier& appl) = 0;
    };

}}}

#endif
