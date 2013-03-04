//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file component_action.hpp

#if !defined(HPX_RUNTIME_ACTIONS_COMPONENT_ACTION_MAR_26_2008_1054AM)
#define HPX_RUNTIME_ACTIONS_COMPONENT_ACTION_MAR_26_2008_1054AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/config/bind.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/components/console_error_sink.hpp>
#include <hpx/util/unused.hpp>
#include <hpx/util/void_cast.hpp>
#include <hpx/util/detail/count_num_args.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

#include <boost/preprocessor/cat.hpp>

#include <cstdlib>
#include <stdexcept>

namespace hpx { namespace actions
{
    // declarations for main templates

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, F funcptr, typename Derived>
    class base_result_action0;

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, F funcptr, typename Derived = detail::this_type>
    struct result_action0;

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, F funcptr, typename Derived = detail::this_type>
    struct direct_result_action0;

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, F funcptr, typename Derived>
    class base_action0;

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, F funcptr, typename Derived = detail::this_type>
    struct action0;

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, F funcptr, typename Derived = detail::this_type>
    struct direct_action0;
}}

///////////////////////////////////////////////////////////////////////////////
// bring in nullary actions and all other arities
#include <hpx/runtime/actions/component_const_action.hpp>
#include <hpx/runtime/actions/component_non_const_action.hpp>
#include <hpx/runtime/actions/component_action_implementations.hpp>

#endif

