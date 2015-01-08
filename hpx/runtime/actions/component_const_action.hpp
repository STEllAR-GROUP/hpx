//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file component_const_action.hpp

#if !defined(HPX_RUNTIME_ACTIONS_COMPONENT_CONST_ACTION_MAR_02_2013_0417PM)
#define HPX_RUNTIME_ACTIONS_COMPONENT_CONST_ACTION_MAR_02_2013_0417PM

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    //  Specialized generic component action types allowing to hold a different
    //  number of arguments
    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename R, typename ...Ps,
        typename TF, TF F, typename Derived>
    class basic_action_impl<
            R (Component::*)(Ps...) const, TF, F, Derived>
      : public basic_action<Component const, R(Ps...), Derived>
    {
    public:
        // Let the component decide whether the id is valid
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }

        template <typename ...Ts>
        static R invoke(naming::address::address_type lva, Ts&&... vs)
        {
            LTM_(debug) << "Executing action("
                        << detail::get_action_name<Derived>()
                        << ") lva(" << reinterpret_cast<void const*>
                            (get_lva<Component const>::call(lva)) << ")";

            return (get_lva<Component const>::call(lva)->*F)
                (std::forward<Ts>(vs)...);
        }
    };

    /// \endcond
}}

#include <hpx/config/warnings_suffix.hpp>

#endif

