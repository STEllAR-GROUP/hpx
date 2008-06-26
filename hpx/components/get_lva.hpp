//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_GET_LVA_JUN_22_2008_0451PM)
#define HPX_COMPONENTS_GET_LVA_JUN_22_2008_0451PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/address.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    /// \class get_lva get_lva.hpp hpx/components/get_lva.hpp
    ///
    /// The \a get_lva template is a helper structure allowing to convert a 
    /// local virtual address as stored in a local address (returned from 
    /// the function \a resolver_client#resolve) to the address of the 
    /// component implementing the action.
    ///
    /// The default implementation uses the template argument \a Component
    /// to deduce the type wrapping the component implementing the action. This
    /// is used to get the needed address.
    ///
    /// The specialization for the \a runtime_support component is needed 
    /// because this is not wrapped by a separate type as all the other 
    /// components.
    ///
    /// \tparam Component  This is the type of the component implementing the 
    ///                    action to execute.
    template <typename Component>
    struct get_lva
    {
        static Component* 
        call(naming::address::address_type lva)
        {
            typedef typename Component::wrapping_type wrapping_type;
            return reinterpret_cast<wrapping_type*>(lva)->get();
        }
    };

    // specialization for server::runtime_support
    template <>
    struct get_lva<server::runtime_support>
    {
        // for server::runtime_support the provided lva is directly usable
        // as the required local address
        static server::runtime_support* 
        call(naming::address::address_type lva)
        {
            return reinterpret_cast<server::runtime_support*>(lva);
        }
    };

    // specialization for server::memory
    template <>
    struct get_lva<server::memory>
    {
        // for server::memory the provided lva is directly usable as the 
        // required local address
        static server::memory* 
        call(naming::address::address_type lva)
        {
            return reinterpret_cast<server::memory*>(lva);
        }
    };

}}

#endif

