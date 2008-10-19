//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_FUNCTIONAL_COMPONENT_OCT_19_2008_1234PM)
#define HPX_COMPONENTS_AMR_FUNCTIONAL_COMPONENT_OCT_19_2008_1234PM

#include <hpx/components/amr/server/functional_component_base.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    template <typename Derived, typename T, int N>
    struct functional_component;           // forward declaration only

}}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server { namespace detail
{
    template <typename Derived, typename T, int N>
    class functional_component;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename T>
    class functional_component<Derived, T, 1>
      : public functional_component_base<T, 1>
    {
    protected:
        Derived& derived() 
            { return *static_cast<Derived*>(this); }
        Derived const& derived() const 
            { return *static_cast<derived const*>(this); }

    public:
        // components must contain a typedef for wrapping_type defining the
        // managed_component_base type used to encapsulate instances of this 
        // component
        typedef amr::server::functional_component<Derived, T, 1> wrapping_type;

        /// This is the main entry point of this component. Calling this 
        /// function (by applying the eval_action) will compute the next 
        /// time step value based on the result values of the previous time 
        /// steps.
        threads::thread_state 
        eval(threads::thread_self&, applier::applier&, T*, T const&);

        threads::thread_state 
        is_last_timestep(threads::thread_self&, applier::applier&, bool*);
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename T>
    class functional_component<Derived, T, 3>
      : public functional_component_base<T, 3>
    {
    protected:
        Derived& derived() 
            { return *static_cast<Derived*>(this); }
        Derived const& derived() const 
            { return *static_cast<derived const*>(this); }

    public:
        // components must contain a typedef for wrapping_type defining the
        // managed_component_base type used to encapsulate instances of this 
        // component
        typedef amr::server::functional_component<Derived, T, 3> wrapping_type;

        /// This is the main entry point of this component. Calling this 
        /// function (by applying the eval_action) will compute the next 
        /// time step value based on the result values of the previous time 
        /// steps.
        threads::thread_state 
        eval(threads::thread_self&, applier::applier&, T*, 
            T const&, T const&, T const&);

        threads::thread_state 
        is_last_timestep(threads::thread_self&, applier::applier&, bool*);
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename T>
    class functional_component<Derived, T, 5>
      : public functional_component_base<T, 5>
    {
    protected:
        Derived& derived() 
            { return *static_cast<Derived*>(this); }
        Derived const& derived() const 
            { return *static_cast<derived const*>(this); }

    public:
        // components must contain a typedef for wrapping_type defining the
        // managed_component_base type used to encapsulate instances of this 
        // component
        typedef amr::server::functional_component<Derived, T, 5> wrapping_type;

        /// This is the main entry point of this component. Calling this 
        /// function (by applying the eval_action) will compute the next 
        /// time step value based on the result values of the previous time 
        /// steps.
        threads::thread_state 
        eval(threads::thread_self&, applier::applier&, T*, 
            T const&, T const&, T const&, T const&, T const&);

        threads::thread_state 
        is_last_timestep(threads::thread_self&, applier::applier&, bool*);
    };

}}}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server
{
    /// This class defines the overall component type. It is needed to build
    /// the proper infrastructure for the HPX component architecture
    template <typename Derived, typename T, int N>
    struct functional_component 
      : public managed_component_base<
            detail::functional_component_base<T, N>, 
            functional_component<Derived, T, N> 
        >
    {
        typedef detail::functional_component_base<T, N> wrapped_type;
        typedef managed_component_base<
            wrapped_type, functional_component
        > base_type;

        functional_component(applier::applier& appl) 
          : base_type(new detail::functional_component<Derived, T, N>()) 
        {}
    };

}}}}

#endif
