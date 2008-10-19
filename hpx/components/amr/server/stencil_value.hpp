//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STENCIL_VALUE_OCT_17_2008_0848AM)
#define HPX_COMPONENTS_AMR_STENCIL_VALUE_OCT_17_2008_0848AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/action.hpp>
#include <hpx/lcos/counting_semaphore.hpp>

#include <hpx/components/amr/server/stencil_value_in_adaptor.hpp>
#include <hpx/components/amr/server/stencil_value_out_adaptor.hpp>

#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server { namespace detail
{

    /// \class stencil_value stencil_value.hpp hpx/components/amr/server/stencil_value.hpp
    template <typename Derived, typename T, int N>
    class stencil_value : boost::noncopyable
    {
    private:
        static component_type value;

    protected:
        typedef T result_type;

        // the in_adaptors_type is a fusion sequence of stencil_value_in_adaptor's
        // of the proper types, one in_adaptor for each parameter of the call
        // function
        typedef server::stencil_value_in_adaptor<T> in_adaptor_type;

        /// 
        typedef server::stencil_value_out_adaptor<T> out_adaptor_type;

        Derived& derived() { return *static_cast<Derived*>(this); }
        Derived const& derived() const { return *static_cast<Derived const*>(this); }

    public:
        // components must contain a typedef for wrapping_type defining the
        // managed_component_base type used to encapsulate instances of this 
        // component
        typedef managed_component_base<stencil_value> wrapping_type;

        /// Construct a new stencil_value instance
        stencil_value(applier::applier& appl)
          : sem_in_(N), sem_out_(0)
        {
            // create adaptors
            for (std::size_t i = 0; i < N; ++i)
            {
                in_[i].reset(new in_adaptor_type(appl));
                out_[i].reset(new out_adaptor_type(appl));
                out_[i]->set_callback(
                    boost::bind(&stencil_value::get_value, this, _1, _2));
            }
        }

        /// The function get_result will be called by the out-ports whenever 
        /// the current value has been requested.
        void get_value(threads::thread_self& self, result_type*);

        ///////////////////////////////////////////////////////////////////////
        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        static component_type get_component_type()
        {
            return value;
        }
        static void set_component_type(component_type type)
        {
            value = type;
        }

        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            stencil_value_call = 0,
            stencil_value_get_output_ports = 1,
            stencil_value_connect_input_ports = 2,
        };

        /// This is the main entry point of this component. Calling this 
        /// function (by applying the call_action) will trigger the repeated 
        /// execution of the whole time step evolution functionality.
        threads::thread_state 
        call (threads::thread_self&, applier::applier&, result_type*, 
            T const& initial);

        /// Return the gid's of the output ports associated with this 
        /// \a stencil_value instance.
        threads::thread_state 
        get_output_ports(threads::thread_self&, applier::applier&, 
            std::vector<naming::id_type> *gids);

        /// Connect the destinations given by the provided gid's with the 
        /// corresponding input ports associated with this \a stencil_value 
        /// instance.
        threads::thread_state 
        connect_input_ports(threads::thread_self&, applier::applier&, 
            std::vector<naming::id_type> const& gids);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action1<
            stencil_value, result_type, stencil_value_call, T const&, 
            &stencil_value::call
        > call_action;

        typedef hpx::actions::result_action0<
            stencil_value, std::vector<naming::id_type>, 
            stencil_value_get_output_ports, &stencil_value::get_output_ports
        > get_output_ports_action;

        typedef hpx::actions::action1<
            stencil_value, stencil_value_connect_input_ports, 
            std::vector<naming::id_type> const&,
            &stencil_value::connect_input_ports
        > connect_input_ports_action;

    private:
        lcos::counting_semaphore sem_in_;
        lcos::counting_semaphore sem_out_;

        boost::scoped_ptr<in_adaptor_type> in_[N];     // adaptors used to gather input
        boost::scoped_ptr<out_adaptor_type> out_[N];   // adaptors used to provide result

        result_type value_;     // current value
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename T, int N>
    component_type stencil_value<Derived, T, N>::value = component_invalid;

}}}}

#endif

