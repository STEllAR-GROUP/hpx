//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_EAGER_FUTURE_JUN_27_2008_0420PM)
#define HPX_LCOS_EAGER_FUTURE_JUN_27_2008_0420PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/util/full_empty_memory.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/lcos/future_value.hpp>

#include <boost/variant.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos 
{
    ///////////////////////////////////////////////////////////////////////////
    /// \class eager_future eager_future.hpp hpx/lcos/eager_future.hpp
    ///
    /// A eager_future can be used by a single \a thread to invoke a 
    /// (remote) action and wait for the result. The result is expected to be 
    /// sent back to the eager_future using the LCO's set_event action
    ///
    /// A eager_future is one of the simplest synchronization primitives 
    /// provided by HPX. It allows to synchronize on a eager evaluated remote
    /// operation returning a result of the type \a Result. 
    ///
    /// \tparam Action   The template parameter \a Action defines the action 
    ///                  to be executed by this eager_future instance. The 
    ///                  arguments \a arg0,... \a argN are used as parameters 
    ///                  for this action.
    /// \tparam Result   The template parameter \a Result defines the type this 
    ///                  eager_future is expected to return from 
    ///                  \a eager_future#get.
    /// \tparam DirectExecute The template parameter \a DirectExecute is an
    ///                  optimization aid allowing to execute the action 
    ///                  directly if the target is local (without spawning a 
    ///                  new thread for this). This template does not have to be
    ///                  supplied explicitly as it is derived from the template 
    ///                  parameter \a Action.
    ///
    /// \note            The action executed using the eager_future as a 
    ///                  continuation must return a value of a type convertible 
    ///                  to the type as specified by the template parameter 
    ///                  \a Result.
    template <typename Action, typename Result, typename DirectExecute>
    class eager_future;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Result>
    class eager_future<Action, Result, boost::mpl::false_> 
        : public future_value<Result>
    {
    private:
        typedef future_value<Result> base_type;

    public:
        /// Construct a (non-functional) instance of an \a eager_future. To use
        /// this instance 
        eager_future()
        {}

        /// The apply function starts the asynchronous operations encapsulated
        /// by this eager future.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        void apply(naming::id_type const& gid)
        {
            hpx::applier::apply_c<Action>(this->get_gid(), gid);
        }

        /// Construct a new \a eager_future instance. The \a thread 
        /// supplied to the function \a eager_future#get will be 
        /// notified as soon as the result of the operation associated with 
        /// this eager_future instance has been returned.
        /// 
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        ///
        /// \note         The result of the requested operation is expected to 
        ///               be returned as the first parameter using a 
        ///               \a base_lco#set_result action. Any error has to be 
        ///               reported using a \a base_lco::set_error action. The 
        ///               target for either of these actions has to be this 
        ///               eager_future instance (as it has to be sent along 
        ///               with the action as the continuation parameter).
        eager_future(naming::id_type const& gid)
        {
            apply(gid);
        }

        /// The apply function starts the asynchronous operations encapsulated
        /// by this eager future.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        /// \param arg0   [in] The parameter \a arg0 will be passed on to the 
        ///               apply operation for the embedded action.
        template <typename Arg0>
        void apply(naming::id_type const& gid, Arg0 const arg0)
        {
            hpx::applier::apply_c<Action>(this->get_gid(), gid, arg0);
        }

        /// Construct a new \a eager_future instance. The \a thread 
        /// supplied to the function \a eager_future#get will be 
        /// notified as soon as the result of the operation associated with 
        /// this eager_future instance has been returned.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        /// \param arg0   [in] The parameter \a arg0 will be passed on to the 
        ///               apply operation for the embedded action.
        ///
        /// \note         The result of the requested operation is expected to 
        ///               be returned as the first parameter using a 
        ///               \a base_lco#set_result action. Any error has to be 
        ///               reported using a \a base_lco::set_error action. The 
        ///               target for either of these actions has to be this 
        ///               eager_future instance (as it has to be sent along 
        ///               with the action as the continuation parameter).
        template <typename Arg0>
        eager_future(naming::id_type const& gid, Arg0 const& arg0)
        {
            apply(gid, arg0);
        }

        // pull in remaining constructors
        #include <hpx/lcos/eager_future_constructors.hpp>
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Result>
    class eager_future<Action, Result, boost::mpl::true_> 
        : public future_value<Result>
    {
    private:
        typedef future_value<Result> base_type;

    public:
        /// The apply function starts the asynchronous operations encapsulated
        /// by this eager future.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        void apply(naming::id_type const& gid)
        {
            // Determine whether the gid is local or remote
            naming::address addr;
            if (hpx::applier::get_applier().address_is_local(gid, addr)) {
                // local, direct execution
                BOOST_ASSERT(components::types_are_compatible(addr.type_, 
                    components::get_component_type<typename Action::component_type>()));
                (*this->impl_)->set_data(0, 
                    Action::execute_function(addr.address_));
            }
            else {
                // remote execution
                hpx::applier::apply_c<Action>(addr, this->get_gid(), gid);
            }
        }

        /// Construct a new \a eager_future instance. The \a thread 
        /// supplied to the function \a eager_future#get will be 
        /// notified as soon as the result of the operation associated with 
        /// this eager_future instance has been returned.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        /// 
        /// \note         The result of the requested operation is expected to 
        ///               be returned as the first parameter using a 
        ///               \a base_lco#set_result action. Any error has to be 
        ///               reported using a \a base_lco::set_error action. The 
        ///               target for either of these actions has to be this 
        ///               eager_future instance (as it has to be sent along 
        ///               with the action as the continuation parameter).
        eager_future(naming::id_type const& gid)
        {
            apply(gid);
        }

        /// The apply function starts the asynchronous operations encapsulated
        /// by this eager future.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        /// \param arg0   [in] The parameter \a arg0 will be passed on to the 
        ///               apply operation for the embedded action.
        template <typename Arg0>
        void apply(naming::id_type const& gid, Arg0 const& arg0)
        {
            // Determine whether the gid is local or remote
            naming::address addr;
            if (hpx::applier::get_applier().address_is_local(gid, addr)) {
                // local, direct execution
                BOOST_ASSERT(components::types_are_compatible(addr.type_, 
                    components::get_component_type<typename Action::component_type>()));
                (*this->impl_)->set_data(
                    0, Action::execute_function(addr.address_, arg0));
            }
            else {
                // remote execution
                hpx::applier::apply_c<Action>(addr, this->get_gid(), gid, arg0);
            }
        }

        /// Construct a new \a eager_future instance. The \a thread 
        /// supplied to the function \a eager_future#get will be 
        /// notified as soon as the result of the operation associated with 
        /// this eager_future instance has been returned.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        /// \param arg0   [in] The parameter \a arg0 will be passed on to the 
        ///               apply operation for the embedded action.
        /// 
        /// \note         The result of the requested operation is expected to 
        ///               be returned as the first parameter using a 
        ///               \a base_lco#set_result action. Any error has to be 
        ///               reported using a \a base_lco::set_error action. The 
        ///               target for either of these actions has to be this 
        ///               eager_future instance (as it has to be sent along 
        ///               with the action as the continuation parameter).
        template <typename Arg0>
        eager_future(naming::id_type const& gid, Arg0 const& arg0)
        {
            apply(gid, arg0);
        }

        // pull in remaining constructors
        #include <hpx/lcos/eager_future_constructors_direct.hpp>
    };

}}

#endif
