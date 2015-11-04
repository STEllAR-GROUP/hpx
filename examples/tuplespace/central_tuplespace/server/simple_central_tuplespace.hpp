//  Copyright (c) 2013 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SERVER_SIMPLE_CENTRAL_TUPLESPACE_MAR_29_2013_0237PM)
#define HPX_SERVER_SIMPLE_CENTRAL_TUPLESPACE_MAR_29_2013_0237PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/util/storage/tuple.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/include/local_lcos.hpp>

#include <boost/thread/locks.hpp>

#include "tuples_warehouse.hpp"

// #define TS_DEBUG

///////////////////////////////////////////////////////////////////////////////
namespace examples { namespace server
{

    ///////////////////////////////////////////////////////////////////////////
    /// This class is a simple central tuplespace (SCTS) as an HPX component. An HPX
    /// component is a class that:
    ///
    ///     * Inherits from a component base class:
    ///       \a hpx::components::component_base
    ///     * Exposes methods that can be called asynchronously and/or remotely.
    ///       These constructs are known as HPX actions.
    ///
    /// By deriving this component from \a locking_hook the runtime system
    /// ensures that all action invocations are serialized. That means that
    /// the system ensures that no two actions are invoked at the same time on
    /// a given component instance. This makes the component thread safe and no
    /// additional locking has to be implemented by the user.
    ///
    /// Components are first-class objects in HPX. This means that they are
    /// globally addressable; all components have a unique GID.
    ///
    /// The SCTS will store all tuples from any objects in a central locality,
    /// to demonstrate the basic function
    ///
    /// (from JavaSpace)
    /// write,
    /// read,
    /// take
    ///
    /// each has the last argument as a timeout value, pre-defined WAIT_FOREVER, NO_WAIT
    /// users can also provide its own timeout values.
    ///
    /// uses mutex, will hurt performance.
    ///
    //[simple_central_tuplespace_server_inherit
    class simple_central_tuplespace
      : public hpx::components::component_base<simple_central_tuplespace>
    //]
    {
        public:

            typedef hpx::util::storage::tuple tuple_type;
            typedef hpx::util::storage::tuple::elem_type elem_type;
            typedef hpx::lcos::local::spinlock mutex_type;

            typedef examples::server::tuples_warehouse tuples_type;

            // pre-defined timeout values
            enum {
                WAIT_FOREVER = -1, // <0 means blocking
                NO_WAIT = 0
            };

            //[simple_central_tuplespace_server_ctor
            simple_central_tuplespace() {}
            //]

            ///////////////////////////////////////////////////////////////////////
            // Exposed functionality of this component.

            //[simple_accumulator_methods

            // put tuple into tuplespace
            // out function
            int write(const tuple_type& tp)
            {
                if(tp.empty())
                {
                    return -1;
                }

                {
                    boost::lock_guard<mutex_type> l(mtx_);

                    tuples_.insert(tp);
                }

                return 0;
            }

            // read from tuplespace
            // rd function
            tuple_type read(const tuple_type& tp, const long timeout) const
            {
                tuple_type result;
                hpx::util::high_resolution_timer t;

                do
                {
                    if(tuples_.empty())
                    {
                        continue;
                    }

                    {
                        boost::lock_guard<mutex_type> l(mtx_);

                        result = tuples_.match(tp);
                    }


                    if(!result.empty())
                    {
                        break; // found
                    }
                } while((timeout < 0) || (timeout > t.elapsed()));

                return result;
            }

            // take from tuplespace
            // in function
            tuple_type take(const tuple_type& tp, const long timeout)
            {
                tuple_type result;
                hpx::util::high_resolution_timer t;

                do
                {

                    if(tuples_.empty())
                    {
                        continue;
                    }

                    {
                        boost::lock_guard<mutex_type> l(mtx_);

                        result = tuples_.match_and_erase(tp);
                    }


                    if(!result.empty())
                    {
                        break; // found
                    }
                } while((timeout < 0) || (timeout > t.elapsed()));

                return result;
            }

            //]



            ///////////////////////////////////////////////////////////////////////
            // Each of the exposed functions needs to be encapsulated into an
            // action type, generating all required boilerplate code for threads,
            // serialization, etc.

            //[simple_central_tuplespace_action_types
            HPX_DEFINE_COMPONENT_ACTION(simple_central_tuplespace, write);
            HPX_DEFINE_COMPONENT_ACTION(simple_central_tuplespace, read);
            HPX_DEFINE_COMPONENT_ACTION(simple_central_tuplespace, take);
            //]

            //[simple_central_tuplespace_server_data_member
        private:
            tuples_type tuples_;
            mutable mutex_type mtx_;
            //]
    };
}} // examples::server


//[simple_central_tuplespace_registration_declarations
HPX_REGISTER_ACTION_DECLARATION(
    examples::server::simple_central_tuplespace::write_action,
    simple_central_tuplespace_write_action);

HPX_REGISTER_ACTION_DECLARATION(
    examples::server::simple_central_tuplespace::read_action,
    simple_central_tuplespace_read_action);

HPX_REGISTER_ACTION_DECLARATION(
    examples::server::simple_central_tuplespace::take_action,
    simple_central_tuplespace_take_action);
//]

#undef TS_DEBUG

#endif

