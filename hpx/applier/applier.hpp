//  Copyright (c) 2007-2008 Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLIER_APPLIER_JUN_03_2008_0438PM)
#define HPX_APPLIER_APPLIER_JUN_03_2008_0438PM

#include <boost/noncopyable.hpp>

#include <hpx/naming.hpp>
#include <hpx/parcelset.hpp>
#include <hpx/threadmanager.hpp>

namespace hpx { namespace applier
{
    /// The \a applier class is used to decide whether a particular action
    /// has to be issued on a local or a remote resource. If the target 
    /// component is local a new \a px_thread will be created, if the target is
    /// remote a parcel will be sent.
    class applier : private boost::noncopyable
    {
    public:
        // constructor
        applier(naming::resolver_client const& dgas_c, 
                parcelset::parcelhandler &ph, threadmanager::threadmanager& tm)
          : dgas_client_(dgas_c), parcel_handler_(ph), thread_manager_(tm)
        {
        }

        // destructor
        ~applier()
        {
        }

        // Invoked by a running PX-thread to apply an action to any resource
        /// \note A call to applier's apply function would look like:
        /// \code
        ///    appl_.apply<add_action>(gid, ...);
        /// \endcode
        template <typename Action>
        parcelset::parcel_id apply (naming::id_type gid)
        {
            // Determine whether the gid is local or remote
            naming::address addr;
            if (address_is_local(gid, addr))
            {
                // If local, register the function with the thread-manager
                // Get the local-virtual address of the resource and register 
                // the action with the TM
                thread_manager_.register_work(
                    Action::construct_thread_function(*this, addr.address_));
                return parcelset::no_parcel_id;     // no parcel has been sent
            }

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid, new Action());
            p.set_destination_addr(addr);   // avoid to resolve address again
            
            // Send the parcel through the parcel handler
            return parcel_handler_.put_parcel(p);
        }

        template <typename Action, typename Arg0>
        parcelset::parcel_id apply (naming::id_type gid, Arg0 const& arg0)
        {
            // Determine whether the gid is local or remote
            naming::address addr;
            if (address_is_local(gid, addr)) 
            {
                // If local, register the function with the thread-manager
                // Get the local-virtual address of the resource and register 
                // the action with the TM
                thread_manager_.register_work(
                    Action::construct_thread_function(*this, addr.address_, arg0));
                return parcelset::no_parcel_id;     // no parcel has been sent
            }

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid, new Action(arg0));
            p.set_destination_addr(addr);   // avoid to resolve address again

            // Send the parcel through the parcel handler
            return parcel_handler_.put_parcel(p);
        }

        template <typename Action, typename Arg0, typename Arg1>
        parcelset::parcel_id apply (naming::id_type gid, Arg0 const& arg0, Arg1 const& arg1)
        {
            // Determine whether the gid is local or remote
            naming::address addr;
            if (address_is_local(gid, addr))
            {
                // If local, register the function with the thread-manager
                // Get the local-virtual address of the resource and register 
                // the action with the TM
                thread_manager_.register_work(
                    Action::construct_thread_function(*this, addr.address_, arg0, arg1));
                return parcelset::no_parcel_id;     // no parcel has been sent
            }

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid, new Action(arg0, arg1));
            p.set_destination_addr(addr);   // avoid to resolve address again

            // Send the parcel through the parcel handler
            return parcel_handler_.put_parcel(p);
        }
        
        // bring in the rest of the apply<> overloads
        #include <hpx/applier/applier_implementations.hpp>
        
        /// \brief Access the threadmanager instance associated with this applier
        threadmanager::threadmanager& get_thread_manager() 
        {
            return thread_manager_;
        }
        
    protected:
        bool address_is_local(naming::id_type gid, naming::address& addr) const
        {
            // Resolve the address of the gid
            if (!dgas_client_.resolve(gid, addr))
            {
                boost::throw_exception(
                    hpx::exception(hpx::unknown_component_address));
                return parcelset::bad_parcel_id;
            }
            return addr.locality_ == parcel_handler_.here();
        }
        
    private:
        naming::resolver_client const& dgas_client_;
        parcelset::parcelhandler& parcel_handler_;
        threadmanager::threadmanager& thread_manager_;
    };
}}

#endif
