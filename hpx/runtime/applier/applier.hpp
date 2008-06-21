//  Copyright (c) 2007-2008 Anshul Tandon
//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLIER_APPLIER_JUN_03_2008_0438PM)
#define HPX_APPLIER_APPLIER_JUN_03_2008_0438PM

#include <boost/noncopyable.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/naming.hpp>
#include <hpx/include/parcelset.hpp>
#include <hpx/runtime/threadmanager/threadmanager.hpp>
#include <hpx/components/action.hpp>
#include <hpx/components/server/runtime_support.hpp>

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
        {}

        // destructor
        ~applier()
        {}

        ///////////////////////////////////////////////////////////////////////
        // zero parameter version
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

        /// \note A call to applier's apply function would look like:
        /// \code
        ///    appl_.apply<add_action>(cont, gid, ...);
        /// \endcode
        template <typename Action>
        parcelset::parcel_id apply (components::continuation* c,
            naming::id_type gid)
        {
            // package continuation into a shared_ptr
            components::continuation_type cont(c);

            // Determine whether the gid is local or remote
            naming::address addr;
            if (address_is_local(gid, addr))
            {
                // If local, register the function with the thread-manager
                // Get the local-virtual address of the resource and register 
                // the action with the TM
                thread_manager_.register_work(
                    Action::construct_thread_function(cont, *this, addr.address_));
                return parcelset::no_parcel_id;     // no parcel has been sent
            }

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid, new Action(), cont);
            p.set_destination_addr(addr);   // avoid to resolve address again

            // Send the parcel through the parcel handler
            return parcel_handler_.put_parcel(p);
        }

        ///////////////////////////////////////////////////////////////////////
        // one parameter version
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

        template <typename Action, typename Arg0>
        parcelset::parcel_id apply (components::continuation* c, 
            naming::id_type gid, Arg0 const& arg0)
        {
            // package continuation into a shared_ptr
            components::continuation_type cont(c);

            // Determine whether the gid is local or remote
            naming::address addr;
            if (address_is_local(gid, addr)) 
            {
                // If local, register the function with the thread-manager
                // Get the local-virtual address of the resource and register 
                // the action with the TM
                thread_manager_.register_work(
                    Action::construct_thread_function(cont, *this, 
                    addr.address_, arg0));
                return parcelset::no_parcel_id;     // no parcel has been sent
            }

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid, new Action(arg0), cont);
            p.set_destination_addr(addr);   // avoid to resolve address again

            // Send the parcel through the parcel handler
            return parcel_handler_.put_parcel(p);
        }

        // bring in the rest of the apply<> overloads
        #include <hpx/runtime/applier/applier_implementations.hpp>

        /// The \a create_async function initiates the creation of a new 
        /// component using the runtime_support as given by targetgid. This function is 
        /// non-blocking as it returns a \a lcos#simple_future. The caller of 
        /// this create_async is responsible to call 
        /// \a lcos#simple_future#get_result to obtain the result. 
        ///
        /// \param self
        /// \param targetgid
        /// \param type
        /// \param count
        ///
        /// \returns    The function returns a \a lcos#simple_future instance 
        ///             returning the the global id of the newly created
        ///             component when used to call get_result.
        ///
        /// \note       For synchronous operation use the function 
        ///             \a applier#create_async.
        lcos::simple_future<naming::id_type> 
        create_async(naming::id_type const& targetgid, 
            components::component_type type, std::size_t count = 1);

        /// The \a create function creates a new component using the runtime_support as 
        /// given by targetgid. This function is blocking for the component to 
        /// be created and until the global id of the new component has been 
        /// returned. 
        ///
        /// \param self
        /// \param targetgid
        /// \param type
        /// \param count
        ///
        /// \returns    The function returns the global id of the newly created
        ///             component.
        ///
        /// \note       For asynchronous operation use the function 
        ///             \a applier#create_async.
        naming::id_type 
        create(threadmanager::px_thread_self& self, 
            naming::id_type const& targetgid, components::component_type type,
            std::size_t count = 1);

        /// \brief The \a free function frees an existing component as given by 
        ///        its type and its gid
        ///
        /// \param type
        /// \param count
        /// \param gid
        void free (components::component_type type, naming::id_type const& gid,
            std::size_t count = 1);

        /// \brief Allow access to the DGAS client instance used with this
        ///        \a applier.
        naming::resolver_client const& get_dgas_client() const
        {
            return dgas_client_;
        }

        /// \brief Access the \a parcelhandler instance associated with this 
        ///        \a applier
        parcelset::parcelhandler& get_parcel_handler() 
        {
            return parcel_handler_;
        }

        /// \brief Access the \a threadmanager instance associated with this 
        ///        \a applier
        threadmanager::threadmanager& get_thread_manager() 
        {
            return thread_manager_;
        }

        /// \brief Allow access to the locality this applier instance is 
        ///        associated with.
        ///
        /// This accessor returns a reference to the locality this applier
        /// instance is associated with.
        naming::locality const& here() const
        {
            return parcel_handler_.here();
        }

        /// \brief Allow access to the prefix of the locality this applier 
        ///        instance is associated with.
        ///
        /// This accessor returns a reference to the locality this applier
        /// instance is associated with.
        naming::id_type const& get_prefix() const
        {
            return parcel_handler_.get_prefix();
        }

    protected:
        bool address_is_local(naming::id_type gid, naming::address& addr) const
        {
            // Resolve the address of the gid
            if (!dgas_client_.resolve(gid, addr))
            {
                boost::throw_exception(
                    hpx::exception(hpx::unknown_component_address));
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
