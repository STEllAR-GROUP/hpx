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
#include <hpx/runtime/applier/apply_helper.hpp>
#include <hpx/components/action.hpp>
#include <hpx/components/server/runtime_support.hpp>

namespace hpx { namespace applier
{
    /// \class applier applier.hpp hpx/runtime/applier/applier.hpp
    ///
    /// The \a applier class is used to decide whether a particular action
    /// has to be issued on a local or a remote resource. If the target 
    /// component is local a new \a px_thread will be created, if the target is
    /// remote a parcel will be sent.
    class applier : private boost::noncopyable
    {
    public:
        // constructor
        applier(naming::resolver_client const& dgas_c, 
                parcelset::parcelhandler &ph, threadmanager::threadmanager& tm,
                components::server::runtime_support const& rts,
                components::server::memory const& mem)
          : dgas_client_(dgas_c), parcel_handler_(ph), thread_manager_(tm),
            runtime_support_(rts), memory_(mem)
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
        bool apply (naming::id_type const& gid)
        {
            // Determine whether the gid is local or remote
            naming::address addr;
            if (address_is_local(gid, addr)) {
                detail::apply_helper0<Action>::call(thread_manager_, 
                    *this, addr.address_);
                return true;     // no parcel has been sent (dest is local)
            }

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid, new Action());
            p.set_destination_addr(addr);   // avoid to resolve address again

            // Send the parcel through the parcel handler
            parcel_handler_.put_parcel(p);
            return false;     // destination is remote
        }

        /// \note A call to applier's apply function would look like:
        /// \code
        ///    appl_.apply<add_action>(cont, gid, ...);
        /// \endcode
        template <typename Action>
        bool apply (components::continuation* c, naming::id_type const& gid)
        {
            components::continuation_type cont(c);
            
            // Determine whether the gid is local or remote
            naming::address addr;
            if (address_is_local(gid, addr)) {
                detail::apply_helper0<Action>::call(cont, thread_manager_, 
                    *this, addr.address_);
                return true;     // no parcel has been sent (dest is local)
            }

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid, new Action(), cont);
            p.set_destination_addr(addr);   // avoid to resolve address again

            // Send the parcel through the parcel handler
            parcel_handler_.put_parcel(p);
            return false;     // destination is remote
        }

        template <typename Action>
        bool apply_c (naming::id_type const& targetgid, 
            naming::id_type const& gid)
        {
            return apply<Action>(new components::continuation(targetgid), gid);
        }

    public:
        ///////////////////////////////////////////////////////////////////////
        // one parameter version
        template <typename Action, typename Arg0>
        bool apply (naming::id_type const& gid, Arg0 const& arg0)
        {
            // Determine whether the gid is local or remote
            naming::address addr;
            if (address_is_local(gid, addr)) {
                detail::apply_helper1<Action, Arg0>::call(thread_manager_, 
                    *this, addr.address_, arg0);
                return true;     // no parcel has been sent (dest is local)
            }

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid, new Action(arg0));
            p.set_destination_addr(addr);   // avoid to resolve address again

            // Send the parcel through the parcel handler
            parcel_handler_.put_parcel(p);
            return false;     // destination is remote
        }

        template <typename Action, typename Arg0>
        bool apply (components::continuation* c, naming::id_type const& gid, 
            Arg0 const& arg0)
        {
            components::continuation_type cont(c);

            // Determine whether the gid is local or remote
            naming::address addr;
            if (address_is_local(gid, addr)) {
                detail::apply_helper1<Action, Arg0>::call(cont, thread_manager_,
                    *this, addr.address_, arg0);
                return true;     // no parcel has been sent (dest is local)
            }

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid, new Action(arg0), cont);
            p.set_destination_addr(addr);   // avoid to resolve address again

            // Send the parcel through the parcel handler
            parcel_handler_.put_parcel(p);
            return false;     // destination is remote
        }

        template <typename Action, typename Arg0>
        bool apply_c (naming::id_type const& targetgid, 
            naming::id_type const& gid, Arg0 const& arg0)
        {
            return apply<Action>(new components::continuation(targetgid), 
                gid, arg0);
        }

        // bring in the rest of the apply<> overloads
        #include <hpx/runtime/applier/applier_implementations.hpp>

        /// The \a create_async function initiates the creation of a new 
        /// component using the runtime_support as given by targetgid. This 
        /// function is non-blocking as it returns a \a lcos#simple_future. The 
        /// caller of this create_async is responsible to call 
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

        /// The \a create function creates a new component using the \a 
        /// runtime_support as given by targetgid. This function is blocking 
        /// for the component to be created and until the global id of the new 
        /// component has been returned. 
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
        ///
        /// This accessor returns a reference to the resolver client this 
        /// applier instance has been created with.
        naming::resolver_client const& get_dgas_client() const
        {
            return dgas_client_;
        }

        /// \brief Access the \a parcelhandler instance associated with this 
        ///        \a applier
        ///
        /// This accessor returns a reference to the parcel handler this 
        /// applier instance has been created with.
        parcelset::parcelhandler& get_parcel_handler() 
        {
            return parcel_handler_;
        }

        /// \brief Access the \a threadmanager instance associated with this 
        ///        \a applier
        ///
        /// This accessor returns a reference to the thread manager this 
        /// applier instance has been created with.
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

        /// By convention the runtime_support has a gid identical to the prefix 
        /// of the locality the runtime_support is responsible for
        naming::id_type get_runtime_support_gid() const
        {
            return naming::id_type(parcel_handler_.get_prefix().get_msb(), 
                boost::uint64_t(&runtime_support_));
        }

        /// By convention every memory address has gid identical to the prefix 
        /// of the locality the runtime_support is responsible for
        naming::id_type get_memory_gid() const
        {
            return naming::id_type(parcel_handler_.get_prefix().get_msb(), 
                boost::uint64_t(&memory_));
        }

        /// Test whether the given address (gid) is local or remote
        bool address_is_local(naming::id_type const& gid, 
            naming::address& addr) const
        {
            // test if the gid is of one of the non-movable objects
            // this is certainly an optimization relying on the fact that the 
            // lsb of the local objects is equal to their address
            if (gid.get_msb() == parcel_handler_.get_prefix().get_msb())
            {
                addr.address_ = gid.get_lsb();
                return true;
            }

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
        components::server::runtime_support const& runtime_support_;
        components::server::memory const& memory_;
    };

}}

#endif
