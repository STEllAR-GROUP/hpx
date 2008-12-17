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
#include <hpx/runtime/threads/threadmanager.hpp>

#include <boost/thread/tss.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace applier
{
    /// \class applier applier.hpp hpx/runtime/applier/applier.hpp
    ///
    /// The \a applier class is used to decide whether a particular action
    /// has to be issued on a local or a remote resource. If the target 
    /// component is local a new \a thread will be created, if the target is
    /// remote a parcel will be sent.
    class HPX_EXPORT applier : private boost::noncopyable
    {
    public:
        // constructor
        applier(parcelset::parcelhandler &ph, threads::threadmanager& tm,
                boost::uint64_t rts, boost::uint64_t mem)
          : parcel_handler_(ph), thread_manager_(tm),
            runtime_support_id_(parcel_handler_.get_prefix().get_msb(), rts), 
            memory_id_(parcel_handler_.get_prefix().get_msb(), mem)
        {}

        // destructor
        ~applier()
        {}

        /// \brief Allow access to the AGAS client instance used with this
        ///        \a applier.
        ///
        /// This function returns a reference to the resolver client this 
        /// applier instance has been created with.
        naming::resolver_client const& get_agas_client() const
        {
            return parcel_handler_.get_resolver();
        }

        /// \brief Access the \a parcelhandler instance associated with this 
        ///        \a applier
        ///
        /// This function returns a reference to the parcel handler this 
        /// applier instance has been created with.
        parcelset::parcelhandler& get_parcel_handler() 
        {
            return parcel_handler_;
        }

        /// \brief Access the \a threadmanager instance associated with this 
        ///        \a applier
        ///
        /// This function returns a reference to the thread manager this 
        /// applier instance has been created with.
        threads::threadmanager& get_thread_manager() 
        {
            return thread_manager_;
        }

        /// \brief Allow access to the locality this applier instance is 
        ///        associated with.
        ///
        /// This function returns a reference to the locality this applier
        /// instance is associated with.
        naming::locality const& here() const
        {
            return parcel_handler_.here();
        }

        /// \brief Allow access to the prefix of the locality this applier 
        ///        instance is associated with.
        ///
        /// This function returns a reference to the locality this applier
        /// instance is associated with.
        naming::id_type const& get_prefix() const
        {
            return parcel_handler_.get_prefix();
        }

        /// \brief Allow access to the prefixes of all remote localities 
        ///        registered with the AGAS service.
        ///
        /// This function returns a list of all remote localities (all 
        /// localities known to AGAS except the local one).
        ///
        /// \param prefixes [out] The reference to a vector of id_types filled
        ///                 by the function.
        ///
        /// \returns The function returns \a true if there is at least one 
        ///          remote locality known to the AGASservice 
        ///          (!prefixes.empty()).
        bool get_remote_prefixes(std::vector<naming::id_type>& prefixes) const
        {
            return parcel_handler_.get_remote_prefixes(prefixes);
        }

        /// By convention the runtime_support has a gid identical to the prefix 
        /// of the locality the runtime_support is responsible for
        naming::id_type const& get_runtime_support_gid() const
        {
            return runtime_support_id_;
        }

        /// By convention every memory address has gid identical to the prefix 
        /// of the locality the runtime_support is responsible for
        naming::id_type const& get_memory_gid() const
        {
            return memory_id_;
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
                // a zero address references the local runtime support component
                if (0 != gid.get_lsb())
                    addr.address_ = gid.get_lsb();
                else 
                    addr.address_ = runtime_support_id_.get_lsb();
                return true;
            }

            // Resolve the address of the gid
            if (!parcel_handler_.get_resolver().resolve(gid, addr))
            {
                HPX_OSSTREAM strm;
                strm << gid;
                HPX_THROW_EXCEPTION(unknown_component_address, 
                    HPX_OSSTREAM_GETSTRING(strm));
            }
            return addr.locality_ == parcel_handler_.here();
        }

    public:
        // the TSS holds a pointer to the applier associated with a given 
        // OS thread
        static boost::thread_specific_ptr<applier*> applier_;
        void init_tss();
        void deinit_tss();

    private:
        parcelset::parcelhandler& parcel_handler_;
        threads::threadmanager& thread_manager_;
        naming::id_type runtime_support_id_;
        naming::id_type memory_id_;
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new \a thread using the given function as the 
    ///        work to be executed
    ///
    HPX_API_EXPORT threads::thread_id_type register_thread(
        boost::function<void()> func, char const* description = "", 
        threads::thread_state initial_state = threads::pending, 
        bool run_now = true);

    HPX_API_EXPORT threads::thread_id_type register_thread_plain(
        boost::function<threads::thread_function_type> func,
        char const* description = "", 
        threads::thread_state initial_state = threads::pending, 
        bool run_now = true);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new \a thread using the given function as the 
    ///        work to be executed
    ///
    HPX_API_EXPORT void register_work(
        boost::function<void()> func, char const* description = "", 
        threads::thread_state initial_state = threads::pending, 
        bool run_now = true);

    HPX_API_EXPORT void register_work_plain(
        boost::function<threads::thread_function_type> func,
        char const* description = "", 
        threads::thread_state initial_state = threads::pending, 
        bool run_now = true);

    /// The \a create_async function initiates the creation of a new 
    /// component using the runtime_support as given by targetgid. This 
    /// function is non-blocking as it returns a \a lcos#future_value. The 
    /// caller of this create_async is responsible to call 
    /// \a lcos#future_value#get to obtain the result. 
    ///
    /// \param targetgid
    /// \param type
    /// \param count
    ///
    /// \returns    The function returns a \a lcos#future_value instance 
    ///             returning the the global id of the newly created
    ///             component when used to call get.
    ///
    /// \note       For synchronous operation use the function 
    ///             \a applier#create_async.
    HPX_API_EXPORT lcos::future_value<naming::id_type> 
        create_async(naming::id_type const& targetgid, 
            components::component_type type, std::size_t count = 1);

    /// The \a create function creates a new component using the \a 
    /// runtime_support as given by targetgid. This function is blocking 
    /// for the component to be created and until the global id of the new 
    /// component has been returned. 
    ///
    /// \param targetgid
    /// \param type
    /// \param count
    ///
    /// \returns    The function returns the global id of the newly created
    ///             component.
    ///
    /// \note       For asynchronous operation use the function 
    ///             \a applier#create_async.
    HPX_API_EXPORT naming::id_type create(naming::id_type const& targetgid, 
        components::component_type type, std::size_t count = 1);

    /// \brief The \a destroy function frees an existing component as given by 
    ///        its type and its gid
    ///
    /// \param type
    /// \param count
    /// \param gid
    HPX_API_EXPORT void destroy (components::component_type type, 
        naming::id_type const& gid, std::size_t count = 1);

}}

#include <hpx/config/warnings_suffix.hpp>

#endif
