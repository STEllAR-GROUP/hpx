//  Copyright (c) 2007-2008 Anshul Tandon
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLIER_APPLIER_JUN_03_2008_0438PM)
#define HPX_APPLIER_APPLIER_JUN_03_2008_0438PM

#include <boost/noncopyable.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/util/thread_specific_ptr.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>

#include <boost/foreach.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace applier
{
    /// The \a applier class is used to decide whether a particular action
    /// has to be issued on a local or a remote resource. If the target
    /// component is local a new \a thread will be created, if the target is
    /// remote a parcel will be sent.
    class HPX_EXPORT applier : private boost::noncopyable
    {
    public:
        // constructor
        applier(parcelset::parcelhandler &ph, threads::threadmanager_base& tm,
                boost::uint64_t rts, boost::uint64_t mem);

        // destructor
        ~applier()
        {}

        /// \brief Allow access to the AGAS client instance used with this
        ///        \a applier.
        ///
        /// This function returns a reference to the resolver client this
        /// applier instance has been created with.
        naming::resolver_client& get_agas_client();

        /// \brief Access the \a parcelhandler instance associated with this
        ///        \a applier
        ///
        /// This function returns a reference to the parcel handler this
        /// applier instance has been created with.
        parcelset::parcelhandler& get_parcel_handler();

        /// \brief Access the \a thread-manager instance associated with this
        ///        \a applier
        ///
        /// This function returns a reference to the thread manager this
        /// applier instance has been created with.
        threads::threadmanager_base& get_thread_manager();

        /// \brief Allow access to the locality this applier instance is
        ///        associated with.
        ///
        /// This function returns a reference to the locality this applier
        /// instance is associated with.
        naming::locality const& here() const;

        /// \brief Allow access to the locality of the locality this applier
        ///        instance is associated with.
        ///
        /// This function returns a reference to the locality this applier
        /// instance is associated with.
        naming::gid_type const& get_raw_locality() const;

        /// \brief Allow access to the id of the locality this applier
        ///        instance is associated with.
        ///
        /// This function returns a reference to the id of the locality this
        /// applier instance is associated with.
        boost::uint32_t get_locality_id() const;

        /// \brief Return list of localities of all remote localities
        ///        registered with the AGAS service for a specific component
        ///        type.
        ///
        /// This function returns a list of all remote localities (all
        /// localities known to AGAS except the local one) supporting the given
        /// component type.
        ///
        /// \param locality_ids [out] The reference to a vector of id_types filled
        ///                 by the function.
        /// \param type     [in] The type of the component which needs to exist
        ///                 on the returned localities.
        ///
        /// \returns The function returns \a true if there is at least one
        ///          remote locality known to the AGASservice
        ///          (!prefixes.empty()).
        bool get_raw_remote_localities(std::vector<naming::gid_type>& locality_ids,
            components::component_type type = components::component_invalid) const;

        bool get_remote_localities(std::vector<naming::id_type>& locality_ids,
            components::component_type type = components::component_invalid) const;

        /// \brief Return list of locality_ids of all localities
        ///        registered with the AGAS service for a specific component
        ///        type.
        ///
        /// This function returns a list of all localities (all
        /// localities known to AGAS except the local one) supporting the given
        /// component type.
        ///
        /// \param locality_ids [out] The reference to a vector of id_types filled
        ///                 by the function.
        /// \param type     [in] The type of the component which needs to exist
        ///                 on the returned localities.
        ///
        /// \returns The function returns \a true if there is at least one
        ///          remote locality known to the AGASservice
        ///          (!prefixes.empty()).
        bool get_raw_localities(std::vector<naming::gid_type>& locality_ids,
            components::component_type type = components::component_invalid) const;

        bool get_localities(std::vector<naming::id_type>& locality_ids,
            components::component_type type = components::component_invalid) const;

        /// By convention the runtime_support has a gid identical to the prefix
        /// of the locality the runtime_support is responsible for
        naming::gid_type const& get_runtime_support_raw_gid() const
        {
            return runtime_support_id_.get_gid();
        }

        /// By convention the runtime_support has a gid identical to the prefix
        /// of the locality the runtime_support is responsible for
        naming::id_type const& get_runtime_support_gid() const
        {
            return runtime_support_id_;
        }

        /// By convention every memory address has gid identical to the prefix
        /// of the locality the runtime_support is responsible for
        naming::gid_type const& get_memory_raw_gid() const
        {
            return memory_id_.get_gid();
        }

        /// By convention every memory address has gid identical to the prefix
        /// of the locality the runtime_support is responsible for
        naming::id_type const& get_memory_gid() const
        {
            return memory_id_;
        }

        // parcel forwarding
        bool route(parcelset::parcel const& arg0);

    public:
        // the TSS holds a pointer to the applier associated with a given
        // OS thread
        struct tls_tag {};
        static hpx::util::thread_specific_ptr<applier*, tls_tag> applier_;
        void init_tss();
        void deinit_tss();

    private:
        parcelset::parcelhandler& parcel_handler_;
        threads::threadmanager_base& thread_manager_;
        naming::id_type runtime_support_id_;
        naming::id_type memory_id_;
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new \a thread using the given function as the work to
    ///        be executed.
    ///
    /// \param func       [in] The function to be executed as the thread-function.
    ///                   This function has to expose the minimal low level
    ///                   PX-thread interface, i.e. it takes one argument (a
    ///                   \a threads#thread_state_ex_enum) and returns a
    ///                   \a threads#thread_state_enum.
    /// \param description [in] A optional string describing the newly created
    ///                   thread. This is useful for debugging and logging
    ///                   purposes as this string will be inserted in the logs.
    /// \param initial_state [in] The thread state the newly created thread
    ///                   should have. If this is not given it defaults to
    ///                   \a threads#pending, which means that the new thread
    ///                   will be scheduled to run as soon as it is created.
    /// \param run_now    [in] If this is set to `true` the thread object will
    ///                   be actually immediately created. Otherwise the
    ///                   thread-manager creates a work-item description, which
    ///                   will result in creating a thread object later (if
    ///                   no work is available any more). The default is to
    ///                   immediately create the thread object.
    /// \param priority   [in] This is the priority the newly created PX-thread
    ///                   should be executed with. The default is \a
    ///                   threads#thread_priority_normal. This parameter is not
    ///                   guaranteed to be taken into account as it depends on
    ///                   the used scheduling policy whether priorities are
    ///                   supported in the first place.
    /// \param os_thread  [in] The number of the shepherd thread the newly
    ///                   created PX-thread should run on. If this is given it
    ///                   will be no more than a hint in any case, mainly
    ///                   because even if the PX-thread gets scheduled on the
    ///                   queue of the requested shepherd thread, it still can
    ///                   be stolen by another shepherd thread. If this is not
    ///                   given, the system will select a shepherd thread.
    /// \param ec         [in,out] This represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \returns This function will return the internal id of the newly created
    ///          PX-thread or threads#invalid_thread_id (if run_now is set to
    ///          `false`).
    ///
    /// \note The value returned by the thread function will be interpreted by
    ///       the thread manager as the new thread state the executed PX-thread
    ///       needs to be switched to. Normally, PX-threads will either return
    ///       \a threads#terminated (if the thread should be destroyed) or
    ///       \a threads#suspended (if the thread needs to be suspended because
    ///       it is waiting for an external event to happen). The external
    ///       event will set the state of the thread back to pending, which
    ///       will re-schedule the PX-thread.
    ///
    /// \throws invalid_status if the runtime system has not been started yet.
    ///
    HPX_API_EXPORT threads::thread_id_type register_thread_plain(
        BOOST_RV_REF(HPX_STD_FUNCTION<threads::thread_function_type>) func,
        char const* description = 0,
        threads::thread_state_enum initial_state = threads::pending,
        bool run_now = true,
        threads::thread_priority priority = threads::thread_priority_normal,
        std::size_t os_thread = std::size_t(-1), error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new \a thread using the given function as the work to
    ///        be executed.
    ///
    /// \param func       [in] The function to be executed as the thread-function.
    ///                   This function has to expose the minimal low level
    ///                   PX-thread interface, i.e. it takes one argument (a
    ///                   \a threads#thread_state_ex_enum). The thread will be
    ///                   terminated after the function returns.
    ///
    /// \note All other arguments are equivalent to those of the function
    ///       \a applier#register_thread_plain
    ///
    HPX_API_EXPORT threads::thread_id_type register_thread(
        BOOST_RV_REF(HPX_STD_FUNCTION<void(threads::thread_state_ex_enum)>) func,
        char const* description = 0,
        threads::thread_state_enum initial_state = threads::pending,
        bool run_now = true,
        threads::thread_priority priority = threads::thread_priority_normal,
        std::size_t os_thread = std::size_t(-1), error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new \a thread using the given function as the work to
    ///        be executed.
    ///
    /// \param func       [in] The function to be executed as the thread-function.
    ///                   This function has to expose the minimal low level
    ///                   PX-thread interface, i.e. it takes no arguments. The
    ///                   thread will be terminated after the function returns.
    ///
    /// \note All other arguments are equivalent to those of the function
    ///       \a applier#register_thread_plain
    ///
    HPX_API_EXPORT threads::thread_id_type register_thread_nullary(
        BOOST_RV_REF(HPX_STD_FUNCTION<void()>) func, char const* description = 0,
        threads::thread_state_enum initial_state = threads::pending,
        bool run_now = true,
        threads::thread_priority priority = threads::thread_priority_normal,
        std::size_t os_thread = std::size_t(-1), error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new \a thread using the given data.
    ///
    /// \note This function is completely equivalent to the first overload
    ///       of applier#register_thread_plain above, except that part of the
    ///       parameters are passed as members of the threads#thread_init_data
    ///       object.
    ///
    HPX_API_EXPORT threads::thread_id_type register_thread_plain(
        threads::thread_init_data& data,
        threads::thread_state_enum initial_state = threads::pending,
        bool run_now = true, error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new work item using the given function as the
    ///        work to be executed. This work item will be used to create a
    ///        \a threads#thread instance whenever the shepherd thread runs out
    ///        of work only. The created work descriptions will be queued
    ///        separately, causing them to be converted into actual thread
    ///        objects on a first-come-first-served basis.
    ///
    /// \param func       [in] The function to be executed as the thread-function.
    ///                   This function has to expose the minimal low level
    ///                   PX-thread interface, i.e. it takes one argument (a
    ///                   \a threads#thread_state_ex_enum) and returns a
    ///                   \a threads#thread_state_enum.
    /// \param description [in] A optional string describing the newly created
    ///                   thread. This is useful for debugging and logging
    ///                   purposes as this string will be inserted in the logs.
    /// \param initial_state [in] The thread state the newly created thread
    ///                   should have. If this is not given it defaults to
    ///                   \a threads#pending, which means that the new thread
    ///                   will be scheduled to run as soon as it is created.
    /// \param priority   [in] This is the priority the newly created PX-thread
    ///                   should be executed with. The default is \a
    ///                   threads#thread_priority_normal. This parameter is not
    ///                   guaranteed to be taken into account as it depends on
    ///                   the used scheduling policy whether priorities are
    ///                   supported in the first place.
    /// \param os_thread  [in] The number of the shepherd thread the newly
    ///                   created PX-thread should run on. If this is given it
    ///                   will be no more than a hint in any case, mainly
    ///                   because even if the PX-thread gets scheduled on the
    ///                   queue of the requested shepherd thread, it still can
    ///                   be stolen by another shepherd thread. If this is not
    ///                   given, the system will select a shepherd thread.
    /// \param ec         [in,out] This represents the error status on exit,
    ///                   if this is pre-initialized to \a hpx#throws
    ///                   the function will throw on error instead.
    ///
    /// \note The value returned by the thread function will be interpreted by
    ///       the thread manager as the new thread state the executed PX-thread
    ///       needs to be switched to. Normally, PX-threads will either return
    ///       \a threads#terminated (if the thread should be destroyed) or
    ///       \a threads#suspended (if the thread needs to be suspended because
    ///       it is waiting for an external event to happen). The external
    ///       event will set the state of the thread back to pending, which
    ///       will re-schedule the PX-thread.
    ///
    /// \throws invalid_status if the runtime system has not been started yet.
    ///
    HPX_API_EXPORT void register_work_plain(
        BOOST_RV_REF(HPX_STD_FUNCTION<threads::thread_function_type>) func,
        char const* description = 0, naming::address::address_type lva = 0,
        threads::thread_state_enum initial_state = threads::pending,
        threads::thread_priority priority = threads::thread_priority_normal,
        std::size_t os_thread = std::size_t(-1), error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new work item using the given function as the
    ///        work to be executed.
    ///
    /// \param func       [in] The function to be executed as the thread-function.
    ///                   This function has to expose the minimal low level
    ///                   PX-thread interface, i.e. it takes one argument (a
    ///                   \a threads#thread_state_ex_enum). The thread will be
    ///                   terminated after the function returns.
    ///
    /// \note All other arguments are equivalent to those of the function
    ///       \a applier#register_work_plain
    ///
    HPX_API_EXPORT void register_work(
        BOOST_RV_REF(HPX_STD_FUNCTION<void(threads::thread_state_ex_enum)>) func,
        char const* description = 0,
        threads::thread_state_enum initial_state = threads::pending,
        threads::thread_priority priority = threads::thread_priority_normal,
        std::size_t os_thread = std::size_t(-1), error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new work item using the given function as the
    ///        work to be executed.
    ///
    /// \param func       [in] The function to be executed as the thread-function.
    ///                   This function has to expose the minimal low level
    ///                   PX-thread interface, i.e. it takes no arguments. The
    ///                   thread will be terminated after the function returns.
    ///
    /// \note All other arguments are equivalent to those of the function
    ///       \a applier#register_work_plain
    ///
    HPX_API_EXPORT void register_work_nullary(
        BOOST_RV_REF(HPX_STD_FUNCTION<void()>) func, char const* description = 0,
        threads::thread_state_enum initial_state = threads::pending,
        threads::thread_priority priority = threads::thread_priority_normal,
        std::size_t os_thread = std::size_t(-1), error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a new work item using the given function as the
    ///        work to be executed.
    ///
    /// \note This function is completely equivalent to the first overload
    ///       of applier#register_work_plain above, except that part of the
    ///       parameters are passed as members of the threads#thread_init_data
    ///       object.
    ///
    HPX_API_EXPORT void register_work_plain(
        threads::thread_init_data& data,
        threads::thread_state_enum initial_state = threads::pending,
        error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// The \a create_async function initiates the creation of a new
    /// component instance using the runtime_support as given by targetgid.
    /// This function is non-blocking as it returns a \a lcos#future. The
    /// caller of this create_async is responsible to call
    /// \a lcos#future#get to obtain the result.
    ///
    /// \param targetgid
    /// \param type
    /// \param count
    ///
    /// \returns    The function returns a \a lcos#future instance
    ///             returning the the global id of the newly created
    ///             component when used to call get.
    ///
    /// \note       For synchronous operation use the function
    ///             \a applier#create_async.
    HPX_API_EXPORT lcos::future<naming::id_type>
        create_async(naming::id_type const& targetgid,
            components::component_type type, std::size_t count = 1);

    ///////////////////////////////////////////////////////////////////////////
    /// The \a create function creates a new component instance using the
    /// \a runtime_support as given by targetgid. This function is blocking
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
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
