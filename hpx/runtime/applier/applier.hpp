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
        naming::gid_type const& get_raw_locality(error_code& ec = throws) const;

        /// \brief Allow access to the id of the locality this applier
        ///        instance is associated with.
        ///
        /// This function returns a reference to the id of the locality this
        /// applier instance is associated with.
        boost::uint32_t get_locality_id(error_code& ec = throws) const;

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
            error_code& ec = throws) const;
        bool get_localities(std::vector<naming::id_type>& locality_ids,
            components::component_type type, error_code& ec = throws) const;

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
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
