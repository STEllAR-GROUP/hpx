//  Copyright (c) 2007-2008 Anshul Tandon
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
#include <hpx/assert.hpp>
#include <hpx/async_distributed/applier_fwd.hpp>    // this needs to go first
#include <hpx/components_base/component_type.hpp>
#include <hpx/modules/threadmanager.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace applier {
    /// The \a applier class is used to decide whether a particular action
    /// has to be issued on a local or a remote resource. If the target
    /// component is local a new \a thread will be created, if the target is
    /// remote a parcel will be sent.
    class HPX_EXPORT applier
    {
    public:
        HPX_NON_COPYABLE(applier);

    public:
        // constructor
#if defined(HPX_HAVE_NETWORKING)
        applier(parcelset::parcelhandler& ph, threads::threadmanager& tm);
#else
        explicit applier(threads::threadmanager& tm);
#endif

        // destructor
        ~applier() = default;

        void initialize(std::uint64_t rts);

#if defined(HPX_HAVE_NETWORKING)
        /// \brief Access the \a parcelhandler instance associated with this
        ///        \a applier
        ///
        /// This function returns a reference to the parcel handler this
        /// applier instance has been created with.
        parcelset::parcelhandler& get_parcel_handler();
#endif

        /// \brief Access the \a thread-manager instance associated with this
        ///        \a applier
        ///
        /// This function returns a reference to the thread manager this
        /// applier instance has been created with.
        threads::threadmanager& get_thread_manager();

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
        std::uint32_t get_locality_id(error_code& ec = throws) const;

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
        bool get_raw_remote_localities(
            std::vector<naming::gid_type>& locality_ids,
            components::component_type type = components::component_invalid,
            error_code& ec = throws) const;

        bool get_remote_localities(std::vector<naming::id_type>& locality_ids,
            components::component_type type = components::component_invalid,
            error_code& ec = throws) const;

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
            components::component_type type =
                components::component_invalid) const;

        bool get_localities(std::vector<naming::id_type>& locality_ids,
            error_code& ec = throws) const;
        bool get_localities(std::vector<naming::id_type>& locality_ids,
            components::component_type type, error_code& ec = throws) const;

        /// By convention the runtime_support has a gid identical to the prefix
        /// of the locality the runtime_support is responsible for
        naming::gid_type const& get_runtime_support_raw_gid() const
        {
            HPX_ASSERT(runtime_support_id_);
            return runtime_support_id_.get_gid();
        }

        /// By convention the runtime_support has a gid identical to the prefix
        /// of the locality the runtime_support is responsible for
        naming::id_type const& get_runtime_support_gid() const
        {
            HPX_ASSERT(runtime_support_id_);
            return runtime_support_id_;
        }

    private:
#if defined(HPX_HAVE_NETWORKING)
        parcelset::parcelhandler& parcel_handler_;
#endif
        threads::threadmanager& thread_manager_;
        naming::id_type runtime_support_id_;
    };
}}    // namespace hpx::applier

#include <hpx/config/warnings_suffix.hpp>

#endif
