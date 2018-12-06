//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_RUNTIME_SUPPORT_JUN_03_2008_0438PM)
#define HPX_COMPONENTS_RUNTIME_SUPPORT_JUN_03_2008_0438PM

#include <hpx/config.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a runtime_support class is the client side representation of a
    /// \a server#runtime_support component
    class HPX_EXPORT runtime_support : public stubs::runtime_support
    {
    private:
        typedef stubs::runtime_support base_type;

    public:
        /// Create a client side representation for the existing
        /// \a server#runtime_support instance with the given global id \a gid.
        runtime_support(naming::id_type const& gid = naming::invalid_id)
          : gid_(naming::invalid_id == gid ?
                naming::id_type(
                    applier::get_applier().get_runtime_support_raw_gid(),
                    naming::id_type::unmanaged
                ) : gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Create a new component type using the runtime_support
        template <typename Component, typename ...Ts>
        naming::id_type create_component(Ts&&... vs)
        {
            return this->base_type::template create_component<Component>
                (gid_, std::forward<Ts>(vs)...);
        }

        /// Asynchronously create a new component using the runtime_support
        template <typename Component, typename ...Ts>
        lcos::future<naming::id_type>
        create_component_async(Ts&&... vs)
        {
            return this->base_type::template create_component_async<Component>
                (gid_, std::forward<Ts>(vs)...);
        }


        /// Asynchronously create N new default constructed components using
        /// the runtime_support
        template <typename Component, typename ...Ts>
        std::vector<naming::id_type> bulk_create_component(
            std::size_t count, Ts&&...vs)
        {
            return this->base_type::template bulk_create_component<Component>
                (gid_, std::forward<Ts>(vs)...);
        }

        /// Asynchronously create a new component using the runtime_support
        template <typename Component, typename ...Ts>
        lcos::future<std::vector<naming::id_type> >
        bulk_create_components_async(
            std::size_t count, Ts&&...vs)
        {
            return this->base_type::template bulk_create_component<Component>
                (gid_, std::forward<Ts>(vs)...);
        }

        ///////////////////////////////////////////////////////////////////////
        lcos::future<int> load_components_async()
        {
            return this->base_type::load_components_async(gid_);
        }

        int load_components()
        {
            return this->base_type::load_components(gid_);
        }

        lcos::future<void> call_startup_functions_async(bool pre_startup)
        {
            return this->base_type::call_startup_functions_async(gid_, pre_startup);
        }

        void call_startup_functions(bool pre_startup)
        {
            this->base_type::call_startup_functions(gid_, pre_startup);
        }

        /// \brief Shutdown the given runtime system
        lcos::future<void> shutdown_async(double timeout = -1)
        {
            return this->base_type::shutdown_async(gid_, timeout);
        }

        void shutdown(double timeout = -1)
        {
            this->base_type::shutdown(gid_, timeout);
        }

        /// \brief Shutdown the runtime systems of all localities
        void shutdown_all(double timeout = -1)
        {
            this->base_type::shutdown_all(gid_, timeout);
        }

        /// \brief Terminate the given runtime system
        lcos::future<void> terminate_async()
        {
            return this->base_type::terminate_async(gid_);
        }

        void terminate()
        {
            this->base_type::terminate(gid_);
        }

        /// \brief Terminate the runtime systems of all localities
        void terminate_all()
        {
            this->base_type::terminate_all(gid_);
        }

        /// \brief Retrieve configuration information
        void get_config(util::section& ini)
        {
            this->base_type::get_config(gid_, ini);
        }

        ///////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_COMPONENT_GET_GID_COMPATIBILITY)
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        naming::id_type const& get_gid() const
        {
            return gid_;
        }
#endif

        naming::id_type const& get_id() const
        {
            return gid_;
        }

        naming::gid_type const & get_raw_gid() const
        {
            return gid_.get_gid();
        }

    private:
        naming::id_type gid_;
    };
}}

#endif
