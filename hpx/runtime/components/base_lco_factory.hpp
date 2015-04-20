//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_BASE_LCO_FACTORY_OCT_10_2013_1118AM)
#define HPX_BASE_LCO_FACTORY_OCT_10_2013_1118AM

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>

#include <hpx/runtime/components/unique_component_name.hpp>
#include <hpx/runtime/components/component_factory_base.hpp>
#include <hpx/runtime/components/server/create_component.hpp>
#include <hpx/runtime/components/server/destroy_component.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/one_size_heap_list_base.hpp>
#include <hpx/util/detail/count_num_args.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/detail/atomic_count.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        HPX_EXPORT util::one_size_heap_list_base* create_promise_heap(
            components::component_type type);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The \a base_lco_factory provides a special implementation of a
    /// component factory for components exposing the base_lco interface.
    ///
    struct base_lco_factory : public component_factory_base
    {
        /// \brief Construct a new factory instance
        ///
        /// \note The contents of both sections has to be cloned in order to
        ///       save the configuration setting for later use.
        base_lco_factory(component_type type)
          : type_(type), heap_(detail::create_promise_heap(type))
        {}

        ///
        ~base_lco_factory() {}

        /// \brief Return the unique identifier of the component type this
        ///        factory is responsible for
        ///
        /// \param locality     [in] The id of the locality this factory
        ///                     is responsible for.
        /// \param agas_client  [in] The AGAS client to use for component id
        ///                     registration (if needed).
        ///
        /// \return Returns the unique identifier of the component type this
        ///         factory instance is responsible for. This function throws
        ///         on any error.
        component_type get_component_type(naming::gid_type const& locality,
            naming::resolver_client& agas_client)
        {
            return type_;
        }

        /// \brief Return the name of the component type this factory is
        ///        responsible for
        ///
        /// \return Returns the name of the component type this factory
        ///         instance is responsible for. This function throws on any
        ///         error.
        std::string get_component_name() const
        {
            return std::string("base_lco_factory for ") +
                get_component_type_name(type_);
        }

        /// \brief Create one or more new component instances.
        ///
        /// \param count  [in] The number of component instances to
        ///               create. The value of this parameter should not
        ///               be zero.
        ///
        /// \return Returns the GID of the first newly created component
        ///         instance. If more than one component instance has been
        ///         created (\a count > 1) the GID's of all new instances are
        ///         sequential in a row.
        naming::gid_type create(std::size_t = 1)
        {
            HPX_THROW_EXCEPTION(bad_request,
                "base_lco_factory::create",
                "this function should be never called");
            return naming::invalid_gid;
        }

        /// \brief Create one new component instance and initialize it using
        ///        the using the given constructor function.
        ///
        /// \param f  [in] The constructor function to call in order to
        ///           initialize the newly allocated object.
        ///
        /// \return   Returns the GID of the first newly created component
        ///           instance.
        naming::gid_type create_with_args(
            util::function_nonser<void(void*)> const&)
        {
            HPX_THROW_EXCEPTION(bad_request,
                "base_lco_factory::create_with_args",
                "this function should be never called");
            return naming::invalid_gid;
        }

        /// \brief Create one new component instance and initialize it using
        ///        the using the given constructor function. Assign the give
        ///        GID to the new object.
        ///
        /// \param assign_gid [in] The GID to assign to the newly created object.
        /// \param f  [in] The constructor function to call in order to
        ///           initialize the newly allocated object.
        ///
        /// \return   Returns the GID of the first newly created component
        ///           instance (this is the same as assign_gid, if successful).
        naming::gid_type create_with_args(
            naming::gid_type const& assign_gid,
            util::function_nonser<void(void*)> const& f)
        {
            HPX_THROW_EXCEPTION(bad_request,
                "base_lco_factory::create_with_args",
                "this function should be never called");
            return naming::invalid_gid;
        }

    public:
        /// \brief Destroy one or more component instances
        ///
        /// \param gid    [in] The gid of the first component instance to
        ///               destroy.
        /// \param addr   [in] The resolved address of the first component
        ///               instance to destroy.
        void destroy(naming::gid_type const& gid,
            naming::address const& addr)
        {
            server::destroy_base_lco(gid, addr, heap_.get(), type_);
        }

        /// \brief Ask how many instances are alive of the type this factory is
        ///        responsible for
        ///
        /// \return Returns the number of instances of the managed object type
        ///         which are currently alive.
        long instance_count() const
        {
            return 0;
        }

#if defined(HPX_HAVE_SECURITY)
        /// \brief Return the required capabilities necessary to create an
        ///        instance of a component using this factory instance.
        ///
        /// \return Returns required capabilities necessary to create a new
        ///         instance of a component using this factory instance.
        components::security::capability get_required_capabilities() const
        {
            using namespace components::security;
            return traits::capability<>::capability_create_component;
        }
#endif

        boost::shared_ptr<util::one_size_heap_list_base> get_heap() const
        {
            return heap_;
        }

    protected:
        component_type type_;
        boost::shared_ptr<util::one_size_heap_list_base> heap_;
    };
}}

#endif

