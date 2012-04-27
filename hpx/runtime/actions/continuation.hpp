//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_CONTINUATION_JUN_13_2008_1031AM)
#define HPX_RUNTIME_ACTIONS_CONTINUATION_JUN_13_2008_1031AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/traits/needs_guid_initialization.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/base_object.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/detail/serialization_registration.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>
#include <boost/move/move.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
// Version of continuation
#define HPX_CONTINUATION_VERSION 0x10

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    namespace detail
    {
        template <typename Target>
        void guid_initialization(boost::mpl::false_) {}

        template <typename Target>
        void guid_initialization(boost::mpl::true_)
        {
            // force serialization self registration to happen
            using namespace boost::archive::detail::extra_detail;
            init_guid<Target>::g.initialize();
        }

        template <typename Target>
        void guid_initialization()
        {
            guid_initialization<Target>(
                typename traits::needs_guid_initialization<Target>::type());
        }

        ///////////////////////////////////////////////////////////////////////
        // Helper to invoke the registration code for serialization at startup
        template <typename Target>
        struct register_base_helper
        {
            register_base_helper()
            {
                Target::register_base();
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////
    // Parcel continuations are polymorphic objects encapsulating the
    // id_type of the destination where the result has to be sent.
    class HPX_EXPORT continuation
    {
    public:
        continuation()
        {}

        explicit continuation(naming::id_type const& gid)
          : gid_(gid)
        {}

        virtual ~continuation() {}

        //
        virtual void trigger() const;

        template <typename Arg0>
        inline void trigger(BOOST_FWD_REF(Arg0) arg0) const;

        //
        virtual void trigger_error(boost::exception_ptr const& e) const;
        virtual void trigger_error(BOOST_RV_REF(boost::exception_ptr) e) const;

        naming::id_type const& get_gid() const
        {
            return gid_;
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & gid_;
        }

    protected:
        naming::id_type gid_;
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct typed_continuation : continuation
    {
        typed_continuation()
        {}

        explicit typed_continuation(naming::id_type const& gid)
          : continuation(gid)
        {}

        ~typed_continuation()
        {
            detail::guid_initialization<typed_continuation>();
        }

        virtual void trigger_value(BOOST_RV_REF(Result)) const = 0;

        static void register_base()
        {
            util::void_cast_register_nonvirt<
                typed_continuation, continuation>();
        }

    private:
        /// serialization support
        friend class boost::serialization::access;
        typedef continuation base_type;

        template <class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & util::base_object_nonvirt<base_type>(*this);
        }
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename Arg0>
    void continuation::trigger(BOOST_FWD_REF(Arg0) arg0) const
    {
        // The static_cast is safe as we know that Arg0 is the result type
        // of the executed action (see apply.hpp).
        static_cast<typed_continuation<Arg0> const*>(this)->trigger_value(
            boost::forward<Arg0>(arg0));
    }
}}

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the id_type serialization format
#ifdef __GNUG__
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
BOOST_CLASS_VERSION(hpx::actions::continuation, HPX_CONTINUATION_VERSION)
#ifdef __GNUG__
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic pop
#endif
#endif

// registration code for serialization
HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (template <typename Result>),
    (hpx::actions::typed_continuation<Result>)
)

#include <hpx/config/warnings_suffix.hpp>

#endif
