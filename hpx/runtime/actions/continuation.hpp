//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_CONTINUATION_JUN_13_2008_1031AM)
#define HPX_RUNTIME_ACTIONS_CONTINUATION_JUN_13_2008_1031AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/base_object.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/detail/serialization_registration.hpp>
#include <hpx/runtime/actions/guid_initialization.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>
#include <boost/archive/detail/check.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
// Version of continuation
#define HPX_CONTINUATION_VERSION 0x10

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
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
        {
            // continuations with invalid id do not make sense
            BOOST_ASSERT(gid_);
        }

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

            // continuations with invalid id do not make sense
            BOOST_ASSERT(gid_);
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

        template <typename F>
        explicit typed_continuation(naming::id_type const& gid,
                BOOST_FWD_REF(F) f)
          : continuation(gid), f_(boost::move(f))
        {}

        virtual ~typed_continuation()
        {
            detail::guid_initialization<typed_continuation>();
        }

        void trigger_value(BOOST_RV_REF(Result) result) const
        {
            LLCO_(info) << "continuation::trigger(" << this->get_gid() << ")";
            if (!f_.empty())
                f_(this->get_gid(), boost::move(result));
            else {
                typedef typename
                    lcos::template base_lco_with_value<Result>::set_value_action
                set_value_action_type;

                hpx::apply<set_value_action_type>(this->get_gid(), boost::move(result));
            }
        }

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
            // serialize function
            bool have_function = !f_.empty();
            ar & have_function;
            if (have_function)
                ar & f_;

            // serialize base class
            ar & util::base_object_nonvirt<base_type>(*this);
        }

        util::function<void(naming::id_type, Result)> f_;
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

// Avoid compile time warnings about serializing continuations using pointers
// without object tracking.
namespace boost { namespace archive { namespace detail
{
    template <>
    inline void check_object_tracking<hpx::actions::continuation>() {}

    template <>
    inline void check_pointer_tracking<hpx::actions::continuation>() {}
}}}

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the id_type serialization format
#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

BOOST_CLASS_VERSION(hpx::actions::continuation, HPX_CONTINUATION_VERSION)
BOOST_CLASS_TRACKING(hpx::actions::continuation, boost::serialization::track_never)

#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic pop
#endif
#endif

// registration code for serialization
HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (template <typename Result>),
    (hpx::actions::typed_continuation<Result>)
)

#define HPX_REGISTER_TYPED_CONTINUATION_DECLARATION(Result, Name)             \
    namespace hpx { namespace traits {                                        \
        template <>                                                           \
        struct needs_guid_initialization<                                     \
                hpx::actions::typed_continuation<Result> >                    \
          : boost::mpl::false_                                                \
        {};                                                                   \
    }}                                                                        \
    namespace boost { namespace archive { namespace detail {                  \
        namespace extra_detail {                                              \
            template <>                                                       \
            struct init_guid<hpx::actions::typed_continuation<Result> >;      \
        }                                                                     \
    }}}                                                                       \
    BOOST_CLASS_EXPORT_KEY2(hpx::actions::typed_continuation<Result>,         \
        BOOST_PP_STRINGIZE(Name))                                             \
/**/

#define HPX_REGISTER_TYPED_CONTINUATION(Result, Name)                         \
    HPX_REGISTER_BASE_HELPER(                                                 \
        hpx::actions::typed_continuation<Result>, Name)                       \
    BOOST_CLASS_EXPORT_IMPLEMENT(                                             \
        hpx::actions::typed_continuation<Result>)                             \
/**/

#include <hpx/config/warnings_suffix.hpp>

#endif
