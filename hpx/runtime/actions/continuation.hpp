//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_CONTINUATION_JUN_13_2008_1031AM)
#define HPX_RUNTIME_ACTIONS_CONTINUATION_JUN_13_2008_1031AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/actions/guid_initialization.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/base_object.hpp>
#include <hpx/util/serialize_empty_type.hpp>
#include <hpx/util/detail/serialization_registration.hpp>
#include <hpx/util/detail/remove_reference.hpp>
#include <hpx/traits/is_callable.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>
#include <boost/archive/detail/check.hpp>
#include <boost/enable_shared_from_this.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
// Version of continuation
#define HPX_CONTINUATION_VERSION 0x10

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    //////////////////////////////////////////////////////////////////////////
    // forward declare the required overload of apply.
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg>
    inline bool apply(
        hpx::actions::action<Component, Result, Arguments, Derived>,
        naming::id_type const&, BOOST_FWD_REF(Arg));

#if !defined(BOOST_MSVC)
    // MSVC complains about async_continue beeing ambiguous if it sees this
    // forward declaration
    template <typename F, typename Arg1, typename Arg2>
    inline typename boost::enable_if<
        traits::is_callable<F>, bool
    >::type
    apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg1), BOOST_FWD_REF(Arg2));

    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0, typename F>
    inline typename boost::enable_if<
        boost::mpl::bool_<boost::fusion::result_of::size<Arguments>::value == 1>,
        bool
    >::type apply_continue(
        hpx::actions::action<Component, Result, Arguments, Derived>,
        naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0,
        BOOST_FWD_REF(F) f);
#endif

    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0>
    inline bool apply_c(
        hpx::actions::action<Component, Result, Arguments, Derived>,
        naming::id_type const& contgid, naming::id_type const& gid,
        BOOST_FWD_REF(Arg0) arg0);

    //////////////////////////////////////////////////////////////////////////
    // handling special case of triggering an LCO
    HPX_API_EXPORT void trigger_lco_event(naming::id_type const& id);

    template <typename T>
    void set_lco_value(naming::id_type const& id, BOOST_FWD_REF(T) t)
    {
        typename lcos::base_lco_with_value<
            typename util::detail::remove_reference<T>::type
        >::set_value_action set;
        apply(set, id, boost::move(t));
    }

    HPX_API_EXPORT void set_lco_error(naming::id_type const& id,
        boost::exception_ptr const& e);
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    // Parcel continuations are polymorphic objects encapsulating the
    // id_type of the destination where the result has to be sent.
    class HPX_EXPORT continuation
      : public boost::enable_shared_from_this<continuation>
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

        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & gid_;
        }

    protected:
        naming::id_type gid_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct set_lco_value_continuation
    {
        typedef void result_type;

        template <typename T>
        BOOST_FORCEINLINE void operator()(id_type const& lco,
            BOOST_FWD_REF(T) t) const
        {
            hpx::set_lco_value(lco, boost::forward<T>(t));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Cont>
    struct continuation_impl
    {
        typedef void result_type;

        continuation_impl() {}

        template <typename Cont_>
        continuation_impl(BOOST_FWD_REF(Cont_) cont, hpx::id_type const& target)
          : cont_(boost::forward<Cont_>(cont)), target_(target)
        {}

        template <typename T>
        void operator()(hpx::id_type const& lco, BOOST_FWD_REF(T) t) const
        {
            hpx::apply_c(cont_, lco, target_, t);
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive& ar, unsigned int const)
        {
            ar & cont_ & target_;
        }

        typedef typename hpx::util::detail::remove_reference<Cont>::type cont_type;
        cont_type cont_;
        hpx::id_type target_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Cont, typename F>
    struct continuation2_impl
    {
        typedef void result_type;

        continuation2_impl() {}

        template <typename Cont_, typename F_>
        continuation2_impl(BOOST_FWD_REF(Cont_) cont, hpx::id_type const& target,
                BOOST_FWD_REF(F_) f)
          : cont_(boost::forward<Cont_>(cont)),
            target_(target),
            f_(boost::forward<F_>(f))
        {}

        template <typename T>
        void operator()(hpx::id_type const& lco, BOOST_FWD_REF(T) t) const
        {
            using hpx::util::placeholders::_2;
            hpx::apply_continue(cont_, target_, t, hpx::util::bind(f_, lco, _2));
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive& ar, unsigned int const)
        {
            ar & cont_ & target_ & f_;
        }

        typedef typename hpx::util::detail::remove_reference<Cont>::type cont_type;
        typedef typename hpx::util::detail::remove_reference<F>::type function_type;

        cont_type cont_;        // continuation type
        hpx::id_type target_;
        function_type f_;       // set_value action  (default: set_lco_value_continuation)
    };

    ///////////////////////////////////////////////////////////////////////////
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
          : continuation(gid), f_(boost::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(BOOST_FWD_REF(F) f)
          : f_(boost::forward<F>(f))
        {}

        virtual ~typed_continuation()
        {
            detail::guid_initialization<typed_continuation>();
        }

        void trigger_value(BOOST_RV_REF(Result) result) const
        {
            LLCO_(info)
                << "typed_continuation<Result>::trigger_value("
                << this->get_gid() << ")";
            if (f_.empty()) {
                if (!this->get_gid()) {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "typed_continuation<Result>::trigger_value",
                        "attempt to trigger invalid LCO (the id is invalid)");
                    return;
                }
                hpx::set_lco_value(this->get_gid(), boost::move(result));
            }
            else {
                f_(this->get_gid(), boost::move(result));
            }
        }

        static void register_base()
        {
            util::void_cast_register_nonvirt<typed_continuation, continuation>();
        }

    private:
        /// serialization support
        friend class boost::serialization::access;
        typedef continuation base_type;

        template <class Archive>
        BOOST_FORCEINLINE void serialize(Archive& ar, const unsigned int /*version*/)
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

    template <>
    struct typed_continuation<hpx::util::unused_type> : continuation
    {
        typed_continuation()
        {}

        explicit typed_continuation(naming::id_type const& gid)
          : continuation(gid)
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type const& gid,
                BOOST_FWD_REF(F) f)
          : continuation(gid), f_(boost::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(BOOST_FWD_REF(F) f)
          : f_(boost::forward<F>(f))
        {}

        virtual ~typed_continuation()
        {
            detail::guid_initialization<typed_continuation>();
        }

        void trigger() const
        {
            LLCO_(info)
                << "typed_continuation<hpx::util::unused_type>::trigger("
                << this->get_gid() << ")";
            if (f_.empty()) {
                if (!this->get_gid()) {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "typed_continuation<hpx::util::unused_type>::trigger",
                        "attempt to trigger invalid LCO (the id is invalid)");
                    return;
                }
                trigger_lco_event(this->get_gid());
            }
            else {
                f_(this->get_gid());
            }
        }

        void trigger_value(BOOST_RV_REF(util::unused_type)) const
        {
            this->trigger();
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
        BOOST_FORCEINLINE void serialize(Archive& ar, const unsigned int /*version*/)
        {
            // serialize function
            bool have_function = !f_.empty();
            ar & have_function;
            if (have_function)
                ar & f_;

            // serialize base class
            ar & util::base_object_nonvirt<base_type>(*this);
        }

        util::function<void(naming::id_type)> f_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Arg0>
    void continuation::trigger(BOOST_FWD_REF(Arg0) arg0) const
    {
        // The static_cast is safe as we know that Arg0 is the result type
        // of the executed action (see apply.hpp).
        static_cast<typed_continuation<Arg0> const*>(this)->trigger_value(
            boost::forward<Arg0>(arg0));
    }
}}

//////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    inline hpx::actions::set_lco_value_continuation
    make_continuation()
    {
        return hpx::actions::set_lco_value_continuation();
    }

    template <typename Cont>
    inline hpx::actions::continuation_impl<
        typename hpx::util::detail::remove_reference<Cont>::type
    >
    make_continuation(BOOST_FWD_REF(Cont) cont)
    {
        typedef typename hpx::util::detail::remove_reference<Cont>::type cont_type;
        return hpx::actions::continuation_impl<cont_type>(
            boost::forward<Cont>(cont), hpx::find_here());
    }

    template <typename Cont>
    inline hpx::actions::continuation_impl<
        typename hpx::util::detail::remove_reference<Cont>::type
    >
    make_continuation(BOOST_FWD_REF(Cont) f, hpx::id_type const& target)
    {
        typedef typename hpx::util::detail::remove_reference<Cont>::type cont_type;
        return hpx::actions::continuation_impl<cont_type>(
            boost::forward<Cont>(f), target);
    }

    template <typename Cont, typename F>
    inline typename boost::disable_if<
        boost::is_same<
            typename hpx::util::detail::remove_reference<F>::type,
            hpx::naming::id_type
        >,
        hpx::actions::continuation2_impl<
            typename hpx::util::detail::remove_reference<Cont>::type,
            typename hpx::util::detail::remove_reference<F>::type
        >
    >::type
    make_continuation(BOOST_FWD_REF(Cont) cont, BOOST_FWD_REF(F) f)
    {
        typedef typename hpx::util::detail::remove_reference<Cont>::type cont_type;
        typedef typename hpx::util::detail::remove_reference<F>::type function_type;

        return hpx::actions::continuation2_impl<cont_type, function_type>(
            boost::forward<Cont>(cont), hpx::find_here(), boost::forward<F>(f));
    }

    template <typename Cont, typename F>
    inline hpx::actions::continuation2_impl<
        typename hpx::util::detail::remove_reference<Cont>::type,
        typename hpx::util::detail::remove_reference<F>::type
    >
    make_continuation(BOOST_FWD_REF(Cont) cont, hpx::id_type const& target,
        BOOST_FWD_REF(F) f)
    {
        typedef typename hpx::util::detail::remove_reference<Cont>::type cont_type;
        typedef typename hpx::util::detail::remove_reference<F>::type function_type;

        return hpx::actions::continuation2_impl<cont_type, function_type>(
            boost::forward<Cont>(cont), target, boost::forward<F>(f));
    }
}

//////////////////////////////////////////////////////////////////////////////
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
