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
#include <hpx/util/polymorphic_factory.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/serialize_empty_type.hpp>
#include <hpx/util/demangle_helper.hpp>
#include <hpx/util/remove_reference.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_callable.hpp>

#include <boost/enable_shared_from_this.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    //////////////////////////////////////////////////////////////////////////
    // forward declare the required overload of apply.
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg>
    inline bool
    apply(hpx::actions::action<Component, Result, Arguments, Derived>,
        naming::id_type const&, BOOST_FWD_REF(Arg));

    // MSVC complains about async_continue beeing ambiguous if it sees this
    // forward declaration
    template <typename F, typename Arg1, typename Arg2>
    inline typename boost::enable_if_c<
        traits::detail::is_callable_not_action<F(Arg1, Arg2)>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , bool
    >::type
    apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(Arg1), BOOST_FWD_REF(Arg2));

    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0, typename F>
    inline typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 1,
        bool
    >::type
    apply_continue(hpx::actions::action<Component, Result, Arguments, Derived>,
        naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0,
        BOOST_FWD_REF(F) f);

    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0>
    inline bool
    apply_c(hpx::actions::action<Component, Result, Arguments, Derived>,
        naming::id_type const& contgid, naming::id_type const& gid,
        BOOST_FWD_REF(Arg0) arg0);

    //////////////////////////////////////////////////////////////////////////
    // handling special case of triggering an LCO
    HPX_API_EXPORT void trigger_lco_event(naming::id_type const& id);

    template <typename T>
    void set_lco_value(naming::id_type const& id, BOOST_FWD_REF(T) t)
    {
        typename lcos::base_lco_with_value<
            typename util::remove_reference<T>::type
        >::set_value_action set;
        apply(set, id, util::detail::make_temporary<T>::call(t));
    }

    HPX_API_EXPORT void set_lco_error(naming::id_type const& id,
        boost::exception_ptr const& e);
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    class HPX_EXPORT continuation;

    namespace detail
    {
        template <typename Continuation>
        char const* get_continuation_name()
#ifdef HPX_DISABLE_AUTOMATIC_SERIALIZATION_REGISTRATION
        ;
#else
        {
            // If you encounter this assert while compiling code, that means that
            // you have a HPX_REGISTER_TYPED_CONTINUATION macro somewhere in a
            // source file, but the header in which the continuation is defined
            // misses a HPX_REGISTER_TYPED_CONTINUATION_DECLARATION
            BOOST_MPL_ASSERT_MSG(
                traits::needs_automatic_registration<Continuation>::value
              , HPX_REGISTER_TYPED_CONTINUATION_DECLARATION_MISSING
              , (Continuation)
            );
            return util::type_id<Continuation>::typeid_.type_id();
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        template <typename Continuation>
        struct continuation_registration
        {
            static boost::shared_ptr<continuation> create()
            {
                return boost::shared_ptr<continuation>(new Continuation());
            }

            continuation_registration()
            {
                util::polymorphic_factory<continuation>::get_instance().
                    add_factory_function(
                        detail::get_continuation_name<Continuation>()
                      , &continuation_registration::create
                    );
            }
        };

        template <typename Continuation, typename Enable =
            typename traits::needs_automatic_registration<Continuation>::type>
        struct automatic_continuation_registration
        {
            automatic_continuation_registration()
            {
                continuation_registration<Continuation> auto_register;
            }

            automatic_continuation_registration & register_continuation()
            {
                return *this;
            }
        };

        template <typename Continuation>
        struct automatic_continuation_registration<Continuation, boost::mpl::false_>
        {
            automatic_continuation_registration()
            {
            }

            automatic_continuation_registration & register_continuation()
            {
                return *this;
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // Parcel continuations are polymorphic objects encapsulating the
    // id_type of the destination where the result has to be sent.
    class HPX_EXPORT continuation
      : public boost::enable_shared_from_this<continuation>
    {
    public:
        continuation() {}

        explicit continuation(naming::id_type const& gid)
          : gid_(gid)
        {
            // continuations with invalid id do not make sense
            HPX_ASSERT(gid_);
        }

        explicit continuation(BOOST_RV_REF(naming::id_type) gid)
          : gid_(boost::move(gid))
        {
            // continuations with invalid id do not make sense
            HPX_ASSERT(gid_);
        }

        virtual ~continuation() {}

        //
        virtual void trigger() const;

        template <typename Arg0>
        inline void trigger(BOOST_FWD_REF(Arg0) arg0) const;

        //
        virtual void trigger_error(boost::exception_ptr const& e) const;
        virtual void trigger_error(BOOST_RV_REF(boost::exception_ptr) e) const;

        virtual char const* get_continuation_name() const = 0;

        // serialization support
        virtual void load(hpx::util::portable_binary_iarchive& ar)
        {
            ar >> gid_;
        }
        virtual void save(hpx::util::portable_binary_oarchive& ar) const
        {
            ar << gid_;
        }

        naming::id_type const& get_gid() const
        {
            return gid_;
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
            hpx::apply_c(cont_, lco, target_, boost::forward<T>(t));
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive& ar, unsigned int const)
        {
            ar & cont_ & target_;
        }

        typedef typename hpx::util::remove_reference<Cont>::type cont_type;
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
            hpx::apply_continue(cont_, target_, boost::forward<T>(t), hpx::util::bind(f_, lco, _2));
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive& ar, unsigned int const)
        {
            ar & cont_ & target_ & f_;
        }

        typedef typename hpx::util::remove_reference<Cont>::type cont_type;
        typedef typename hpx::util::remove_reference<F>::type function_type;

        cont_type cont_;        // continuation type
        hpx::id_type target_;
        function_type f_;       // set_value action  (default: set_lco_value_continuation)
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Continuation>
    struct init_registration;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct typed_continuation : continuation
    {
        typed_continuation()
        {}

        explicit typed_continuation(naming::id_type const& gid)
          : continuation(gid)
        {}

        explicit typed_continuation(BOOST_RV_REF(naming::id_type) gid)
          : continuation(boost::move(gid))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type const& gid,
                BOOST_FWD_REF(F) f)
          : continuation(gid), f_(boost::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(BOOST_RV_REF(naming::id_type) gid,
                BOOST_FWD_REF(F) f)
          : continuation(boost::move(gid)), f_(boost::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(BOOST_FWD_REF(F) f)
          : f_(boost::forward<F>(f))
        {}

        virtual ~typed_continuation()
        {
            init_registration<typed_continuation>::g.register_continuation();
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

    private:
        char const* get_continuation_name() const
        {
            return detail::get_continuation_name<typed_continuation>();
        }

        /// serialization support
        void load(hpx::util::portable_binary_iarchive& ar)
        {
            // serialize base class
            typedef continuation base_type;
            this->base_type::load(ar);

            // serialize function
            bool have_function = false;
            ar.load(have_function);
            if (have_function)
                ar >> f_;
        }
        void save(hpx::util::portable_binary_oarchive& ar) const
        {
            // serialize base class
            typedef continuation base_type;
            this->base_type::save(ar);

            // serialize function
            bool have_function = !f_.empty();
            ar.save(have_function);
            if (have_function)
                ar << f_;
        }

        util::function<void(naming::id_type, Result)> f_;
    };

    ///////////////////////////////////////////////////////////////////////////
    // registration code for serialization
    template <typename Result>
    struct init_registration<typed_continuation<Result> >
    {
        static detail::automatic_continuation_registration<typed_continuation<Result> > g;
    };
}}

namespace hpx { namespace traits
{
    template <>
    struct needs_automatic_registration<
            hpx::actions::typed_continuation<hpx::util::unused_type> >
        : boost::mpl::false_
    {};
}}

namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    template <>
    struct typed_continuation<hpx::util::unused_type> : continuation
    {
        typed_continuation()
        {}

        explicit typed_continuation(naming::id_type const& gid)
          : continuation(gid)
        {}

        explicit typed_continuation(BOOST_RV_REF(naming::id_type) gid)
          : continuation(boost::move(gid))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type const& gid,
                BOOST_FWD_REF(F) f)
          : continuation(gid), f_(boost::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(BOOST_RV_REF(naming::id_type) gid,
                BOOST_FWD_REF(F) f)
          : continuation(boost::move(gid)), f_(boost::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(BOOST_FWD_REF(F) f)
          : f_(boost::forward<F>(f))
        {}

        virtual ~typed_continuation()
        {
            init_registration<typed_continuation>::g.register_continuation();
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

    private:
        char const* get_continuation_name() const
        {
            return "hpx_unused_typed_continuation";
        }

        /// serialization support
        void load(hpx::util::portable_binary_iarchive& ar)
        {
            // serialize base class
            typedef continuation base_type;
            this->base_type::load(ar);

            // serialize function
            bool have_function = false;
            ar.load(have_function);
            if (have_function)
                ar >> f_;
        }
        void save(hpx::util::portable_binary_oarchive& ar) const
        {
            // serialize base class
            typedef continuation base_type;
            this->base_type::save(ar);

            // serialize function
            bool have_function = !f_.empty();
            ar.save(have_function);
            if (have_function)
                ar << f_;
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

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    detail::automatic_continuation_registration<typed_continuation<Result> >
        init_registration<typed_continuation<Result> >::g =
            detail::automatic_continuation_registration<typed_continuation<Result> >();
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
        typename hpx::util::remove_reference<Cont>::type
    >
    make_continuation(BOOST_FWD_REF(Cont) cont)
    {
        typedef typename hpx::util::remove_reference<Cont>::type cont_type;
        return hpx::actions::continuation_impl<cont_type>(
            boost::forward<Cont>(cont), hpx::find_here());
    }

    template <typename Cont>
    inline hpx::actions::continuation_impl<
        typename hpx::util::remove_reference<Cont>::type
    >
    make_continuation(BOOST_FWD_REF(Cont) f, hpx::id_type const& target)
    {
        typedef typename hpx::util::remove_reference<Cont>::type cont_type;
        return hpx::actions::continuation_impl<cont_type>(
            boost::forward<Cont>(f), target);
    }

    template <typename Cont, typename F>
    inline typename boost::disable_if<
        boost::is_same<
            typename hpx::util::remove_reference<F>::type,
            hpx::naming::id_type
        >,
        hpx::actions::continuation2_impl<
            typename hpx::util::remove_reference<Cont>::type,
            typename hpx::util::remove_reference<F>::type
        >
    >::type
    make_continuation(BOOST_FWD_REF(Cont) cont, BOOST_FWD_REF(F) f)
    {
        typedef typename hpx::util::remove_reference<Cont>::type cont_type;
        typedef typename hpx::util::remove_reference<F>::type function_type;

        return hpx::actions::continuation2_impl<cont_type, function_type>(
            boost::forward<Cont>(cont), hpx::find_here(), boost::forward<F>(f));
    }

    template <typename Cont, typename F>
    inline hpx::actions::continuation2_impl<
        typename hpx::util::remove_reference<Cont>::type,
        typename hpx::util::remove_reference<F>::type
    >
    make_continuation(BOOST_FWD_REF(Cont) cont, hpx::id_type const& target,
        BOOST_FWD_REF(F) f)
    {
        typedef typename hpx::util::remove_reference<Cont>::type cont_type;
        typedef typename hpx::util::remove_reference<F>::type function_type;

        return hpx::actions::continuation2_impl<cont_type, function_type>(
            boost::forward<Cont>(cont), target, boost::forward<F>(f));
    }
}

///////////////////////////////////////////////////////////////////////////////
#define HPX_CONTINUATION_REGISTER_CONTINUATION_FACTORY(Continuation, Name)    \
    static ::hpx::actions::detail::continuation_registration<Continuation>    \
        const BOOST_PP_CAT(Name, _continuation_factory_registration) =        \
        ::hpx::actions::detail::continuation_registration<Continuation>();    \
/**/

#define HPX_DECLARE_GET_CONTINUATION_NAME_(continuation, name)                \
    namespace hpx { namespace actions { namespace detail {                    \
        template<> HPX_ALWAYS_EXPORT                                          \
        char const* get_continuation_name<continuation>();                    \
    }}}                                                                       \
/**/

#define HPX_REGISTER_TYPED_CONTINUATION_DECLARATION(Result, Name)             \
    HPX_DECLARE_GET_CONTINUATION_NAME_(                                       \
        hpx::actions::typed_continuation<Result>, Name)                       \
    namespace hpx { namespace traits {                                        \
        template <>                                                           \
        struct needs_automatic_registration<                                  \
                hpx::actions::typed_continuation<Result> >                    \
          : boost::mpl::false_                                                \
        {};                                                                   \
    }}                                                                        \
/**/

#define HPX_DEFINE_GET_CONTINUATION_NAME_(continuation, name)                 \
    namespace hpx { namespace actions { namespace detail {                    \
        template<> HPX_ALWAYS_EXPORT                                          \
        char const* get_continuation_name<continuation>()                     \
        {                                                                     \
            return BOOST_PP_STRINGIZE(name);                                  \
        }                                                                     \
    }}}                                                                       \
/**/

#define HPX_REGISTER_TYPED_CONTINUATION(Result, Name)                         \
    HPX_CONTINUATION_REGISTER_CONTINUATION_FACTORY(                           \
        hpx::actions::typed_continuation<Result>, Name)                       \
    HPX_DEFINE_GET_CONTINUATION_NAME_(                                        \
        hpx::actions::typed_continuation<Result>, Name)                       \
/**/

#include <hpx/config/warnings_suffix.hpp>

#endif
