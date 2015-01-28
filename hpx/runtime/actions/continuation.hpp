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
#include <hpx/util/invoke.hpp>
#include <hpx/util/polymorphic_factory.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/serialize_empty_type.hpp>
#include <hpx/util/demangle_helper.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_callable.hpp>

#include <boost/enable_shared_from_this.hpp>
#include <boost/type_traits/remove_reference.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    //////////////////////////////////////////////////////////////////////////
    // forward declare the required overload of apply.
    template <typename Component, typename Signature, typename Derived,
        typename ...Ts>
    inline bool
    apply(hpx::actions::basic_action<Component, Signature, Derived>,
        naming::id_type const&, Ts&&... vs);

    // MSVC complains about ambiguities if it sees this forward declaration
#ifndef BOOST_MSVC
    template <typename F, typename ...Ts>
    typename boost::enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(typename util::decay<Ts>::type...)
        >::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , bool
    >::type
    apply(F&& f, Ts&&... vs);

    template <
        typename Component, typename Signature, typename Derived,
        typename Cont, typename ...Ts>
    bool apply_continue(
        hpx::actions::basic_action<Component, Signature, Derived>,
        Cont&& cont, naming::id_type const& gid, Ts&&... vs);
#endif

    template <typename Component, typename Signature, typename Derived,
        typename ...Ts>
    inline bool
    apply_c(hpx::actions::basic_action<Component, Signature, Derived>,
        naming::id_type const& contgid, naming::id_type const& gid,
        Ts&&... vs);

    //////////////////////////////////////////////////////////////////////////
    // handling special case of triggering an LCO
    template <typename T>
    void set_lco_value(naming::id_type const& id, T && t)
    {
        typename lcos::base_lco_with_value<
            typename boost::remove_reference<T>::type
        >::set_value_action set;
        apply(set, id, util::detail::make_temporary<T>(t));
    }

    template <typename T>
    void set_lco_value(naming::id_type const& id, T && t,
        naming::id_type const& cont)
    {
        typename lcos::base_lco_with_value<
            typename boost::remove_reference<T>::type
        >::set_value_action set;
        apply_c(set, cont, id, util::detail::make_temporary<T>(t));
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

        explicit continuation(naming::id_type && gid)
          : gid_(std::move(gid))
        {
            // continuations with invalid id do not make sense
            HPX_ASSERT(gid_);
        }

        virtual ~continuation() {}

        //
        virtual void trigger() const;

        template <typename Arg0>
        inline void trigger(Arg0 && arg0) const;

        //
        virtual void trigger_error(boost::exception_ptr const& e) const;
        virtual void trigger_error(boost::exception_ptr && e) const;

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
    template <typename F, typename ...Ts>
    typename boost::disable_if_c<
        boost::is_void<typename util::result_of<F(Ts...)>::type>::value
    >::type trigger(continuation& cont, F&& f, Ts&&... vs)
    {
        typedef typename util::result_of<F(Ts...)>::type result_type;
        try {
            cont.trigger(util::invoke_r<result_type>(std::forward<F>(f),
                std::forward<Ts>(vs)...));
        }
        catch (...) {
            // make sure hpx::exceptions are propagated back to the client
            cont.trigger_error(boost::current_exception());
        }
    }

    template <typename F, typename ...Ts>
    typename boost::enable_if_c<
        boost::is_void<typename util::result_of<F(Ts...)>::type>::value
    >::type trigger(continuation& cont, F&& f, Ts&&... vs)
    {
        try {
            util::invoke_r<void>(std::forward<F>(f), std::forward<Ts>(vs)...);
            cont.trigger();
        }
        catch (...) {
            // make sure hpx::exceptions are propagated back to the client
            cont.trigger_error(boost::current_exception());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    struct set_lco_value_continuation
    {
        template <typename T>
        struct result;

        template <typename F, typename T1, typename T2>
        struct result<F(T1, T2)>
        {
            typedef T2 type;
        };

        template <typename T>
        BOOST_FORCEINLINE T operator()(id_type const& lco, T && t) const
        {
            hpx::set_lco_value(lco, std::forward<T>(t));

            // Yep, 't' is a zombie, however we don't use the returned value
            // anyways. We need it for result type calculation, though.
            return std::move(t);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Cont>
    struct continuation_impl
    {
        template <typename T>
        struct result;

        template <typename F, typename T1, typename T2>
        struct result<F(T1, T2)>
        {
            typedef T2 type;
        };

        continuation_impl() {}

        template <typename Cont_>
        continuation_impl(Cont_ && cont, hpx::id_type const& target)
          : cont_(std::forward<Cont_>(cont)), target_(target)
        {}

        template <typename T>
        T operator()(hpx::id_type const& lco, T && t) const
        {
            hpx::apply_c(cont_, lco, target_, std::forward<T>(t));

            // Yep, 't' is a zombie, however we don't use the returned value
            // anyways. We need it for result type calculation, though.
            return std::move(t);
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive& ar, unsigned int const)
        {
            ar & cont_ & target_;
        }

        typedef typename util::decay<Cont>::type cont_type;
        cont_type cont_;
        hpx::id_type target_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Cont, typename F>
    struct continuation2_impl
    {
        template <typename T>
        struct result;

        template <typename This, typename T1, typename T2>
        struct result<This(T1, T2)>
        {
            typedef T2 type;
        };

        continuation2_impl() {}

        template <typename Cont_, typename F_>
        continuation2_impl(Cont_ && cont, hpx::id_type const& target,
                F_ && f)
          : cont_(std::forward<Cont_>(cont)),
            target_(target),
            f_(std::forward<F_>(f))
        {}

        template <typename T>
        T operator()(hpx::id_type const& lco, T && t) const
        {
            using hpx::util::placeholders::_2;
            hpx::apply_continue(cont_, hpx::util::bind(f_, lco, _2),
                target_, std::forward<T>(t));

            // Yep, 't' is a zombie, however we don't use the returned value
            // anyways. We need it for result type calculation, though.
            return std::move(t);
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive& ar, unsigned int const)
        {
            ar & cont_ & target_ & f_;
        }

        typedef typename boost::remove_reference<Cont>::type cont_type;
        typedef typename boost::remove_reference<F>::type function_type;

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

        explicit typed_continuation(naming::id_type && gid)
          : continuation(std::move(gid))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type const& gid,
                F && f)
          : continuation(gid), f_(std::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type && gid,
                F && f)
          : continuation(std::move(gid)), f_(std::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(F && f)
          : f_(std::forward<F>(f))
        {}

        virtual ~typed_continuation()
        {
            init_registration<typed_continuation>::g.register_continuation();
        }

        virtual void trigger_value(Result && result) const
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
                hpx::set_lco_value(this->get_gid(), std::move(result));
            }
            else {
                f_(this->get_gid(), std::move(result));
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
            hpx::actions::typed_continuation<void> >
        : boost::mpl::false_
    {};
}}

namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    template <>
    struct typed_continuation<void> : continuation
    {
        typed_continuation()
        {}

        explicit typed_continuation(naming::id_type const& gid)
          : continuation(gid)
        {}

        explicit typed_continuation(naming::id_type && gid)
          : continuation(std::move(gid))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type const& gid,
                F && f)
          : continuation(gid), f_(std::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type && gid,
                F && f)
          : continuation(std::move(gid)), f_(std::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(F && f)
          : f_(std::forward<F>(f))
        {}

        virtual ~typed_continuation()
        {
            init_registration<typed_continuation>::g.register_continuation();
        }

        void trigger() const
        {
            LLCO_(info)
                << "typed_continuation<void>::trigger("
                << this->get_gid() << ")";
            if (f_.empty()) {
                if (!this->get_gid()) {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "typed_continuation<void>::trigger",
                        "attempt to trigger invalid LCO (the id is invalid)");
                    return;
                }
                trigger_lco_event(this->get_gid());
            }
            else {
                f_(this->get_gid());
            }
        }

        virtual void trigger_value(util::unused_type &&) const
        {
            this->trigger();
        }

    private:
        char const* get_continuation_name() const
        {
            return "hpx_void_typed_continuation";
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
    void continuation::trigger(Arg0 && arg0) const
    {
        // The static_cast is safe as we know that Arg0 is the result type
        // of the executed action (see apply.hpp).
        static_cast<typed_continuation<Arg0> const*>(this)->trigger_value(
            std::forward<Arg0>(arg0));
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
        typename util::decay<Cont>::type
    >
    make_continuation(Cont && cont)
    {
        typedef typename util::decay<Cont>::type cont_type;
        return hpx::actions::continuation_impl<cont_type>(
            std::forward<Cont>(cont), hpx::find_here());
    }

    template <typename Cont>
    inline hpx::actions::continuation_impl<
        typename util::decay<Cont>::type
    >
    make_continuation(Cont && f, hpx::id_type const& target)
    {
        typedef typename util::decay<Cont>::type cont_type;
        return hpx::actions::continuation_impl<cont_type>(
            std::forward<Cont>(f), target);
    }

    template <typename Cont, typename F>
    inline typename boost::disable_if<
        boost::is_same<
            typename util::decay<F>::type,
            hpx::naming::id_type
        >,
        hpx::actions::continuation2_impl<
            typename util::decay<Cont>::type,
            typename util::decay<F>::type
        >
    >::type
    make_continuation(Cont && cont, F && f)
    {
        typedef typename util::decay<Cont>::type cont_type;
        typedef typename util::decay<F>::type function_type;

        return hpx::actions::continuation2_impl<cont_type, function_type>(
            std::forward<Cont>(cont), hpx::find_here(), std::forward<F>(f));
    }

    template <typename Cont, typename F>
    inline hpx::actions::continuation2_impl<
        typename util::decay<Cont>::type,
        typename util::decay<F>::type
    >
    make_continuation(Cont && cont, hpx::id_type const& target,
        F && f)
    {
        typedef typename util::decay<Cont>::type cont_type;
        typedef typename util::decay<F>::type function_type;

        return hpx::actions::continuation2_impl<cont_type, function_type>(
            std::forward<Cont>(cont), target, std::forward<F>(f));
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
