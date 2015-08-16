//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_CONTINUATION_JUN_13_2008_1031AM)
#define HPX_RUNTIME_ACTIONS_CONTINUATION_JUN_13_2008_1031AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/trigger_lco.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>
#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/base_object.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/demangle_helper.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/traits/serialize_as_future.hpp>

#include <boost/enable_shared_from_this.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/utility/enable_if.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    //////////////////////////////////////////////////////////////////////////
    // forward declare the required overload of apply.
    template <typename Action, typename ...Ts>
    bool apply(naming::id_type const& gid, Ts&&... vs);

    template <
        typename Component, typename Signature, typename Derived,
        typename Cont, typename ...Ts>
    bool apply_continue(
        hpx::actions::basic_action<Component, Signature, Derived>,
        Cont&& cont, naming::id_type const& gid, Ts&&... vs);

    template <typename Component, typename Signature, typename Derived,
        typename ...Ts>
    inline bool
    apply_c(hpx::actions::basic_action<Component, Signature, Derived>,
        naming::id_type const& contgid, naming::id_type const& gid,
        Ts&&... vs);

    //////////////////////////////////////////////////////////////////////////
    // handling special case of triggering an LCO
    template <typename T>
    void set_lco_value(naming::id_type const& id, T && t, bool move_credits)
    {
        typedef typename lcos::base_lco_with_value<
            typename util::decay<T>::type
        >::set_value_action set_value_action;
        if (move_credits)
        {
            naming::id_type target(id.get_gid(), id_type::managed_move_credit);
            id.make_unmanaged();

            apply<set_value_action>(target, util::detail::make_temporary<T>(t));
        }
        else
        {
            apply<set_value_action>(id, util::detail::make_temporary<T>(t));
        }
    }

    template <typename T>
    void set_lco_value(naming::id_type const& id, T && t,
        naming::id_type const& cont, bool move_credits)
    {
        typedef typename lcos::base_lco_with_value<
            typename util::decay<T>::type
        >::set_value_action set_value_action;
        if (move_credits)
        {
            naming::id_type target(id.get_gid(), id_type::managed_move_credit);
            id.make_unmanaged();

            apply_c<set_value_action>(cont, target,
                util::detail::make_temporary<T>(t));
        }
        else
        {
            apply_c<set_value_action>(cont, id,
                util::detail::make_temporary<T>(t));
        }
    }

    HPX_API_EXPORT void set_lco_error(naming::id_type const& id,
        boost::exception_ptr const& e, bool move_credits);
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    class HPX_EXPORT continuation;

    namespace detail
    {
        template <typename Continuation>
        char const* get_continuation_name()
#ifndef HPX_HAVE_AUTOMATIC_SERIALIZATION_REGISTRATION
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
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & gid_;
        }
        HPX_SERIALIZATION_POLYMORPHIC_ABSTRACT(continuation);

#if defined(HPX_HAVE_COMPONENT_GET_GID_COMPATIBILITY)
        naming::id_type const& get_gid() const
        {
            return gid_;
        }
#endif

        naming::id_type const& get_id() const
        {
            return gid_;
        }

        virtual bool has_to_wait_for_futures() = 0;
        virtual void wait_for_futures() = 0;

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
    private:
        typedef typename util::decay<Cont>::type cont_type;

    public:
        template <typename T>
        struct result;

        template <typename F, typename T1, typename T2>
        struct result<F(T1, T2)>
        {
            typedef typename util::result_of<cont_type(T1, T2)>::type type;
        };

        continuation_impl() {}

        template <typename Cont_>
        continuation_impl(Cont_ && cont, hpx::id_type const& target)
          : cont_(std::forward<Cont_>(cont)), target_(target)
        {}

        virtual ~continuation_impl() {}

        template <typename T>
        typename result<continuation_impl(hpx::id_type, T)>::type
        operator()(hpx::id_type const& lco, T && t) const
        {
            hpx::apply_c(cont_, lco, target_, std::forward<T>(t));

            // Unfortunately we need to default construct the return value,
            // this possibly imposes an additional restriction of return types.
            typedef typename result<continuation_impl(hpx::id_type, T)>::type
                result_type;
            return result_type();
        }

        virtual bool has_to_wait_for_futures()
        {
            return traits::serialize_as_future<cont_type>::call_if(cont_);
        }

        virtual void wait_for_futures()
        {
            traits::serialize_as_future<cont_type>::call(cont_);
        }

    private:
        // serialization support
        friend class hpx::serialization::access;

        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive& ar, unsigned int const)
        {
            ar & cont_ & target_;
        }

        cont_type cont_;
        hpx::id_type target_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Cont, typename F>
    struct continuation2_impl
    {
    private:
        typedef typename util::decay<Cont>::type cont_type;
        typedef typename util::decay<F>::type function_type;

    public:
        template <typename T>
        struct result;

        template <typename This, typename T1, typename T2>
        struct result<This(T1, T2)>
        {
            typedef typename util::result_of<
                    cont_type(T1, T2)
                >::type result_type;
            typedef typename util::result_of<
                    function_type(hpx::id_type, result_type)
                >::type type;
        };

        continuation2_impl() {}

        template <typename Cont_, typename F_>
        continuation2_impl(Cont_ && cont, hpx::id_type const& target,
                F_ && f)
          : cont_(std::forward<Cont_>(cont)),
            target_(target),
            f_(std::forward<F_>(f))
        {}

        virtual ~continuation2_impl() {}

        template <typename T>
        typename result<continuation2_impl(hpx::id_type, T)>::type
        operator()(hpx::id_type const& lco, T && t) const
        {
            using hpx::util::placeholders::_2;
            hpx::apply_continue(cont_, hpx::util::bind(f_, lco, _2),
                target_, std::forward<T>(t));

            // Unfortunately we need to default construct the return value,
            // this possibly imposes an additional restriction of return types.
            typedef typename result<continuation2_impl(hpx::id_type, T)>::type
                result_type;
            return result_type();
        }

        virtual bool has_to_wait_for_futures()
        {
            return traits::serialize_as_future<cont_type>::call_if(cont_) ||
                traits::serialize_as_future<function_type>::call_if(f_);
        }

        virtual void wait_for_futures()
        {
            traits::serialize_as_future<cont_type>::call(cont_);
            traits::serialize_as_future<function_type>::call(f_);
        }

    private:
        // serialization support
        friend class hpx::serialization::access;

        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive& ar, unsigned int const)
        {
            ar & cont_ & target_ & f_;
        }

        cont_type cont_;        // continuation type
        hpx::id_type target_;
        function_type f_;       // set_value action  (default: set_lco_value_continuation)
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct typed_continuation : continuation
    {
    private:
        typedef util::function<void(naming::id_type, Result)> function_type;

    public:
        typed_continuation()
        {}

        explicit typed_continuation(naming::id_type const& gid)
          : continuation(gid)
        {}

        explicit typed_continuation(naming::id_type && gid)
          : continuation(std::move(gid))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type const& gid, F && f)
          : continuation(gid), f_(std::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type && gid, F && f)
          : continuation(std::move(gid)), f_(std::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(F && f)
          : f_(std::forward<F>(f))
        {}

        virtual void trigger_value(Result && result) const
        {
            LLCO_(info)
                << "typed_continuation<Result>::trigger_value("
                << this->get_id() << ")";

            if (f_.empty()) {
                if (!this->get_id()) {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "typed_continuation<Result>::trigger_value",
                        "attempt to trigger invalid LCO (the id is invalid)");
                    return;
                }
                hpx::set_lco_value(this->get_id(), std::move(result));
            }
            else {
                f_(this->get_id(), std::move(result));
            }
        }

        virtual bool has_to_wait_for_futures()
        {
            return traits::serialize_as_future<function_type>::call_if(f_);
        }

        virtual void wait_for_futures()
        {
            traits::serialize_as_future<function_type>::call(f_);
        }

    private:
        char const* get_continuation_name() const
        {
            return detail::get_continuation_name<typed_continuation>();
        }

        /// serialization support
        friend class hpx::serialization::access;

        void serialize(serialization::input_archive & ar)
        {
            // serialize function
            bool have_function = false;
            ar >> have_function;
            if (have_function)
                ar >> f_;
        }

        void serialize(serialization::output_archive & ar)
        {
            // serialize function
            bool have_function = !f_.empty();
            ar << have_function;
            if (have_function)
                ar << f_;
        }
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            // serialize base class
            ar & hpx::serialization::base_object<continuation>(*this);

            serialize(ar);
        }
        HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME(
            typed_continuation
          , detail::get_continuation_name<typed_continuation>()
        );

        function_type f_;
    };
}}

namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    template <>
    struct typed_continuation<void> : continuation
    {
    private:
        typedef util::function<void(naming::id_type)> function_type;

    public:
        typed_continuation()
        {}

        explicit typed_continuation(naming::id_type const& gid)
          : continuation(gid)
        {}

        explicit typed_continuation(naming::id_type && gid)
          : continuation(std::move(gid))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type const& gid, F && f)
          : continuation(gid), f_(std::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type && gid, F && f)
          : continuation(std::move(gid)), f_(std::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(F && f)
          : f_(std::forward<F>(f))
        {}

        void trigger() const
        {
            LLCO_(info)
                << "typed_continuation<void>::trigger("
                << this->get_id() << ")";

            if (f_.empty()) {
                if (!this->get_id()) {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "typed_continuation<void>::trigger",
                        "attempt to trigger invalid LCO (the id is invalid)");
                    return;
                }
                trigger_lco_event(this->get_id());
            }
            else {
                f_(this->get_id());
            }
        }

        virtual void trigger_value(util::unused_type &&) const
        {
            this->trigger();
        }

        virtual bool has_to_wait_for_futures()
        {
            return traits::serialize_as_future<function_type>::call_if(f_);
        }

        virtual void wait_for_futures()
        {
            traits::serialize_as_future<function_type>::call(f_);
        }

    private:
        char const* get_continuation_name() const
        {
            return "hpx_void_typed_continuation";
        }

        /// serialization support
        friend class hpx::serialization::access;

        void serialize(serialization::input_archive & ar)
        {
            // serialize function
            bool have_function = false;
            ar >> have_function;
            if (have_function)
                ar >> f_;
        }

        void serialize(serialization::output_archive & ar)
        {
            // serialize function
            bool have_function = !f_.empty();
            ar << have_function;
            if (have_function)
                ar << f_;
        }
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            // serialize base class
            ar & hpx::serialization::base_object<continuation>(*this);

            serialize(ar);
        }
        HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME(
            typed_continuation
          , "hpx_void_typed_continuation"
        );

        function_type f_;
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
    HPX_DEFINE_GET_CONTINUATION_NAME_(                                        \
        hpx::actions::typed_continuation<Result>, Name)                       \
/**/

#include <hpx/config/warnings_suffix.hpp>

#endif
