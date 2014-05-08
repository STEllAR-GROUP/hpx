//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file action_support.hpp

#if !defined(HPX_RUNTIME_ACTIONS_ACTION_SUPPORT_NOV_14_2008_0711PM)
#define HPX_RUNTIME_ACTIONS_ACTION_SUPPORT_NOV_14_2008_0711PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/config/bind.hpp>
#include <hpx/config/tuple.hpp>
#include <hpx/config/function.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/void_guard.hpp>
#include <hpx/traits/action_priority.hpp>
#include <hpx/traits/action_stacksize.hpp>
#include <hpx/traits/action_serialization_filter.hpp>
#include <hpx/traits/action_message_handler.hpp>
#include <hpx/traits/action_may_require_id_splitting.hpp>
#include <hpx/traits/action_does_termination_detection.hpp>
#include <hpx/traits/action_is_target_valid.hpp>
#include <hpx/traits/action_decorate_function.hpp>
#include <hpx/traits/action_decorate_continuation.hpp>
#include <hpx/traits/action_schedule_thread.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/type_size.hpp>
#include <hpx/runtime/get_lva.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/polymorphic_factory.hpp>
#include <hpx/util/serialize_sequence.hpp>
#include <hpx/util/serialize_exception.hpp>
#include <hpx/util/demangle_helper.hpp>
#include <hpx/util/register_locks.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/detail/count_num_args.hpp>
#include <hpx/util/static.hpp>
#include <hpx/lcos/async_fwd.hpp>
#include <hpx/lcos/future.hpp>

#if defined(HPX_HAVE_SECURITY)
#include <hpx/traits/action_capability_provider.hpp>
#endif

#include <boost/version.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/size.hpp>
#include <boost/fusion/include/transform_view.hpp>
#include <boost/ref.hpp>
#include <boost/foreach.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/is_bitwise_serializable.hpp>
#include <boost/serialization/array.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/identity.hpp>
#if defined(BOOST_NO_CXX11_DECLTYPE)
#  include <boost/typeof/typeof.hpp>
#endif
#include <boost/utility/enable_if.hpp>
#include <boost/preprocessor/cat.hpp>

#include <hpx/config/warnings_prefix.hpp>

/// \cond NOINTERNAL
namespace hpx { namespace actions { namespace detail
{
    struct action_serialization_data
    {
        boost::uint64_t parent_id_;
        boost::uint64_t parent_phase_;
        boost::uint32_t parent_locality_;
        boost::uint16_t priority_;
        boost::uint16_t stacksize_;
    };
}}}

namespace boost { namespace serialization
{
    template <>
    struct is_bitwise_serializable<
            hpx::actions::detail::action_serialization_data>
       : boost::mpl::true_
    {};
}}

/// \endcond

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    /// \cond NOINTERNAL

    struct base_action;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Action>
        char const* get_action_name()
#ifdef HPX_DISABLE_AUTOMATIC_SERIALIZATION_REGISTRATION
        ;
#else
        {
            /// If you encounter this assert while compiling code, that means that
            /// you have a HPX_REGISTER_ACTION macro somewhere in a source file,
            /// but the header in which the action is defined misses a
            /// HPX_REGISTER_ACTION_DECLARATION
            BOOST_MPL_ASSERT_MSG(
                traits::needs_automatic_registration<Action>::value
              , HPX_REGISTER_ACTION_DECLARATION_MISSING
              , (Action)
            );
            return util::type_id<Action>::typeid_.type_id();
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        // If an action returns a future, we need to do special things
        template <typename Result>
        struct remote_action_result
        {
            typedef Result type;
        };

        template <typename Result>
        struct remote_action_result<lcos::future<Result> >
        {
            typedef Result type;
        };

        template <>
        struct remote_action_result<lcos::future<void> >
        {
            typedef hpx::util::unused_type type;
        };

        template <typename Result>
        struct remote_action_result<lcos::shared_future<Result> >
        {
            typedef Result type;
        };

        template <>
        struct remote_action_result<lcos::shared_future<void> >
        {
            typedef hpx::util::unused_type type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Action>
        struct action_registration
        {
            static boost::shared_ptr<base_action> create()
            {
                return boost::shared_ptr<base_action>(new Action());
            }

            action_registration()
            {
                util::polymorphic_factory<base_action>::get_instance().
                    add_factory_function(
                        detail::get_action_name<typename Action::derived_type>()
                      , &action_registration::create
                    );
            }
        };

        template <typename Action, typename Enable =
            typename traits::needs_automatic_registration<Action>::type>
        struct automatic_action_registration
        {
            automatic_action_registration()
            {
                action_registration<Action> auto_register;
            }

            automatic_action_registration & register_action()
            {
                return *this;
            }
        };

        template <typename Action>
        struct automatic_action_registration<Action, boost::mpl::false_>
        {
            automatic_action_registration()
            {
            }

            automatic_action_registration & register_action()
            {
                return *this;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Future>
        struct serializable_ready_future_wrapper
        {
            typedef typename util::decay<Future>::type future_type;

            explicit serializable_ready_future_wrapper(Future& future)
              : future_(future)
            {}

            // serialization support
            template <typename Archive>
            void load(Archive& ar, unsigned)
            {
                using traits::future_access;
                future_access<future_type>::load(ar, future_);
            }

            template <typename Archive>
            void save(Archive& ar, unsigned) const
            {
                using traits::future_access;
                future_access<future_type>::save(ar, future_);
            }

            BOOST_SERIALIZATION_SPLIT_MEMBER();

            Future& future_;
        };

        struct serializable_arguments
        {
            template <typename, typename Enable = void>
            struct result;

            template <typename This, typename T>
            struct result<This(T&), typename boost::disable_if<
                traits::is_future<typename util::decay<T>::type> >::type>
            {
                typedef T& type;
            };

            template <typename This, typename T>
            struct result<This(T&), typename boost::enable_if<
                traits::is_future<typename util::decay<T>::type> >::type>
            {
                typedef serializable_ready_future_wrapper<T> type;
            };

            template <typename T>
            BOOST_FORCEINLINE typename boost::lazy_disable_if<
                traits::is_future<typename util::decay<T>::type>,
                result<serializable_arguments(T&)>
            >::type operator()(T& v) const
            {
                return v;
            }

            template <typename T>
            BOOST_FORCEINLINE typename boost::lazy_enable_if<
                traits::is_future<typename util::decay<T>::type>,
                result<serializable_arguments(T&)>
            >::type operator()(T& v) const
            {
                return serializable_ready_future_wrapper<T>(v);
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The \a base_action class is an abstract class used as the base class
    /// for all action types. It's main purpose is to allow polymorphic
    /// serialization of action instances through a shared_ptr.
    struct base_action
    {
        /// The type of an action defines whether this action will be executed
        /// directly or by a HPX-threads
        enum action_type
        {
            plain_action = 0,   ///< The action will be executed by a newly created thread
            direct_action = 1   ///< The action needs to be executed directly
        };

        /// Destructor
        virtual ~base_action() {}

        /// The function \a get_component_type returns the \a component_type
        /// of the component this action belongs to.
        virtual int get_component_type() const = 0;

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        virtual char const* get_action_name() const = 0;

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        virtual action_type get_action_type() const = 0;

        /// The \a get_thread_function constructs a proper thread function for
        /// a \a thread, encapsulating the functionality and the arguments
        /// of the action it is called for.
        ///
        /// \param lva    [in] This is the local virtual address of the
        ///               component the action has to be invoked on.
        ///
        /// \returns      This function returns a proper thread function usable
        ///               for a \a thread.
        ///
        /// \note This \a get_thread_function will be invoked to retrieve the
        ///       thread function for an action which has to be invoked without
        ///       continuations.
        virtual HPX_STD_FUNCTION<threads::thread_function_type>
            get_thread_function(naming::address::address_type lva) = 0;

        /// The \a get_thread_function constructs a proper thread function for
        /// a \a thread, encapsulating the functionality, the arguments, and
        /// the continuations of the action it is called for.
        ///
        /// \param cont   [in] This is the list of continuations to be
        ///               triggered after the execution of the action
        /// \param lva    [in] This is the local virtual address of the
        ///               component the action has to be invoked on.
        ///
        /// \returns      This function returns a proper thread function usable
        ///               for a \a thread.
        ///
        /// \note This \a get_thread_function will be invoked to retrieve the
        ///       thread function for an action which has to be invoked with
        ///       continuations.
        virtual HPX_STD_FUNCTION<threads::thread_function_type>
            get_thread_function(continuation_type& cont,
                naming::address::address_type lva) = 0;

        /// return the id of the locality of the parent thread
        virtual boost::uint32_t get_parent_locality_id() const = 0;

        /// Return the thread id of the parent thread
        virtual threads::thread_id_repr_type get_parent_thread_id() const = 0;

        /// Return the thread phase of the parent thread
        virtual boost::uint64_t get_parent_thread_phase() const = 0;

        /// Return the thread priority this action has to be executed with
        virtual threads::thread_priority get_thread_priority() const = 0;

        /// Return the thread stacksize this action has to be executed with
        virtual threads::thread_stacksize get_thread_stacksize() const = 0;

        /// Return the size of action arguments in bytes
        virtual std::size_t get_type_size() const = 0;

        /// Return whether the embedded action may require id-splitting
        virtual bool may_require_id_splitting() const = 0;

        /// Return whether the embedded action is part of termination detection
        virtual bool does_termination_detection() const = 0;

        virtual void load(hpx::util::portable_binary_iarchive & ar) = 0;
        virtual void save(hpx::util::portable_binary_oarchive & ar) const = 0;

//         /// Return all data needed for thread initialization
//         virtual threads::thread_init_data&
//         get_thread_init_data(naming::id_type const& target,
//             naming::address::address_type lva, threads::thread_init_data& data) = 0;
//
//         virtual threads::thread_init_data&
//         get_thread_init_data(continuation_type& cont,
//             naming::id_type const& target, naming::address::address_type lva,
//             threads::thread_init_data& data) = 0;

        /// Return all data needed for thread initialization
        virtual void schedule_thread(naming::id_type const& target,
            naming::address::address_type lva,
            threads::thread_state_enum initial_state) = 0;

        virtual void schedule_thread(continuation_type& cont,
            naming::id_type const& target, naming::address::address_type lva,
            threads::thread_state_enum initial_state) = 0;

        /// Return a pointer to the filter to be used while serializing an
        /// instance of this action type.
        virtual util::binary_filter* get_serialization_filter(
            parcelset::parcel const& p) const = 0;

        /// Return a pointer to the message handler to be used for this action.
        virtual parcelset::policies::message_handler* get_message_handler(
            parcelset::parcelhandler* ph, naming::locality const& loc,
            parcelset::connection_type t, parcelset::parcel const& p) const = 0;

#if defined(HPX_HAVE_SECURITY)
        /// Return the set of capabilities required to invoke this action
        virtual components::security::capability get_required_capabilities(
            naming::address::address_type lva) const = 0;
#endif
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // Figure out what priority the action has to be be associated with
        // A dynamically specified default priority results in using the static
        // Priority.
        template <threads::thread_priority Priority>
        struct thread_priority
        {
            static threads::thread_priority
            call(threads::thread_priority priority)
            {
                if (priority == threads::thread_priority_default)
                    return Priority;
                return priority;
            }
        };

        // If the static Priority is default, a dynamically specified default
        // priority results in using the normal priority.
        template <>
        struct thread_priority<threads::thread_priority_default>
        {
            static threads::thread_priority
            call(threads::thread_priority priority)
            {
                if (priority == threads::thread_priority_default)
                    return threads::thread_priority_normal;
                return priority;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // Figure out what stacksize the action has to be be associated with
        // A dynamically specified default stacksize results in using the static
        // Stacksize.
        template <threads::thread_stacksize Stacksize>
        struct thread_stacksize
        {
            static threads::thread_stacksize
            call(threads::thread_stacksize stacksize)
            {
                if (stacksize == threads::thread_stacksize_default)
                    return Stacksize;
                return stacksize;
            }
        };

        // If the static Stacksize is default, a dynamically specified default
        // stacksize results in using the normal stacksize.
        template <>
        struct thread_stacksize<threads::thread_stacksize_default>
        {
            static threads::thread_stacksize
            call(threads::thread_stacksize stacksize)
            {
                if (stacksize == threads::thread_stacksize_default)
                    return threads::thread_stacksize_minimal;
                return stacksize;
            }
        };
    }

    template <typename Action>
    struct init_registration;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct transfer_action : base_action
    {
        typedef typename Action::component_type component_type;
        typedef typename Action::derived_type derived_type;
        typedef typename Action::result_type result_type;
        typedef typename Action::arguments_type arguments_type;

        // This is the priority value this action has been instantiated with
        // (statically). This value might be different from the priority member
        // holding the runtime value an action has been created with
        enum { priority_value = traits::action_priority<Action>::value };

        // This is the stacksize value this action has been instantiated with
        // (statically). This value might be different from the stacksize member
        // holding the runtime value an action has been created with
        enum { stacksize_value = traits::action_stacksize<Action>::value };

        typedef typename Action::direct_execution direct_execution;

        // default constructor is needed for serialization
        transfer_action() {}

        // construct an action from its arguments
        template <typename Args>
        explicit transfer_action(Args && args)
          : arguments_(std::forward<Args>(args)),
#if HPX_THREAD_MAINTAIN_PARENT_REFERENCE
            parent_locality_(transfer_action::get_locality_id()),
            parent_id_(reinterpret_cast<boost::uint64_t>(threads::get_parent_id())),
            parent_phase_(threads::get_parent_phase()),
#endif
            priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(threads::thread_priority_default)),
            stacksize_(
                detail::thread_stacksize<
                    static_cast<threads::thread_stacksize>(stacksize_value)
                >::call(threads::thread_stacksize_default))
        {}

        template <typename Args>
        transfer_action(threads::thread_priority priority, Args && args)
          : arguments_(std::forward<Args>(args)),
#if HPX_THREAD_MAINTAIN_PARENT_REFERENCE
            parent_locality_(transfer_action::get_locality_id()),
            parent_id_(reinterpret_cast<boost::uint64_t>(threads::get_parent_id())),
            parent_phase_(threads::get_parent_phase()),
#endif
            priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(priority_value)
                >::call(priority)),
            stacksize_(
                detail::thread_stacksize<
                    static_cast<threads::thread_stacksize>(stacksize_value)
                >::call(threads::thread_stacksize_default))
        {}

        //
        ~transfer_action()
        {
            init_registration<transfer_action<Action> >::g.register_action();
        }

    public:
        /// retrieve component type
        static int get_static_component_type()
        {
            return derived_type::get_component_type();
        }

    private:
        /// The function \a get_component_type returns the \a component_type
        /// of the component this action belongs to.
        int get_component_type() const
        {
            return derived_type::get_component_type();
        }

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* get_action_name() const
        {
            return detail::get_action_name<derived_type>();
        }

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        action_type get_action_type() const
        {
            return derived_type::get_action_type();
        }

        /// The \a get_thread_function constructs a proper thread function for
        /// a \a thread, encapsulating the functionality and the arguments
        /// of the action it is called for.
        ///
        /// \param lva    [in] This is the local virtual address of the
        ///               component the action has to be invoked on.
        ///
        /// \returns      This function returns a proper thread function usable
        ///               for a \a thread.
        ///
        /// \note This \a get_thread_function will be invoked to retrieve the
        ///       thread function for an action which has to be invoked without
        ///       continuations.
        HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(naming::address::address_type lva)
        {
            return derived_type::construct_thread_function(lva,
                std::move(arguments_));
        }

        /// The \a get_thread_function constructs a proper thread function for
        /// a \a thread, encapsulating the functionality, the arguments, and
        /// the continuations of the action it is called for.
        ///
        /// \param cont   [in] This is the list of continuations to be
        ///               triggered after the execution of the action
        /// \param lva    [in] This is the local virtual address of the
        ///               component the action has to be invoked on.
        ///
        /// \returns      This function returns a proper thread function usable
        ///               for a \a thread.
        ///
        /// \note This \a get_thread_function will be invoked to retrieve the
        ///       thread function for an action which has to be invoked with
        ///       continuations.
        HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(continuation_type& cont,
            naming::address::address_type lva)
        {
            return derived_type::construct_thread_function(cont, lva,
                std::move(arguments_));
        }

#if !HPX_THREAD_MAINTAIN_PARENT_REFERENCE
        /// Return the locality of the parent thread
        boost::uint32_t get_parent_locality_id() const
        {
            return naming::invalid_locality_id;
        }

        /// Return the thread id of the parent thread
        threads::thread_id_repr_type get_parent_thread_id() const
        {
            return threads::invalid_thread_id_repr;
        }

        /// Return the phase of the parent thread
        boost::uint64_t get_parent_thread_phase() const
        {
            return 0;
        }
#else
        /// Return the locality of the parent thread
        boost::uint32_t get_parent_locality_id() const
        {
            return parent_locality_;
        }

        /// Return the thread id of the parent thread
        threads::thread_id_repr_type get_parent_thread_id() const
        {
            return reinterpret_cast<threads::thread_id_repr_type>(parent_id_);
        }

        /// Return the phase of the parent thread
        boost::uint64_t get_parent_thread_phase() const
        {
            return parent_phase_;
        }
#endif

        /// Return the thread priority this action has to be executed with
        threads::thread_priority get_thread_priority() const
        {
            return priority_;
        }

        /// Return the thread stacksize this action has to be executed with
        threads::thread_stacksize get_thread_stacksize() const
        {
            return stacksize_;
        }

        /// Return the size of action arguments in bytes
        std::size_t get_type_size() const
        {
            return traits::type_size<arguments_type>::call(arguments_);
        }

        /// Return whether the embedded action may require id-splitting
        bool may_require_id_splitting() const
        {
            return traits::action_may_require_id_splitting<derived_type>::call(arguments_);
        }

        /// Return whether the embedded action is part of termination detection
        bool does_termination_detection() const
        {
            return traits::action_does_termination_detection<derived_type>::call();
        }

        /// Return all data needed for thread initialization
        threads::thread_init_data&
        get_thread_init_data(naming::id_type const& target,
            naming::address::address_type lva, threads::thread_init_data& data)
        {
            data.func = get_thread_function(lva);
#if HPX_THREAD_MAINTAIN_TARGET_ADDRESS
            data.lva = lva;
#endif
#if HPX_THREAD_MAINTAIN_DESCRIPTION
            data.description = detail::get_action_name<derived_type>();
#endif
#if HPX_THREAD_MAINTAIN_PARENT_REFERENCE
            data.parent_id = reinterpret_cast<threads::thread_id_repr_type>(parent_id_);
            data.parent_locality_id = parent_locality_;
#endif
            data.priority = priority_;
            data.stacksize = threads::get_stack_size(stacksize_);

            data.target = target;
            return data;
        }

        threads::thread_init_data&
        get_thread_init_data(continuation_type& cont, naming::id_type const& target,
            naming::address::address_type lva, threads::thread_init_data& data)
        {
            data.func = get_thread_function(cont, lva);
#if HPX_THREAD_MAINTAIN_TARGET_ADDRESS
            data.lva = lva;
#endif
#if HPX_THREAD_MAINTAIN_DESCRIPTION
            data.description = detail::get_action_name<derived_type>();
#endif
#if HPX_THREAD_MAINTAIN_PARENT_REFERENCE
            data.parent_id = reinterpret_cast<threads::thread_id_repr_type>(parent_id_);
            data.parent_locality_id = parent_locality_;
#endif
            data.priority = priority_;
            data.stacksize = threads::get_stack_size(stacksize_);

            data.target = target;
            return data;
        }

        // schedule a new thread
        void schedule_thread(naming::id_type const& target,
            naming::address::address_type lva,
            threads::thread_state_enum initial_state)
        {
            continuation_type cont;
            threads::thread_init_data data;
            if (traits::action_decorate_continuation<derived_type>::call(cont))
            {
                traits::action_schedule_thread<derived_type>::call(lva,
                    get_thread_init_data(cont, target, lva, data), initial_state);
            }
            else
            {
                traits::action_schedule_thread<derived_type>::call(lva,
                    get_thread_init_data(target, lva, data), initial_state);
            }
        }

        void schedule_thread(continuation_type& cont,
            naming::id_type const& target, naming::address::address_type lva,
            threads::thread_state_enum initial_state)
        {
            // first decorate the continuation
            traits::action_decorate_continuation<derived_type>::call(cont);

            // now, schedule the thread
            threads::thread_init_data data;
            traits::action_schedule_thread<derived_type>::call(lva,
                get_thread_init_data(cont, target, lva, data), initial_state);
        }

        /// Return a pointer to the filter to be used while serializing an
        /// instance of this action type.
        util::binary_filter* get_serialization_filter(
            parcelset::parcel const& p) const
        {
            return traits::action_serialization_filter<derived_type>::call(p);
        }

        /// Return a pointer to the message handler to be used for this action.
        parcelset::policies::message_handler* get_message_handler(
            parcelset::parcelhandler* ph, naming::locality const& loc,
            parcelset::connection_type t, parcelset::parcel const& p) const
        {
            return traits::action_message_handler<derived_type>::
                call(ph, loc, t, p);
        }

#if defined(HPX_HAVE_SECURITY)
        /// Return the set of capabilities required to invoke this action
        components::security::capability get_required_capabilities(
            naming::address::address_type lva) const
        {
            return traits::action_capability_provider<derived_type>::call(lva);
        }
#endif
    public:
        /// retrieve the N's argument
        template <int N>
        typename boost::fusion::result_of::at_c<arguments_type, N>::type
        get()
        {
            return boost::fusion::at_c<N>(arguments_);
        }

        // serialization support
        void load(hpx::util::portable_binary_iarchive & ar)
        {
            boost::fusion::transform_view<
                arguments_type, detail::serializable_arguments
            > serializable_arguments(arguments_, detail::serializable_arguments());
            util::serialize_sequence(ar, serializable_arguments);

            // Always serialize the parent information to maintain binary
            // compatibility on the wire.

            if (ar.flags() & util::disable_array_optimization) {
#if !HPX_THREAD_MAINTAIN_PARENT_REFERENCE
                boost::uint32_t parent_locality_ = naming::invalid_locality_id;
                boost::uint64_t parent_id_ = boost::uint64_t(-1);
                boost::uint64_t parent_phase_ = 0;
#endif
                ar >> parent_locality_;
                ar >> parent_id_;
                ar >> parent_phase_;

                ar >> priority_;
                ar >> stacksize_;
            }
            else {
                detail::action_serialization_data data;
                ar.load(data);

#if HPX_THREAD_MAINTAIN_PARENT_REFERENCE
                parent_id_ = data.parent_id_;
                parent_phase_ = data.parent_phase_;
                parent_locality_ = data.parent_locality_;
#endif
                priority_ = static_cast<threads::thread_priority>(data.priority_);
                stacksize_ = static_cast<threads::thread_stacksize>(data.stacksize_);
            }
        }

        void save(hpx::util::portable_binary_oarchive & ar) const
        {
            boost::fusion::transform_view<
                arguments_type const, detail::serializable_arguments
            > serializable_arguments(arguments_, detail::serializable_arguments());
            util::serialize_sequence(ar, serializable_arguments);

            // Always serialize the parent information to maintain binary
            // compatibility on the wire.

#if !HPX_THREAD_MAINTAIN_PARENT_REFERENCE
            boost::uint32_t parent_locality_ = naming::invalid_locality_id;
            boost::uint64_t parent_id_ = boost::uint64_t(-1);
            boost::uint64_t parent_phase_ = 0;
#endif
            if (ar.flags() & util::disable_array_optimization) {
                ar << parent_locality_;
                ar << parent_id_;
                ar << parent_phase_;

                ar << priority_;
                ar << stacksize_;
            }
            else {
                detail::action_serialization_data data;
                data.parent_id_ = parent_id_;
                data.parent_phase_ = parent_phase_;
                data.parent_locality_ = parent_locality_;
                data.priority_ = priority_;
                data.stacksize_ = stacksize_;

                ar.save(data);
            }
        }

    private:
        static boost::uint32_t get_locality_id()
        {
            error_code ec(lightweight);      // ignore any errors
            return hpx::get_locality_id(ec);
        }

    protected:
        arguments_type arguments_;

#if HPX_THREAD_MAINTAIN_PARENT_REFERENCE
        boost::uint32_t parent_locality_;
        boost::uint64_t parent_id_;
        boost::uint64_t parent_phase_;
#endif
        threads::thread_priority priority_;
        threads::thread_stacksize stacksize_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <int N, typename Action>
    inline typename boost::fusion::result_of::at_c<
        typename transfer_action<Action>::arguments_type, N
    >::type
    get(transfer_action<Action> & args)
    {
        return args.template get<N>();
    }

    // bring in all overloads for
    //    construct_continuation_thread_functionN()
    //    construct_continuation_thread_function_voidN()
    #include <hpx/runtime/actions/construct_continuation_function_objects.hpp>

    ///////////////////////////////////////////////////////////////////////////
    /// \tparam Component         component type
    /// \tparam Result            return type
    /// \tparam Arguments         arguments (tuple)
    /// \tparam Derived           derived action class
    template <typename Component, typename Result,
        typename Arguments, typename Derived>
    struct action
    {
        typedef Component component_type;
        typedef Derived derived_type;
        typedef Arguments arguments_type;

        typedef void action_tag;

        ///////////////////////////////////////////////////////////////////////
        static bool is_target_valid(naming::id_type const& id)
        {
            return true;        // by default we don't do any verification
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Func, typename Arguments_>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_continuation_thread_function_void(
            continuation_type cont, Func && func, Arguments_ && args)
        {
            typedef typename boost::remove_reference<Arguments_>::type arguments_type;
            return detail::construct_continuation_thread_function_voidN<
                    derived_type, util::tuple_size<arguments_type>::value
                >::call(cont, std::forward<Func>(func), std::forward<Arguments_>(args));
        }

        template <typename Func, typename Arguments_>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_continuation_thread_function(
            continuation_type cont, Func && func, Arguments_ && args)
        {
            typedef typename boost::remove_reference<Arguments_>::type arguments_type;
            return detail::construct_continuation_thread_functionN<
                    derived_type, util::tuple_size<arguments_type>::value
                >::call(cont, std::forward<Func>(func), std::forward<Arguments_>(args));
        }

        // bring in all overloads for
        //    construct_continuation_thread_function_void()
        //    construct_continuation_thread_object_function_void()
        //    construct_continuation_thread_function()
        //    construct_continuation_thread_object_function()
        #include <hpx/runtime/actions/construct_continuation_functions.hpp>

        typedef typename traits::promise_local_result<Result>::type local_result_type;
        typedef typename traits::is_future<local_result_type>::type is_future_pred;

        // bring in the definition for all overloads for operator()
        #include <hpx/runtime/actions/define_function_operators.hpp>

        /// retrieve component type
        static int get_component_type()
        {
            return static_cast<int>(components::get_component_type<Component>());
        }

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        static base_action::action_type get_action_type()
        {
            return base_action::plain_action;
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive& ar, const unsigned int) {}
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // simple type allowing to distinguish whether an action is the most
        // derived one
        struct this_type {};

        template <typename Action, typename Derived>
        struct action_type
          : boost::mpl::if_<boost::is_same<Derived, this_type>, Action, Derived>
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    // Base template allowing to generate a concrete action type from a function
    // pointer. It is instantiated only if the supplied pointer is not a
    // supported function pointer.
    template <typename F, F funcptr, typename Derived = detail::this_type,
        typename Direct = boost::mpl::false_>
    struct make_action;

    template <typename F, F funcptr, typename Derived = detail::this_type>
    struct make_direct_action
      : make_action<F, funcptr, Derived, boost::mpl::true_>
    {};

// older compilers require BOOST_TYPEOF, newer compilers have decltype()
#if defined(HPX_HAVE_CXX11_DECLTYPE)
#  define HPX_TYPEOF(x)       decltype(x)
#  define HPX_TYPEOF_TPL(x)   decltype(x)
#elif defined(HPX_GCC_VERSION) && (HPX_GCC_VERSION <= 40400)
#  define HPX_TYPEOF(x)       __typeof__(x)
#  define HPX_TYPEOF_TPL(x)   __typeof__(x)
#else
#  define HPX_TYPEOF(x)       BOOST_TYPEOF(x)
#  define HPX_TYPEOF_TPL(x)   BOOST_TYPEOF_TPL(x)
#endif

    // Macros usable to refer to an action given the function to expose
    #define HPX_MAKE_ACTION(f)                                                \
        hpx::actions::make_action<HPX_TYPEOF(&f), &f>        /**/             \
    /**/
    #define HPX_MAKE_DIRECT_ACTION(f)                                         \
        hpx::actions::make_direct_action<HPX_TYPEOF(&f), &f> /**/             \
    /**/

    #define HPX_MAKE_ACTION_TPL(f)                                            \
        hpx::actions::make_action<HPX_TYPEOF_TPL(&f), &f>        /**/         \
    /**/
    #define HPX_MAKE_DIRECT_ACTION_TPL(f)                                     \
        hpx::actions::make_direct_action<HPX_TYPEOF_TPL(&f), &f> /**/         \
    /**/

#if BOOST_WORKAROUND(BOOST_MSVC, == 1600)
    // workarounds for VC2010
    #define HPX_MAKE_COMPONENT_ACTION(component, f)                           \
        hpx::actions::make_action<                                            \
            HPX_TYPEOF(component::f) component::*, &component::f>  /**/       \
    /**/
    #define HPX_MAKE_DIRECT_COMPONENT_ACTION(component, f)                    \
        hpx::actions::make_direct_action<                                     \
            HPX_TYPEOF(component::f) component::*, &component::f>  /**/       \
    /**/

    #define HPX_MAKE_COMPONENT_ACTION_TPL(component, f)                       \
        hpx::actions::make_action<                                            \
            HPX_TYPEOF_TPL(component::f) component::*, &component::f>  /**/   \
    /**/
    #define HPX_MAKE_DIRECT_COMPONENT_ACTION_TPL(component, f)                \
        hpx::actions::make_direct_action<                                     \
            HPX_TYPEOF_TPL(component::f) component::*, &component::f>  /**/   \
    /**/

    namespace detail
    {
        template <typename Obj, typename F>
        struct synthesize_const_mf;

        template <typename F> F replicate_type(F);
    }

    #define HPX_MAKE_CONST_COMPONENT_ACTION(component, f)                     \
        hpx::actions::make_action<                                            \
            hpx::actions::detail::synthesize_const_mf<                        \
                component, HPX_TYPEOF(                                        \
                    hpx::actions::detail::replicate_type(&component::f)       \
                )                                                             \
            >::type, &component::f>  /**/                                     \
    /**/
    #define HPX_MAKE_CONST_DIRECT_COMPONENT_ACTION(component, f)              \
        hpx::actions::make_direct_action<                                     \
            hpx::actions::detail::synthesize_const_mf<                        \
                component, HPX_TYPEOF(                                        \
                    hpx::actions::detail::replicate_type(&component::f)       \
                )                                                             \
            >::type, &component::f>  /**/                                     \
    /**/

    #define HPX_MAKE_CONST_COMPONENT_ACTION_TPL(component, f)                 \
        hpx::actions::make_action<                                            \
            typename hpx::actions::detail::synthesize_const_mf<               \
                component, HPX_TYPEOF_TPL(                                    \
                    hpx::actions::detail::replicate_type(&component::f)       \
                )                                                             \
            >::type, &component::f>  /**/                                     \
    /**/
    #define HPX_MAKE_CONST_DIRECT_COMPONENT_ACTION_TPL(component, f)          \
        hpx::actions::make_direct_action<                                     \
            typename hpx::actions::detail::synthesize_const_mf<               \
                component, HPX_TYPEOF_TPL(                                    \
                    hpx::actions::detail::replicate_type(&component::f)       \
                )                                                             \
            >::type, &component::f>  /**/                                     \
    /**/
#else
    // the implementation on conforming compilers is almost trivial
    #define HPX_MAKE_COMPONENT_ACTION(component, f)                           \
        HPX_MAKE_ACTION(component::f)                                         \
    /**/
    #define HPX_MAKE_DIRECT_COMPONENT_ACTION(component, f)                    \
        HPX_MAKE_DIRECT_ACTION(component::f)                                  \
    /**/
    #define HPX_MAKE_CONST_COMPONENT_ACTION(component, f)                     \
        HPX_MAKE_ACTION(component::f)                                         \
    /**/
    #define HPX_MAKE_CONST_DIRECT_COMPONENT_ACTION(component, f)              \
        HPX_MAKE_DIRECT_ACTION(component::f)                                  \
    /**/

    #define HPX_MAKE_COMPONENT_ACTION_TPL(component, f)                       \
        HPX_MAKE_ACTION_TPL(component::f)                                     \
    /**/
    #define HPX_MAKE_DIRECT_COMPONENT_ACTION_TPL(component, f)                \
        HPX_MAKE_DIRECT_ACTION_TPL(component::f)                              \
    /**/
    #define HPX_MAKE_CONST_COMPONENT_ACTION_TPL(component, f)                 \
        HPX_MAKE_ACTION_TPL(component::f)                                     \
    /**/
    #define HPX_MAKE_CONST_DIRECT_COMPONENT_ACTION_TPL(component, f)          \
        HPX_MAKE_DIRECT_ACTION_TPL(component::f)                              \
    /**/
#endif

    /// \endcond
}}

/// \cond NOINTERNAL

#include <hpx/config/warnings_suffix.hpp>

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_BASE_HELPER(action, actionname)                          \
    hpx::actions::detail::register_base_helper<action>                        \
            BOOST_PP_CAT(                                                     \
                BOOST_PP_CAT(__hpx_action_register_base_helper_, __LINE__),   \
                _##actionname);                                               \
/**/

///////////////////////////////////////////////////////////////////////////////
// Helper macro for action serialization, each of the defined actions needs to
// be registered with the serialization library
#define HPX_DEFINE_GET_ACTION_NAME(action)                                    \
    HPX_DEFINE_GET_ACTION_NAME_(action, action)                               \
/**/
#define HPX_DEFINE_GET_ACTION_NAME_(action, actionname)                       \
    namespace hpx { namespace actions { namespace detail {                    \
        template<> HPX_ALWAYS_EXPORT                                          \
        char const* get_action_name<action>()                                 \
        {                                                                     \
            return BOOST_PP_STRINGIZE(actionname);                            \
        }                                                                     \
    }}}                                                                       \
/**/

#define HPX_ACTION_REGISTER_ACTION_FACTORY(Action, Name)                      \
    static ::hpx::actions::detail::action_registration<Action>                \
        const BOOST_PP_CAT(Name, _action_factory_registration) =              \
        ::hpx::actions::detail::action_registration<Action>();                \
/**/

#define HPX_REGISTER_ACTION_(...)                                             \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_ACTION_, HPX_UTIL_PP_NARG(__VA_ARGS__)                   \
    )(__VA_ARGS__))                                                           \
/**/
#define HPX_REGISTER_ACTION_1(action)                                         \
    HPX_REGISTER_ACTION_2(action, action)                                     \
/**/
#define HPX_REGISTER_ACTION_2(action, actionname)                             \
    HPX_ACTION_REGISTER_ACTION_FACTORY(hpx::actions::transfer_action<action>, \
        actionname)                                                           \
    HPX_DEFINE_GET_ACTION_NAME_(action, actionname)                           \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID1(action)              \
    namespace hpx { namespace actions { namespace detail {                    \
        template <> HPX_ALWAYS_EXPORT                                         \
        char const* get_action_name<action>();                                \
    }}}                                                                       \
/**/
#define HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID2(action)              \
    namespace hpx { namespace traits {                                        \
        template <>                                                           \
        struct needs_automatic_registration<action>                           \
          : boost::mpl::false_                                                \
        {};                                                                   \
    }}                                                                        \
/**/

#define HPX_REGISTER_ACTION_DECLARATION_(...)                                 \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_ACTION_DECLARATION_, HPX_UTIL_PP_NARG(__VA_ARGS__)       \
    )(__VA_ARGS__))                                                           \
/**/
#define HPX_REGISTER_ACTION_DECLARATION_1(action)                             \
    HPX_REGISTER_ACTION_DECLARATION_2(action, action)                         \
/**/
#define HPX_REGISTER_ACTION_DECLARATION_2(action, actionname)                 \
    HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID1(action)                  \
    HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID2(                         \
        hpx::actions::transfer_action<action>)                                \
/**/

namespace hpx { namespace actions
{
    template <typename Action>
    struct init_registration<transfer_action<Action> >
    {
        static detail::automatic_action_registration<transfer_action<Action> > g;
    };

    template <typename Action>
    detail::automatic_action_registration<transfer_action<Action> >
        init_registration<transfer_action<Action> >::g =
            detail::automatic_action_registration<transfer_action<Action> >();
}}

#if 0 //WIP
///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID1_TEMPLATE(            \
        TEMPLATE, TYPE)                                                       \
    namespace hpx { namespace actions { namespace detail {                    \
        HPX_UTIL_STRIP(TEMPLATE) HPX_ALWAYS_EXPORT                            \
        char const* get_action_name<HPX_UTIL_STRIP(TYPE)>();                  \
    }}}                                                                       \
/**/
#define HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID2_TEMPLATE(            \
        TEMPLATE, TYPE)                                                       \
    namespace hpx { namespace traits {                                        \
        HPX_UTIL_STRIP(TEMPLATE)                                              \
        struct needs_guid_initialization<HPX_UTIL_STRIP(TYPE)>                \
          : boost::mpl::false_                                                \
        {};                                                                   \
    }}                                                                        \
/**/
#define HPX_REGISTER_ACTION_DECLARATION_GUID_TEMPLATE(TEMPLATE, TYPE)         \
    namespace boost { namespace archive { namespace detail {                  \
        namespace extra_detail {                                              \
            HPX_UTIL_STRIP(TEMPLATE)                                          \
            struct init_guid<HPX_UTIL_STRIP(TYPE)>;                           \
        }                                                                     \
    }}}                                                                       \
/**/
#define HPX_REGISTER_ACTION_DECLARATION_TEMPLATE(TEMPLATE, TYPE)              \
    HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID1_TEMPLATE(                \
        TEMPLATE, HPX_UTIL_STRIP(TYPE))                                       \
    HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID2_TEMPLATE(                \
        TEMPLATE, hpx::actions::transfer_action<HPX_UTIL_STRIP(TYPE)>)        \
    HPX_SERIALIZATION_REGISTER_TEMPLATE(                                      \
        TEMPLATE, hpx::actions::transfer_action<HPX_UTIL_STRIP(TYPE)>)        \
    HPX_REGISTER_ACTION_DECLARATION_GUID_TEMPLATE(                            \
        TEMPLATE, hpx::actions::transfer_action<HPX_UTIL_STRIP(TYPE)>)        \
/**/
#endif

///////////////////////////////////////////////////////////////////////////////
#define HPX_ACTION_USES_STACK(action, size)                                   \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        template <>                                                           \
        struct action_stacksize<action>                                       \
        {                                                                     \
            enum { value = size };                                            \
        };                                                                    \
    }}                                                                        \
/**/

#define HPX_ACTION_USES_SMALL_STACK(action)                                   \
    HPX_ACTION_USES_STACK(action, threads::thread_stacksize_small)            \
/**/
#define HPX_ACTION_USES_MEDIUM_STACK(action)                                  \
    HPX_ACTION_USES_STACK(action, threads::thread_stacksize_medium)           \
/**/
#define HPX_ACTION_USES_LARGE_STACK(action)                                   \
    HPX_ACTION_USES_STACK(action, threads::thread_stacksize_large)            \
/**/
#define HPX_ACTION_USES_HUGE_STACK(action)                                    \
    HPX_ACTION_USES_STACK(action, threads::thread_stacksize_huge)             \
/**/
#define HPX_ACTION_DOES_NOT_SUSPEND(action)                                   \
    HPX_ACTION_USES_STACK(action, threads::thread_stacksize_nostack)          \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_ACTION_HAS_PRIORITY(action, priority)                             \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        template <>                                                           \
        struct action_priority<action>                                        \
        {                                                                     \
            enum { value = priority };                                        \
        };                                                                    \
    }}                                                                        \
/**/

#define HPX_ACTION_HAS_LOW_PRIORITY(action)                                   \
    HPX_ACTION_HAS_PRIORITY(action, threads::thread_priority_low)             \
/**/
#define HPX_ACTION_HAS_NORMAL_PRIORITY(action)                                \
    HPX_ACTION_HAS_PRIORITY(action, threads::thread_priority_normal)          \
/**/
#define HPX_ACTION_HAS_CRITICAL_PRIORITY(action)                              \
    HPX_ACTION_HAS_PRIORITY(action, threads::thread_priority_critical)        \
/**/

/// \endcond

/// \def HPX_REGISTER_ACTION_DECLARATION(action)
///
/// \brief Declare the necessary component action boilerplate code.
///
/// The macro \a HPX_REGISTER_ACTION_DECLARATION can be used to declare all the
/// boilerplate code which is required for proper functioning of component
/// actions in the context of HPX.
///
/// The parameter \a action is the type of the action to declare the
/// boilerplate for.
///
/// This macro can be invoked with an optional second parameter. This parameter
/// specifies a unique name of the action to be used for serialization purposes.
/// The second parameter has to be specified if the first parameter is not
/// usable as a plain (non-qualified) C++ identifier, i.e. the first parameter
/// contains special characters which cannot be part of a C++ identifier, such
/// as '<', '>', or ':'.
///
/// \par Example:
///
/// \code
///      namespace app
///      {
///          // Define a simple component exposing one action 'print_greating'
///          class HPX_COMPONENT_EXPORT server
///            : public hpx::components::simple_component_base<server>
///          {
///              void print_greating ()
///              {
///                  hpx::cout << "Hey, how are you?\n" << hpx::flush;
///              }
///
///              // Component actions need to be declared, this also defines the
///              // type 'print_greating_action' representing the action.
///              HPX_DEFINE_COMPONENT_ACTION(server, print_greating, print_greating_action);
///          };
///      }
///
///      // Declare boilerplate code required for each of the component actions.
///      HPX_REGISTER_ACTION_DECLARATION(app::server::print_greating_action);
/// \endcode
///
/// \note This macro has to be used once for each of the component actions
/// defined using one of the \a HPX_DEFINE_COMPONENT_ACTION macros. It has to
/// be visible in all translation units using the action, thus it is
/// recommended to place it into the header file defining the component.
#define HPX_REGISTER_ACTION_DECLARATION(...)                                  \
    HPX_REGISTER_ACTION_DECLARATION_(__VA_ARGS__)                             \
/**/

/// \def HPX_REGISTER_ACTION_DECLARATION_TEMPLATE(template, action)
///
/// \brief Declare the necessary component action boilerplate code for actions
///        taking template type arguments.
///
/// The macro \a HPX_REGISTER_ACTION_DECLARATION_TEMPLATE can be used to
/// declare all the boilerplate code which is required for proper functioning
/// of component actions in the context of HPX, if those actions take template
/// type arguments.
///
/// The parameter \a template specifies the list of template type declarations
/// for the action type. This argument has to be wrapped into an additional
/// pair of parenthesis.
///
/// The parameter \a action is the type of the action to declare the
/// boilerplate for. This argument has to be wrapped into an additional pair
/// of parenthesis.
///
/// \par Example:
///
/// \code
///      namespace app
///      {
///          // Define a simple component exposing one action 'print_greating'
///          class HPX_COMPONENT_EXPORT server
///            : public hpx::components::simple_component_base<server>
///          {
///              template <typename T>
///              void print_greating (T t)
///              {
///                  hpx::cout << "Hey " << t << ", how are you?\n" << hpx::flush;
///              }
///
///              // Component actions need to be declared, this also defines the
///              // type 'print_greating_action' representing the action.
///
///              // Actions with template arguments (like print_greating<>()
///              // above) require special type definitions. The simplest way
///              // to define such an action type is by deriving from the HPX
///              // facility make_action:
///              template <typename T>
///              struct print_greating_action
///                : hpx::actions::make_action<
///                      void (server::*)(T), &server::template print_greating<T>,
///                      print_greating_action<T> >
///              {};
///          };
///      }
///
///      // Declare boilerplate code required for each of the component actions.
///      HPX_REGISTER_ACTION_DECLARATION_TEMPLATE((template T), (app::server::print_greating_action<T>));
/// \endcode
///
/// \note This macro has to be used once for each of the component actions
/// defined as above. It has to be visible in all translation units using the
/// action, thus it is recommended to place it into the header file defining the
/// component.
#define HPX_REGISTER_ACTION_DECLARATION_TEMPLATE(TEMPLATE, TYPE)              \
    HPX_SERIALIZATION_REGISTER_TEMPLATE_ACTION(TEMPLATE, TYPE)                \
/**/

/// \def HPX_REGISTER_ACTION(action)
///
/// \brief Define the necessary component action boilerplate code.
///
/// The macro \a HPX_REGISTER_ACTION can be used to define all the
/// boilerplate code which is required for proper functioning of component
/// actions in the context of HPX.
///
/// The parameter \a action is the type of the action to define the
/// boilerplate for.
///
/// This macro can be invoked with an optional second parameter. This parameter
/// specifies a unique name of the action to be used for serialization purposes.
/// The second parameter has to be specified if the first parameter is not
/// usable as a plain (non-qualified) C++ identifier, i.e. the first parameter
/// contains special characters which cannot be part of a C++ identifier, such
/// as '<', '>', or ':'.
///
/// \note This macro has to be used once for each of the component actions
/// defined using one of the \a HPX_DEFINE_COMPONENT_ACTION macros. It has to
/// occur exactly once for each of the actions, thus it is recommended to
/// place it into the source file defining the component. There is no need
/// to use this macro for actions which have template type arguments (see
/// \a HPX_REGISTER_ACTION_DECLARATION_TEMPLATE)
#define HPX_REGISTER_ACTION(...)                                              \
    HPX_REGISTER_ACTION_(__VA_ARGS__)                                         \
/**/

#endif

