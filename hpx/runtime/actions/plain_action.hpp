//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_PLAIN_ACTION_NOV_14_2008_0706PM)
#define HPX_RUNTIME_ACTIONS_PLAIN_ACTION_NOV_14_2008_0706PM

#include <cstdlib>
#include <stdexcept>

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/components/server/plain_function.hpp>

#include <boost/version.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
#define HPX_FUNCTION_ARG_ENUM(z, n, data)                                     \
        BOOST_PP_CAT(function_action_arg, BOOST_PP_INC(n)) =                  \
            function_action_base + BOOST_PP_INC(n),                           \
    /**/
#define HPX_FUNCTION_RETARG_ENUM(z, n, data)                                  \
        BOOST_PP_CAT(function_result_action_arg, BOOST_PP_INC(n)) =           \
            function_result_action_base + BOOST_PP_INC(n),                    \
    /**/

    enum function_action
    {
        /// plain (free) remotely callable function identifiers
        function_action_base = 100,
        function_action_arg0 = function_action_base + 0, 
        BOOST_PP_REPEAT(HPX_PLAIN_ACTION_ARGUMENT_LIMIT, HPX_FUNCTION_ARG_ENUM, _)

        /// plain (free) remotely callable function identifiers with result
        function_result_action_base = 200,
        function_result_action_arg0 = function_result_action_base + 0, 
        BOOST_PP_REPEAT(HPX_PLAIN_ACTION_ARGUMENT_LIMIT, HPX_FUNCTION_RETARG_ENUM, _)
    };

#undef HPX_FUNCTION_RETARG_ENUM
#undef HPX_FUNCTION_ARG_ENUM

    ///////////////////////////////////////////////////////////////////////////
    //  Specialized generic plain (free) action types allowing to hold a 
    //  different number of arguments
    ///////////////////////////////////////////////////////////////////////////

    // zero argument version
    template <typename Result, Result (*F)()>
    class plain_base_result_action0 
      : public action<
            components::server::plain_function, function_result_action_arg0, 
            boost::fusion::vector<> 
        >
    {
        typedef action<
            components::server::plain_function, function_result_action_arg0, 
            boost::fusion::vector<> 
        > base_type;

    public:
        plain_base_result_action0()
        {}

    private:
        /// The \a continuation_thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the 
        /// original function (given by \a func), while ignoring the return 
        /// value.
        static threads::thread_state thread_function(threads::thread_state_ex)
        {
            F();      // call the function, ignoring the return value
            return threads::terminated;
        }

    public:
        typedef boost::mpl::false_ direct_execution;
        typedef Result result_type;

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the \a plain_base_result_action0 type. This is used by 
        /// the \a applier in case no continuation has been supplied.
        static boost::function<threads::thread_function_type> 
        construct_thread_function(naming::address::address_type lva)
        {
            return &plain_base_result_action0::thread_function;
        }

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the \a base_result_action0 type. This is used by the \a 
        /// applier in case a continuation has been supplied
        static boost::function<threads::thread_function_type> 
        construct_thread_function(continuation_type& cont, 
            naming::address::address_type lva)
        {
            return base_type::construct_continuation_thread_function(F, cont);
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<plain_base_result_action0, base_type>();
            base_type::register_base();
        }

    private:
        /// This \a get_thread_function will be invoked to retrieve the thread 
        /// function for an action which has to be invoked without continuations.
        boost::function<threads::thread_function_type>
        get_thread_function(naming::address::address_type lva) const
        {
            return construct_thread_function(lva);
        }

        /// This \a get_thread_function will be invoked to retrieve the thread 
        /// function for an action which has to be invoked with continuations.
        boost::function<threads::thread_function_type>
        get_thread_function(continuation_type& cont,
            naming::address::address_type lva) const
        {
            return construct_thread_function(cont, lva);
        }

    private:
        // serialization support    
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, Result (*F)()>
    class plain_result_action0 
      : public plain_base_result_action0<Result, F>
    {
    private:
        typedef plain_base_result_action0<Result, F> base_type;

    public:
        plain_result_action0()
        {}

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* const get_action_name() const
        {
            return detail::get_action_name<plain_result_action0>();
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<plain_result_action0, base_type>();
            base_type::register_base();
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, Result (*F)()>
    class plain_direct_result_action0 
      : public plain_base_result_action0<Result, F>
    {
    private:
        typedef plain_base_result_action0<Result, F> base_type;

    public:
        plain_direct_result_action0()
        {}

    public:
        typedef boost::mpl::true_ direct_execution;

        ///
        static Result execute_function(naming::address::address_type lva)
        {
            return F();
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<plain_direct_result_action0, base_type>();
            base_type::register_base();
        }

    private:
        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* const get_action_name() const
        {
            return detail::get_action_name<plain_direct_result_action0>();
        }

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        action_type get_action_type() const 
        {
            return base_action::direct_action;
        }

        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  zero parameter version, no result value
    template <void (*F)()>
    class plain_base_action0 
      : public action<
            components::server::plain_function, function_action_arg0, 
            boost::fusion::vector<> 
        >
    {
    private:
        typedef action<
            components::server::plain_function, function_action_arg0, 
            boost::fusion::vector<> 
        > base_type;

    public:
        plain_base_action0()
        {}

    private:
        /// The \a continuation_thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the 
        /// original function (given by \a func), while ignoring the return 
        /// value.
        static threads::thread_state thread_function(threads::thread_state_ex)
        {
            F();      // just call the function
            return threads::terminated;
        }

    public:
        typedef boost::mpl::false_ direct_execution;
        typedef void result_type;

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the base_action0 type. This is used by the \a applier in 
        /// case no continuation has been supplied.
        static boost::function<threads::thread_function_type> 
        construct_thread_function(naming::address::address_type lva)
        {
            return &plain_base_action0::thread_function;
        }

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the base_action0 type. This is used by the \a applier in 
        /// case a continuation has been supplied
        static boost::function<threads::thread_function_type> 
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva)
        {
            return base_type::construct_continuation_thread_function_void(F, cont);
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<plain_base_action0, base_type>();
            base_type::register_base();
        }

    private:
        boost::function<threads::thread_function_type>
        get_thread_function(naming::address::address_type lva) const
        {
            return construct_thread_function(lva);
        }

        boost::function<threads::thread_function_type>
        get_thread_function(continuation_type& cont,
            naming::address::address_type lva) const
        {
            return construct_thread_function(cont, lva);
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <void (*F)()>
    class plain_action0 : public plain_base_action0<F>
    {
    private:
        typedef plain_base_action0<F> base_type;

    public:
        plain_action0()
        {}

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* const get_action_name() const
        {
            return detail::get_action_name<plain_action0>();
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<plain_action0, base_type>();
            base_type::register_base();
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <void (*F)()>
    class plain_direct_action0 : public plain_base_action0<F>
    {
    private:
        typedef plain_base_action0<F> base_type;

    public:
        plain_direct_action0()
        {}

    public:
        typedef boost::mpl::true_ direct_execution;

        ///
        static void execute_function(naming::address::address_type lva)
        {
            F();
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<plain_direct_action0, base_type>();
            base_type::register_base();
        }

    private:
        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* const get_action_name() const
        {
            return detail::get_action_name<plain_direct_action0>();
        }

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        action_type get_action_type() const 
        {
            return base_action::direct_action;
        }

        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  one parameter version
    template <typename Result, typename T0, Result (*F)(T0)>
    class plain_base_result_action1
      : public action<
            components::server::plain_function, function_result_action_arg1, 
            boost::fusion::vector<typename detail::remove_qualifiers<T0>::type> 
        >
    {
    private:
        typedef action<
            components::server::plain_function, function_result_action_arg1, 
            boost::fusion::vector<typename detail::remove_qualifiers<T0>::type> 
        > base_type;

    public:
        plain_base_result_action1() 
        {}

        // construct an action from its arguments
        template <typename Arg0>
        plain_base_result_action1(Arg0 const& arg0) 
          : base_type(arg0) 
        {}

    private:
        /// The \a continuation_thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the 
        /// original function (given by \a func).
        template <typename Arg0>
        static threads::thread_state thread_function(Arg0 const& arg0)
        {
            F(arg0);
            return threads::terminated;
        }

    public:
        typedef boost::mpl::false_ direct_execution;
        typedef Result result_type;

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the \a base_result_action1 type. This is used by the \a 
        /// applier in case no continuation has been supplied.
        template <typename Arg0>
        static boost::function<threads::thread_function_type> 
        construct_thread_function(naming::address::address_type lva, 
            Arg0 const& arg0) 
        {
            // we need to assign the address of the thread function to a 
            // variable to  help the compiler to deduce the function type
            threads::thread_state (*f)(Arg0 const&) =
                &plain_base_result_action1::template thread_function<Arg0>;

            return boost::bind(f, arg0);
        }

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the \a base_result_action1 type. This is used by the \a 
        /// applier in case a continuation has been supplied
        template <typename Arg0>
        static boost::function<threads::thread_function_type> 
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arg0 const& arg0) 
        {
            return base_type::construct_continuation_thread_function(
                boost::bind(F, arg0), cont);
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<plain_base_result_action1, base_type>();
            base_type::register_base();
        }

    private:
        /// This \a get_thread_function will be invoked to retrieve the thread 
        /// function for an action which has to be invoked without continuations.
        boost::function<threads::thread_function_type>
        get_thread_function(naming::address::address_type lva) const
        {
            return construct_thread_function(lva, this->get<0>());
        }

        /// This \a get_thread_function will be invoked to retrieve the thread 
        /// function for an action which has to be invoked with continuations.
        boost::function<threads::thread_function_type>
        get_thread_function(continuation_type& cont,
            naming::address::address_type lva) const
        {
            return construct_thread_function(cont, lva, this->get<0>());
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename T0, Result (*F)(T0)>
    class plain_result_action1 
      : public plain_base_result_action1<Result, T0, F>
    {
    private:
        typedef plain_base_result_action1<Result, T0, F> base_type;

    public:
        plain_result_action1()
        {}

        template <typename Arg0>
        plain_result_action1(Arg0 const& arg0)
          : base_type(arg0)
        {}

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* const get_action_name() const
        {
            return detail::get_action_name<plain_result_action1>();
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<plain_result_action1, base_type>();
            base_type::register_base();
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename T0, Result (*F)(T0)>
    class plain_direct_result_action1 
      : public plain_base_result_action1<Result, T0, F>
    {
    private:
        typedef plain_base_result_action1<Result, T0, F> base_type;

    public:
        plain_direct_result_action1()
        {}

        template <typename Arg0>
        plain_direct_result_action1(Arg0 const& arg0)
          : base_type(arg0)
        {}

    public:
        typedef boost::mpl::true_ direct_execution;

        ///
        template <typename Arg0>
        static Result execute_function(naming::address::address_type lva,
            Arg0 const& arg0)
        {
            return F(arg0);
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<plain_direct_result_action1, base_type>();
            base_type::register_base();
        }

    private:
        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* const get_action_name() const
        {
            return detail::get_action_name<plain_direct_result_action1>();
        }

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        action_type get_action_type() const 
        {
            return base_action::direct_action;
        }

        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };

    //  one parameter version, no result value
    template <typename T0, void (*F)(T0)>
    class plain_base_action1 
      : public action<
            components::server::plain_function, function_action_arg1, 
            boost::fusion::vector<typename detail::remove_qualifiers<T0>::type> 
        >
    {
    private:
        typedef action<
            components::server::plain_function, function_action_arg1, 
            boost::fusion::vector<typename detail::remove_qualifiers<T0>::type> 
        > base_type;

    public:
        plain_base_action1() 
        {}

        // construct an action from its arguments
        template <typename Arg0>
        plain_base_action1(Arg0 const& arg0) 
          : base_type(arg0) 
        {}

    private:
        /// The \a continuation_thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the 
        /// original function (given by \a func).
        template <typename Arg0>
        static threads::thread_state thread_function(Arg0 const& arg0)
        {
            F(arg0);
            return threads::terminated;
        }

    public:
        typedef boost::mpl::false_ direct_execution;
        typedef void result_type;

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the \a base_action1 type. This is used by the \a applier in 
        /// case no continuation has been supplied.
        template <typename Arg0>
        static boost::function<threads::thread_function_type> 
        construct_thread_function(naming::address::address_type lva, 
            Arg0 const& arg0) 
        {
            // we need to assign the address of the thread function to a 
            // variable to  help the compiler to deduce the function type
            threads::thread_state (*f)(Arg0 const&) =
                &plain_base_action1::template thread_function<Arg0>;

            return boost::bind(f, arg0);
        }

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the \a base_action1 type. This is used by the \a applier in 
        /// case a continuation has been supplied
        template <typename Arg0>
        static boost::function<threads::thread_function_type> 
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arg0 const& arg0) 
        {
            return base_type::construct_continuation_thread_function_void(
                boost::bind(F, arg0), cont);
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<plain_base_action1, base_type>();
            base_type::register_base();
        }

    private:
        ///
        boost::function<threads::thread_function_type>
        get_thread_function(naming::address::address_type lva) const
        {
            return construct_thread_function(lva, this->get<0>());
        }

        ///
        boost::function<threads::thread_function_type>
        get_thread_function(continuation_type& cont,
            naming::address::address_type lva) const
        {
            return construct_thread_function(cont, lva, this->get<0>());
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T0, void (*F)(T0)>
    class plain_action1 : public plain_base_action1<T0, F>
    {
    private:
        typedef plain_base_action1<T0, F> base_type;

    public:
        plain_action1()
        {}

        // construct an action from its arguments
        template <typename Arg0>
        plain_action1(Arg0 const& arg0) 
          : base_type(arg0) 
        {}

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* const get_action_name() const
        {
            return detail::get_action_name<plain_action1>();
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<plain_action1, base_type>();
            base_type::register_base();
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T0, void (*F)(T0)>
    class plain_direct_action1 : public plain_base_action1<T0, F>
    {
    private:
        typedef plain_base_action1<T0, F> base_type;

    public:
        plain_direct_action1()
        {}

        // construct an action from its arguments
        template <typename Arg0>
        plain_direct_action1(Arg0 const& arg0) 
          : base_type(arg0) 
        {}

    public:
        typedef boost::mpl::true_ direct_execution;

        ///
        template <typename Arg0>
        static void execute_function(naming::address::address_type lva, 
            Arg0 const& arg0)
        {
            F(arg0);
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<plain_direct_action1, base_type>();
            base_type::register_base();
        }

    private:
        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* const get_action_name() const
        {
            return detail::get_action_name<plain_direct_action1>();
        }

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        action_type get_action_type() const 
        {
            return base_action::direct_action;
        }

        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };

    // bring in the rest of the implementations
    #include <hpx/runtime/actions/plain_action_implementations.hpp>

///////////////////////////////////////////////////////////////////////////////
}}

#include <hpx/config/warnings_suffix.hpp>

#endif

