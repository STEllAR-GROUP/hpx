//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_COMPONENT_ACTION_MAR_26_2008_1054AM)
#define HPX_RUNTIME_ACTIONS_COMPONENT_ACTION_MAR_26_2008_1054AM

#include <cstdlib>
#include <stdexcept>

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/util/unused.hpp>

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
    //  Specialized generic component action types allowing to hold a different 
    //  number of arguments
    ///////////////////////////////////////////////////////////////////////////

    // zero argument version
    template <
        typename Component, typename Result, int Action, 
        Result (Component::*F)()
    >
    class base_result_action0 
      : public action<Component, Action, boost::fusion::vector<> >
    {
        typedef action<Component, Action, boost::fusion::vector<> > base_type;

    public:
        base_result_action0()
        {}

    private:
        /// The \a continuation_thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the 
        /// original function (given by \a func), while ignoring the return 
        /// value.
        static threads::thread_state 
        thread_function(naming::address::address_type lva)
        {
            (get_lva<Component>::call(lva)->*F)();      // just call the function
            return threads::terminated;
        }

    public:
        typedef boost::mpl::false_ direct_execution;
        typedef Result result_type;

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the \a base_result_action0 type. This is used by the \a 
        /// applier in case no continuation has been supplied.
        static boost::function<threads::thread_function_type> 
        construct_thread_function(naming::address::address_type lva)
        {
            return boost::bind(&base_result_action0::thread_function, lva);
        }

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the \a base_result_action0 type. This is used by the \a 
        /// applier in case a continuation has been supplied
        static boost::function<threads::thread_function_type> 
        construct_thread_function(continuation_type& cont, 
            naming::address::address_type lva)
        {
            return base_type::construct_continuation_thread_function(
                boost::bind(F, get_lva<Component>::call(lva)), cont);
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<base_result_action0, base_type>();
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
    template <
        typename Component, typename Result, int Action, 
        Result (Component::*F)()
    >
    class result_action0 
      : public base_result_action0<Component, Result, Action, F>
    {
    private:
        typedef base_result_action0<Component, Result, Action, F> base_type;

    public:
        result_action0()
        {}

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* const get_action_name() const
        {
            return detail::get_action_name<result_action0>();
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<result_action0, base_type>();
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

    private:
        threads::thread_init_data& 
        get_thread_init_data(naming::address::address_type lva,
            threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(lva);
            data.description = detail::get_action_name<result_action0>();
            data.parent_id = reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_prefix = this->parent_locality_;
            return data;
        }

        threads::thread_init_data& 
        get_thread_init_data(continuation_type& cont,
            naming::address::address_type lva, 
            threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(cont, lva);
            data.description = detail::get_action_name<result_action0>();
            data.parent_id = reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_prefix = this->parent_locality_;
            return data;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename Result, int Action, 
        Result (Component::*F)()
    >
    class direct_result_action0 
      : public base_result_action0<Component, Result, Action, F>
    {
    private:
        typedef base_result_action0<Component, Result, Action, F> base_type;

    public:
        direct_result_action0()
        {}

    public:
        typedef boost::mpl::true_ direct_execution;

        ///
        static Result execute_function(naming::address::address_type lva)
        {
            return (get_lva<Component>::call(lva)->*F)();
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<direct_result_action0, base_type>();
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

    private:
        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* const get_action_name() const
        {
            return detail::get_action_name<direct_result_action0>();
        }

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        base_action::action_type get_action_type() const 
        {
            return base_action::direct_action;
        }

        threads::thread_init_data& 
        get_thread_init_data(naming::address::address_type lva,
            threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(lva);
            data.description = detail::get_action_name<direct_result_action0>();
            data.parent_id = reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_prefix = this->parent_locality_;
            return data;
        }

        threads::thread_init_data& 
        get_thread_init_data(continuation_type& cont,
            naming::address::address_type lva, 
            threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(cont, lva);
            data.description = detail::get_action_name<direct_result_action0>();
            data.parent_id = reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_prefix = this->parent_locality_;
            return data;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  zero parameter version, no result value
    template <typename Component, int Action, void (Component::*F)()>
    class base_action0 
      : public action<Component, Action, boost::fusion::vector<> >
    {
    private:
        typedef action<Component, Action, boost::fusion::vector<> > base_type;

    public:
        base_action0()
        {}

    private:
        /// The \a continuation_thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the 
        /// original function (given by \a func), while ignoring the return 
        /// value.
        static threads::thread_state 
        thread_function(naming::address::address_type lva)
        {
            (get_lva<Component>::call(lva)->*F)();      // just call the function
            return threads::terminated;
        }

    public:
        typedef boost::mpl::false_ direct_execution;
        typedef util::unused_type result_type;

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the base_action0 type. This is used by the \a applier in 
        /// case no continuation has been supplied.
        static boost::function<threads::thread_function_type> 
        construct_thread_function(naming::address::address_type lva)
        {
            return boost::bind(&base_action0::thread_function, lva);
        }

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the base_action0 type. This is used by the \a applier in 
        /// case a continuation has been supplied
        static boost::function<threads::thread_function_type> 
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva)
        {
            return base_type::construct_continuation_thread_function_void(
                boost::bind(F, get_lva<Component>::call(lva)), cont);
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<base_action0, base_type>();
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
    template <typename Component, int Action, void (Component::*F)()>
    class action0 : public base_action0<Component, Action, F>
    {
    private:
        typedef base_action0<Component, Action, F> base_type;

    public:
        action0()
        {}

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* const get_action_name() const
        {
            return detail::get_action_name<action0>();
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<action0, base_type>();
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

    private:
        threads::thread_init_data& 
        get_thread_init_data(naming::address::address_type lva,
            threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(lva);
            data.description = detail::get_action_name<action0>();
            data.parent_id = reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_prefix = this->parent_locality_;
            return data;
        }

        threads::thread_init_data& 
        get_thread_init_data(continuation_type& cont,
            naming::address::address_type lva, 
            threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(cont, lva);
            data.description = detail::get_action_name<action0>();
            data.parent_id = reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_prefix = this->parent_locality_;
            return data;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, int Action, void (Component::*F)()>
    class direct_action0 : public base_action0<Component, Action, F>
    {
    private:
        typedef base_action0<Component, Action, F> base_type;

    public:
        direct_action0()
        {}

    public:
        typedef boost::mpl::true_ direct_execution;

        ///
        static util::unused_type execute_function(naming::address::address_type lva)
        {
            (get_lva<Component>::call(lva)->*F)();
            return util::unused;
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<direct_action0, base_type>();
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

    private:
        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* const get_action_name() const
        {
            return detail::get_action_name<direct_action0>();
        }

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        base_action::action_type get_action_type() const 
        {
            return base_action::direct_action;
        }

        threads::thread_init_data& 
        get_thread_init_data(naming::address::address_type lva,
            threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(lva);
            data.description = detail::get_action_name<direct_action0>();
            data.parent_id = reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_prefix = this->parent_locality_;
            return data;
        }

        threads::thread_init_data& 
        get_thread_init_data(continuation_type& cont,
            naming::address::address_type lva, 
            threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = this->construct_thread_function(cont, lva);
            data.description = detail::get_action_name<direct_action0>();
            data.parent_id = reinterpret_cast<threads::thread_id_type>(this->parent_id_);
            data.parent_prefix = this->parent_locality_;
            return data;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // bring in the rest of the implementations
    #include <hpx/runtime/actions/component_action_implementations.hpp>

///////////////////////////////////////////////////////////////////////////////
}}

#include <hpx/config/warnings_suffix.hpp>

#endif

