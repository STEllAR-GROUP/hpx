//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_ACTION_MAR_26_2008_1054AM)
#define HPX_COMPONENTS_ACTION_MAR_26_2008_1054AM

#include <cstdlib>
#include <stdexcept>

#include <boost/serialization/version.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/size.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>
#include <boost/ref.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/components/component_type.hpp>
#include <hpx/util/serialize_sequence.hpp>
#include <hpx/naming/address.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a action_base class is a abstract class used as the base class for 
    /// all action types. It's main purpose is to allow polymorphic 
    /// serialization of action instances
    struct action_base
    {
        virtual ~action_base() {}
        
        /// The function \a get_action_code returns the code of the action 
        /// instance it is called for.
        virtual std::size_t get_action_code() const = 0;

        /// The function \a get_component_type returns the \a component_type 
        /// of the component this action belongs to.
        virtual component_type get_component_type() const = 0;
        
        /// The \a get_thread_function constructs a proper thread function for 
        /// a \a px_thread, encapsulating the functionality and the arguments 
        /// of the action is is called for.
        /// 
        /// \param appl   This is a reference to the \a applier instance to be
        ///               passed as the second parameter to the action function
        /// \param lva    This is the local virtual address of the component 
        ///               the action has to be invoked on.
        ///
        /// \returns      This function returns a proper thread function usable
        ///               for a \a px_thread.
        virtual boost::function<threadmanager::thread_function_type> 
            get_thread_function(applier::applier& appl, 
                naming::address::address_type lva) const = 0;
    };

    typedef boost::shared_ptr<action_base> action_type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, int Action, typename Arguments>
    class action : public action_base
    {
    public:
        typedef Arguments arguments_type;
        
        // This is the action code (id) of this action. It is exposed to allow 
        // generic handling of actions.
        enum { value = Action };
        
        // construct an action from its arguments
        action() 
          : arguments_() 
        {}
        
        template <typename Arg0>
        action(Arg0 const& arg0) 
          : arguments_(arg0) 
        {}

        template <typename Arg0, typename Arg1>
        action(Arg0 const& arg0, Arg1 const& arg1) 
          : arguments_(arg0, arg1) 
        {}

        // bring in the rest of the constructors
        #include <hpx/components/action_constructors.hpp>
        
        /// destructor
        ~action()
        {}
        
    public:        
        /// retrieve the N's argument
        template <int N>
        typename boost::fusion::result_of::at_c<arguments_type, N>::type 
        get() 
        { 
            return boost::fusion::at_c<N>(arguments_); 
        }
        
        template <int N>
        typename boost::fusion::result_of::at_c<arguments_type const, N>::type 
        get() const
        { 
            return boost::fusion::at_c<N>(arguments_); 
        }

    private:
        /// retrieve action code
        std::size_t get_action_code() const 
        { 
            return static_cast<std::size_t>(value); 
        }
        
        /// retrieve component type
        component_type get_component_type() const
        {
            return static_cast<component_type>(Component::value);
        }

        // serialization support    
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            using namespace boost::serialization;
            void_cast_register<action, action_base>();
            
            util::serialize_sequence(ar, arguments_);
        }
        
    private:
        arguments_type arguments_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <int N, typename Component, int Action, typename Arguments>
    inline typename boost::fusion::result_of::at_c<
        typename action<Component, Action, Arguments>::arguments_type const, N
    >::type 
    get(action<Component, Action, Arguments> const& args) 
    { 
        return args.get<N>(); 
    }

    ///////////////////////////////////////////////////////////////////////////
    //  Specialized generic action types allowing to hold a different number of
    //  arguments
    ///////////////////////////////////////////////////////////////////////////
    
    ///////////////////////////////////////////////////////////////////////////
    //  zero parameter version
    template <
        typename Component, int Action, 
        threadmanager::thread_state(Component::*F)(
            threadmanager::px_thread_self&, applier::applier&)
    >
    class action0 : public action<Component, Action, boost::fusion::vector<> >
    {
    public:
        action0()
        {}
        
        /// \brief The static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a px_thread without having to 
        /// instantiate the action0 type. This is used by the \a applier.
        static boost::function<threadmanager::thread_function_type> 
        construct_thread_function(applier::applier& appl, 
            naming::address::address_type lva)
        {
            return boost::bind(F, reinterpret_cast<Component*>(lva), _1, 
                boost::ref(appl));
        }

    private:
        boost::function<threadmanager::thread_function_type>
        get_thread_function(applier::applier& appl, 
            naming::address::address_type lva) const
        {
            return construct_thread_function(appl, lva);
        }

    private:
        // serialization support    
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<action>(*this);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  one parameter version
    template <
        typename Component, int Action, typename Arg0, 
        threadmanager::thread_state(Component::*F)(
            threadmanager::px_thread_self&, applier::applier&, Arg0) 
    >
    class action1 
      : public action<Component, Action, boost::fusion::vector<Arg0> >
    {
    private:
        typedef 
            action<Component, Action, boost::fusion::vector<Arg0> >
        base_type;
        
    public:
        action1() 
        {}
        
        // construct an action from its arguments
        template <typename Arg0>
        action1(Arg0 const& arg0) 
          : base_type(arg0) 
        {}

        static boost::function<threadmanager::thread_function_type> 
        construct_thread_function(applier::applier& appl, 
            naming::address::address_type lva, Arg0 const& arg0) 
        {
            return boost::bind(F, reinterpret_cast<Component*>(lva), _1, 
                boost::ref(appl), arg0);
        }

    private:
        boost::function<threadmanager::thread_function_type>
            get_thread_function(applier::applier& appl, 
                naming::address::address_type lva) const
        {
            return construct_thread_function(appl, lva, this->get<0>());
        }

    private:
        // serialization support    
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<action>(*this);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  two parameter version
    template <
        typename Component, int Action, typename Arg0, typename Arg1, 
        threadmanager::thread_state(Component::*F)(
            threadmanager::px_thread_self&, applier::applier&, Arg0, Arg1)
    >
    class action2
      : public action<Component, Action, boost::fusion::vector<Arg0, Arg1> >
    {
    private:
        typedef 
            action<Component, Action, boost::fusion::vector<Arg0, Arg1> >
        base_type;
        
    public:
        action2() 
        {}

        // construct an action from its arguments
        template <typename Arg0, typename Arg1>
        action2(Arg0 const& arg0, Arg1 const& arg1) 
          : base_type(arg0, arg2) 
        {}

        static boost::function<threadmanager::thread_function_type> 
        construct_thread_function(applier::applier& appl, 
            naming::address::address_type lva, Arg0 const& arg0, 
            Arg1 const& arg1) 
        {
            return boost::bind(F, reinterpret_cast<Component*>(lva), _1, 
                boost::ref(appl), arg0, arg1);
        }

    private:
        boost::function<threadmanager::thread_function_type>
            get_thread_function(applier::applier& appl, 
                naming::address::address_type lva) const
        {
            return construct_thread_function(appl, lva, this->get<0>(),
                this->get<1>());
        }

    private:
        // serialization support    
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<action>(*this);
        }
    };

    // bring in the rest of the implementations
    #include <hpx/components/action_implementations.hpp>
    
///////////////////////////////////////////////////////////////////////////////
}}

#endif

