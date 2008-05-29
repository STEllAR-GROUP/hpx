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

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/components/component_type.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/serialize_sequence.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    struct action_base
    {
        virtual ~action_base() {}
        virtual std::size_t get_action_code() const = 0;
        virtual component_type get_component_type() const = 0;
        virtual boost::function<bool (hpx::threadmanager::px_thread_self&)>
            get_thread_function(void *component) const = 0;
    };

    typedef boost::shared_ptr<action_base> action_type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, int Action, typename Arguments>
    class action : public action_base
    {
    public:
        typedef Arguments arguments_type;
        
        // this is the action code (id) it is exposed to allow generic handling
        // of this action 
        enum { value = Action };
        
        // construct an action from its arguments
        action() 
          : arguments_() 
        {}
        
        template <typename Arg1>
        action(Arg1 const& arg1) 
          : arguments_(arg1) 
        {}

        template <typename Arg1, typename Arg2>
        action(Arg1 const& arg1, Arg2 const& arg2) 
          : arguments_(arg1, arg2) 
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
        bool (Component::*F)(hpx::threadmanager::px_thread_self&)
    >
    class action0 : public action<Component, Action, boost::fusion::vector<> >
    {
    public:
        action0()
        {}
        
    private:
        boost::function<bool (hpx::threadmanager::px_thread_self&)>
            get_thread_function(void *component) const
        {
            return boost::bind(F, reinterpret_cast<Component*>(component), _1);
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
        bool (Component::*F)(hpx::threadmanager::px_thread_self&, Arg0) 
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

    private:
        boost::function<bool (hpx::threadmanager::px_thread_self&)>
            get_thread_function(void *component) const
        {
            return boost::bind(F, reinterpret_cast<Component*>(component), _1,
                this->get<0>());
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
        bool (Component::*F)(hpx::threadmanager::px_thread_self&, Arg0, Arg1) 
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

    private:
        boost::function<bool (hpx::threadmanager::px_thread_self&)>
            get_thread_function(void *component) const
        {
            return boost::bind(F, reinterpret_cast<Component*>(component), _1,
                this->get<0>(), this->get<1>());
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

