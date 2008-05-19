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
        virtual bool execute(void *component) const = 0;
    };

    typedef boost::shared_ptr<action_base> action_type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, int Action, typename Arguments>
    class action : public action_base
    {
    public:
        typedef Arguments arguments_type;
        
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

        template <typename Arg1, typename Arg2, typename Arg3>
        action(Arg1 const& arg1, Arg2 const& arg2, Arg3 const& arg3) 
          : arguments_(arg1, arg2, arg3) 
        {}

        /// destructor
        ~action()
        {}
        
    protected:        
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
            return static_cast<std::size_t>(Action); 
        }
        
        /// retrieve component type
        component_type get_component_type() const
        {
            return static_cast<component_type>(Component::value);
        }

        // serialization support    
        friend class boost::serialization::access;

        template<class Archive>
        void load(Archive& ar, const unsigned int /*version*/)
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
    template <typename Component, int Action, bool (Component::*F)()>
    class action0 : public action<Component, Action, boost::fusion::vector<> >
    {
        bool execute(void *component) const
        {
            return (reinterpret_cast<server::accumulator*>(component)->*F)();
        }

    private:
        // serialization support    
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            using namespace boost::serialization;
            void_cast_register<action0, action>();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  one parameter version
    template <
        typename Component, int Action, 
        typename Arg1, 
        bool (Component::*F)(Arg1) 
    >
    class action1 
      : public action<Component, Action, boost::fusion::vector<Arg1> >
    {
    private:
        typedef 
            action<Component, Action, boost::fusion::vector<Arg1> >
        base_type;
        
    public:
        action1() 
        {}
        
        // construct an action from its arguments
        template <typename Arg1>
        action1(Arg1 const& arg1) 
          : base_type(arg1) 
        {}

    private:
        bool execute(void *component) const
        {
            return (reinterpret_cast<server::accumulator*>(component)->*F)(
                this->get<0>());
        }

    private:
        // serialization support    
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            using namespace boost::serialization;
            void_cast_register<action1, action>();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  two parameter version
    template <
        typename Component, int Action, 
        typename Arg1, typename Arg2,
        bool (Component::*F)(Arg1, Arg2) 
    >
    class action2
      : public action<Component, Action, boost::fusion::vector<Arg1, Arg2> >
    {
    private:
        typedef 
            action<Component, Action, boost::fusion::vector<Arg1, Arg2> >
        base_type;
        
    public:
        action2() 
        {}

        // construct an action from its arguments
        template <typename Arg1, typename Arg2>
        action2(Arg1 const& arg1, Arg2 const& arg2) 
          : base_type(arg1, arg2) 
        {}

    private:
        bool execute(void *component) const
        {
            return (reinterpret_cast<server::accumulator*>(component)->*F)(
                this->get<0>(), this->get<1>());
        }

    private:
        // serialization support    
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            using namespace boost::serialization;
            void_cast_register<action2, action>();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  three parameter version
    template <
        typename Component, int Action, 
        typename Arg1, typename Arg2, typename Arg3,
        bool (Component::*F)(Arg1, Arg2, Arg3) 
    >
    class action3
      : public action<Component, Action, boost::fusion::vector<Arg1, Arg2, Arg3> >
    {
    private:
        typedef 
            action<Component, Action, boost::fusion::vector<Arg1, Arg2, Arg3> >
        base_type;
        
    public:
        action3() 
        {}

        // construct an action from its arguments
        template <typename Arg1, typename Arg2, typename Arg3>
        action3(Arg1 const& arg1, Arg2 const& arg2, Arg3 const& arg3) 
          : base_type(arg1, arg2, arg3) 
        {}
        
    private:
        bool execute(void *component) const
        {
            return (reinterpret_cast<server::accumulator*>(component)->*F)(
                this->get<0>(), this->get<1>(), this->get<2>());
        }

    private:
        // serialization support    
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            using namespace boost::serialization;
            void_cast_register<action3, action>();
        }
    };

///////////////////////////////////////////////////////////////////////////////
}}

#endif

