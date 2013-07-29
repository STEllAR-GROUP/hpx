//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file action_factory.hpp

#if !defined(HPX_RUNTIME_ACTIONS_ACTION_FACTORY_HPP)
#define HPX_RUNTIME_ACTIONS_ACTION_FACTORY_HPP

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/cstdint.hpp>
#include <map>

namespace hpx { namespace traits {
    // This trait is used to decide whether a class (or specialization) is
    // required to automatically register to the action factory
    template <typename Action, typename Enable = void>
    struct needs_automatic_registration
      : boost::mpl::true_
    {};

}}

namespace hpx { namespace actions {
    namespace detail
    {
        template <typename Action>
        const char * get_action_name();
    }
    
    struct base_action;

    class HPX_ALWAYS_EXPORT action_factory
    {
        public:
            typedef boost::shared_ptr<base_action>(*ctor_type)();
            typedef std::map<std::string, ctor_type> ctor_map;

            static boost::shared_ptr<base_action> create(std::string const & name);

        private:
            void add_action(std::string const & name, ctor_type ctor);
            static HPX_ALWAYS_EXPORT action_factory & get_instance();

            ctor_map ctor_map_;

            template <typename Action>
            friend struct action_registration;
    };

    template <typename Action>
    struct action_registration
    {
        static boost::shared_ptr<base_action> create()
        {
            return boost::shared_ptr<base_action>(new Action());
        }

        action_registration()
        {
            action_factory::get_instance().add_action(
                detail::get_action_name<typename Action::derived_type>(), &action_registration::create
            );
        }
    };

    template <typename Action>
    void automatic_factory_registration()
    {
        action_registration<Action>();
    }

    template <typename Action, typename Enable = typename traits::needs_automatic_registration<Action>::type>
    struct automatic_action_registration
    {
        automatic_action_registration()
        {
            std::cout << "auto registering ...\n";
            automatic_factory_registration<Action> act;
        }

        automatic_action_registration & register_action()
        {
            automatic_factory_registration<Action> act;
            return *this;
        }
    };

    template <typename Action>
    struct automatic_action_registration<Action, boost::mpl::false_>
    {
        automatic_action_registration()
        {
            std::cout << "not auto registering ...\n";
        }

        automatic_action_registration & register_action()
        {
            return *this;
        }
    };
}}

#define HPX_ACTION_REGISTER_ACTION_FACTORY(Action, Name)                        \
    static ::hpx::actions::action_registration<Action>                          \
        const BOOST_PP_CAT(Name, _action_factory_registration) =                \
        ::hpx::actions::action_registration<Action>();                          \
/**/
#endif

