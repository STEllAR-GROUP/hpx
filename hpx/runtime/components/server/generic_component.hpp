//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_GENERIC_COMPONENT_OCT_12_2008_0353PM)
#define HPX_COMPONENTS_SERVER_GENERIC_COMPONENT_OCT_12_2008_0353PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/util/serialize_sequence.hpp>

#include <boost/preprocessor/stringize.hpp>
#include <boost/serialization/export.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/mpl.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    ///
    /// Use this as:
    /// \code
    ///
    ///     #include <hpx/hpx.hpp>
    ///     #include <hpx/runtime/components/server/generic_component.hpp>
    ///
    ///     // This is the function to wrap into a component
    ///     void doit (threads::thread_self&, applier::applier&)
    ///     {
    ///         // do something useful here
    ///     }
    ///
    ///     // This has to be placed into a source file (needs to be compiled 
    ///     // once). We use generic_component0 here because the function doit()
    ///     // takes zero additional arguments. The number of additional 
    ///     // arguments N needs to be reflected in the name of the 
    ///     // generic_componentN.
    ///     typedef generic_component0<doit> doit_wrapper;
    ///
    ///     HPX_REGISTER_GENERIC_COMPONENT(doit_wrapper)
    ///
    ///     // In addition to the above, if you are creating a shared component 
    ///     // library, you need to place the following into one of the 
    ///     // translation units of your shared library.
    ///     HPX_REGISTER_COMPONENT_MODULE()
    ///
    ///     // If you are creating an executable HPX application you need to 
    ///     // place the following into one of the translation units of your
    ///     // application (instead of HPX_REGISTER_COMPONENT_MODULE()).
    ///     HPX_REGISTER_COMPONENT_APPLICATION()
    ///
    /// \endcode
    template <
        typename Result, 
        Result (*F)(threads::thread_self&, applier::applier&)
    > 
    class generic_component0
      : public simple_component_base<generic_component0<Result, F> >
    {
    public:
        typedef Result result_type;
        typedef typename 
            boost::fusion::result_of::as_vector<boost::mpl::vector<> >::type
        parameter_block_type;

        // parcel action code: the action to be performed 
        enum actions
        {
            generic_component_action = 0
        };

        generic_component0(applier::applier& appl)
          : simple_component_base<generic_component0<Result, F> >(appl)
        {}

        threads::thread_state
        eval (threads::thread_self& self, applier::applier& appl, Result* r) 
        {
            if (NULL != r)
                *r = F(self, appl);
            else
                F(self, appl);
            return threads::terminated;
        }

        typedef hpx::actions::result_action0<
            generic_component0, Result, generic_component_action, 
            &generic_component0::eval
        > eval_action;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <void (*F)(threads::thread_self&, applier::applier&)>
    class generic_component0<void, F>
      : public simple_component_base<generic_component0<void, F> >
    {
    public:
        typedef void result_type;
        typedef typename 
            boost::fusion::result_of::as_vector<boost::mpl::vector<> >::type
        parameter_block_type;

        // parcel action code: the action to be performed 
        enum actions
        {
            generic_component_action = 0
        };

        generic_component0(applier::applier& appl)
          : simple_component_base<generic_component0<void, F> >(appl)
        {}

        threads::thread_state 
        eval (threads::thread_self& self, applier::applier& appl) 
        {
            F(self, appl);
            return threads::terminated;
        }

        typedef hpx::actions::action0<
            generic_component0, generic_component_action, 
            &generic_component0::eval
        > eval_action;
    };

    // bring in higher order generic components
    #include <hpx/runtime/components/server/generic_component_implementation.hpp>

}}}

///////////////////////////////////////////////////////////////////////////////
/// Define a helper macro allowing to define all additional facilities needed 
/// for a generic component declared using the generic_componentN templates
#define HPX_REGISTER_GENERIC_COMPONENT(c)                                     \
        HPX_REGISTER_ACTION(c::eval_action)                                   \
        HPX_DEFINE_GET_COMPONENT_TYPE(c)                                      \
        HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(                               \
            hpx::components::simple_component<c>, c)                          \
    /**/

#endif

