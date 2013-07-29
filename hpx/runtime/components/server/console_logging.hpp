//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CONSOLE_DEC_16_2008_0427PM)
#define HPX_COMPONENTS_CONSOLE_DEC_16_2008_0427PM

#include <string>
#include <vector>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/void_cast.hpp>

#include <boost/fusion/include/vector.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    typedef hpx::util::tuple3<
        logging_destination, std::size_t, std::string
    > message_type;

    typedef std::vector<message_type> messages_type;
}}

//////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    // console logging happens here
    void console_logging(messages_type const&);

    ///////////////////////////////////////////////////////////////////////////
    // this type is a dummy template to avoid premature instantiation of the
    // serialization support instances
    template <typename Dummy = void>
    class console_logging_action
      : public actions::plain_direct_action1<messages_type const&,
        console_logging, console_logging_action<Dummy> >
    {
    private:
        typedef actions::plain_direct_action1<
            messages_type const&, console_logging, console_logging_action>
        base_type;

    public:
        console_logging_action() {}

        // construct an action from its arguments
        console_logging_action(messages_type const& msgs)
          : base_type(msgs)
        {}

        console_logging_action(threads::thread_priority,
                messages_type const& msgs)
          : base_type(msgs)
        {}

    public:
        template <typename Arguments>
        static util::unused_type
        execute_function(naming::address::address_type lva,
            BOOST_FWD_REF(Arguments) args)
        {
            try {
                // call the function, ignoring the return value
                console_logging(
                    boost::move(boost::fusion::at_c<0>(args)));
            }
            catch (hpx::exception const& /*e*/) {
                /**/;      // no logging!
            }
            return util::unused;
        }
    };
}}}

namespace hpx { namespace traits
{
    template <typename Dummy>
    struct needs_guid_initialization<
            hpx::actions::transfer_action<
                hpx::components::server::console_logging_action<Dummy> > >
      : boost::mpl::false_
    {};
}}

HPX_REGISTER_PLAIN_ACTION_DECLARATION(
    hpx::components::server::console_logging_action<>
)

#endif

