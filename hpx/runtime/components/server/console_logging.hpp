//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CONSOLE_DEC_16_2008_0427PM)
#define HPX_COMPONENTS_CONSOLE_DEC_16_2008_0427PM

#include <string>

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/actions/plain_action.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server 
{
    // we know about three different console log destinations
    enum logging_destination
    {
        destination_hpx = 0,
        destination_timing = 1,
        destination_agas = 2,
        destination_app = 3,
    };

    ///////////////////////////////////////////////////////////////////////////
    // console logging happens here
    void console_logging(logging_destination dest, int level, 
        std::string const& msg);

    ///////////////////////////////////////////////////////////////////////////
    // this type is a dummy template to avoid premature instantiation of the 
    // serialization support instances
    template <typename Dummy = void>
    class console_logging_action
      : public actions::plain_direct_action3<logging_destination, int, 
            std::string const&, console_logging, console_logging_action<Dummy> >
    {
    private:
        typedef actions::plain_direct_action3<logging_destination, int, 
            std::string const&, console_logging, console_logging_action> 
        base_type;

    public:
        console_logging_action() {}

        // construct an action from its arguments
        console_logging_action(logging_destination dest, int level, 
                std::string const& msg) 
          : base_type(dest, level, msg) 
        {}

        console_logging_action(threads::thread_priority, 
                logging_destination dest, int level, std::string const& msg) 
          : base_type(dest, level, msg) 
        {}

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<console_logging_action, base_type>();
            base_type::register_base();
        }

    public:
       util::unused_type
       execute_function(naming::address::address_type lva, 
           logging_destination dest, int level, std::string const& msg)
       {
            try {
                // call the function, ignoring the return value
                console_logging(dest, level, msg);
            }
            catch (hpx::exception const& /*e*/) {
                /***/;      // no logging!
            }
            return util::unused;
       }

       static util::unused_type
       execute_function_nonvirt(naming::address::address_type lva, 
           logging_destination dest, int level, std::string const& msg)
       {
            try {
                // call the function, ignoring the return value
                console_logging(dest, level, msg);
            }
            catch (hpx::exception const& /*e*/) {
                /***/;      // no logging!
            }
            return util::unused;
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
}}}

#endif
