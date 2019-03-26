// destination_named.hpp

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details


#ifndef JT28092007_destination_named_HPP_DEFINED
#define JT28092007_destination_named_HPP_DEFINED

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(push)
#pragma warning(disable: 4355)
#endif

#include <hpx/util/logging/detail/fwd.hpp>
#include <hpx/util/logging/detail/manipulator.hpp>
#include <hpx/util/logging/format/array.hpp>    // array
#include <hpx/util/logging/format/destination/convert_destination.hpp>

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace hpx { namespace util { namespace logging { namespace destination {


namespace detail {
    struct named_context {
        typedef base  destination_base_type;
        typedef ::hpx::util::logging::array::ptr_holder<destination_base_type >
            array;

        struct write_info {
            array destinations;
            typedef std::map<std::string, destination_base_type* > coll;
            coll name_to_destination;
            std::string format_string;

            typedef std::vector< destination_base_type* > step_array;
            step_array write_steps;
        };
        write_info m_info;

        template<class destination_type> void add(const std::string & name,
            destination_type dest) {
            // care about if generic or not
            typedef hpx::util::logging::manipulator::is_generic is_generic;
            add_impl<destination_type>( name, dest,
                std::is_base_of<is_generic,destination_type>() );
            compute_write_steps();
        }

        void del(const std::string & name) {
            {
            destination_base_type * p = m_info.name_to_destination[name];
            m_info.name_to_destination.erase(name);
            m_info.destinations.del(p);
            }
            compute_write_steps();
        }

        void configure(const std::string & name,
            const std::string & configure_str) {
            destination_base_type * p = m_info.name_to_destination[name];
            if ( p)
                p->configure(configure_str);
        }

        void format_string(const std::string & str) {
            {
              m_info.format_string = str;
            }
            compute_write_steps();
        }

        void write(const msg_type & msg) const {
            for ( typename write_info::step_array::const_iterator b =
                m_info.write_steps.begin(), e = m_info.write_steps.end(); b != e ; ++b)
                (**b)(msg);
        }

    private:
        // non-generic
        template<class destination_type> void add_impl(const std::string & name,
            destination_type dest, const std::false_type& ) {
            destination_base_type * p = m_info.destinations.append(dest);
            m_info.name_to_destination[name] = p;
        }
        // generic manipulator
        template<class destination_type> void add_impl(const std::string & name,
            destination_type dest, const std::true_type& ) {
            typedef hpx::util::logging::manipulator::detail
                ::generic_holder<destination_type,destination_base_type> holder;
            add_impl( name, holder(dest), std::false_type() );
        }

        // recomputes the write steps - note taht this takes place after each operation
        // for instance, the user might have first set the string and
        // later added the formatters
        void HPX_EXPORT compute_write_steps();
    };

}

/**
@brief Allows you to contain multiple destinations,
give each such destination a name.
Then, at run-time, you can specify a format string which will specify which
destinations to be called, and on what order.

This allows you:
- to hold multiple destinations
- each destination is given a name, when being added.
The name <b>must not</b> contain spaces and must not start with '+'/'-' signs
- you have a %format string, which contains what destinations to be called,
and on which order

The %format string contains destination names, separated by space.

When a message is written to this destination,
I parse the format string. When a name is encountered, if there's a destination
corresponding to this name, I will call it.

Example:

@code
g_l()->writer().add_destination(
    destination::named("cout out debug")
        .add( "cout", destination::cout())
        .add( "debug", destination::dbg_window() )
        .add( "out", destination::file("out.txt"))
     );
@endcode

In the above code, we'll write to 3 destinations, in the following order:
- first, to the console
- second, to the out.txt file
- third, to the debug window



@section If you deal with config files

As an extra feature:
- if a name starts with '-' is ignored
- if a name starts with '+', is included.

This is useful if you want to set this format string in a config file.
The good thing is that this way you can easily turn on/off
certain destinations, while seing all the available destinations as well.

Example: \n <tt>+out_file -debug_window +console</tt> \n
In the above example, I know that the available destinations are @c out_file,
@c debug_window and @c console, but I'm not writing to @c debug_window.


@code
#include <hpx/util/logging/format/destination/named.hpp>
@endcode
*/
struct named : is_generic, non_const_context<detail::named_context > {
    typedef non_const_context<detail::named_context > non_const_context_base;

    /**
        @brief constructs the named destination

        @param named_name name of the named
        @param set [optional] named settings - see named_settings class,
        and @ref dealing_with_flags
    */
    named(const std::string & format_string = std::string() ) {
        non_const_context_base::context().format_string( format_string);
    }
    void operator()(const msg_type & msg) const { //-V659
        non_const_context_base::context().write(msg);
    }


    named & string(const std::string & str) {
        non_const_context_base::context().format_string(str);
        return *this;
    }

    template<class destination> named & add(const std::string & name,
        destination dest) {
        non_const_context_base::context().add(name, dest);
        return *this;
    }

    void del(const std::string & name) {
        non_const_context_base::context().del(name);
    }

    void configure_inner(const std::string & name, const std::string & configure_str) {
        non_const_context_base::context().configure(name, configure_str);
    }

    bool operator==(const named & other) const {
        return &( non_const_context_base::context()) ==
            &( other.non_const_context_base::context());
    }
};

}}}}

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(pop)
#endif

#endif
