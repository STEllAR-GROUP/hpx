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

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#if defined(HPX_MSVC)
#pragma warning ( disable : 4355)
#endif



#include <hpx/util/logging/detail/fwd.hpp>
#include <hpx/util/logging/detail/manipulator.hpp>
#include <hpx/util/logging/format/destination/convert_destination.hpp>
#include <hpx/util/logging/format/array.hpp> // array
#include <boost/type_traits/is_base_of.hpp>
#include <map>
#include <sstream>
#include <vector>

namespace hpx { namespace util { namespace logging { namespace destination {


namespace detail {
    template<class lock_resource, class destination_base> struct named_context {
        typedef typename use_default<lock_resource,
            hpx::util::logging::lock_resource_finder::tss_with_cache<> >
            ::type lock_resource_type;
        typedef typename use_default<destination_base, base<> >
            ::type  destination_base_type;
        typedef ::hpx::util::logging::array::shared_ptr_holder<destination_base_type,
            hpx::util::logging::threading::no_mutex > array;
        typedef hold_string_type string_type;

        struct write_info {
            array destinations;
            typedef std::map<string_type, destination_base_type* > coll;
            coll name_to_destination;
            string_type format_string;

            typedef std::vector< destination_base_type* > step_array;
            step_array write_steps;
        };
        typedef typename lock_resource_type::template finder<write_info>::type data;
        data m_data;

        template<class destination_type> void add(const string_type & name,
            destination_type dest) {
            // care about if generic or not
            typedef hpx::util::logging::manipulator::is_generic is_generic;
            add_impl<destination_type>( name, dest,
                boost::is_base_of<is_generic,destination_type>() );
            compute_write_steps();
        }

        void del(const string_type & name) {
            {
            typename data::write info(m_data);
            destination_base_type * p = info->name_to_destination[name];
            info->name_to_destination.erase(name);
            info->destinations.del(p);
            }
            compute_write_steps();
        }

        void configure(const string_type & name,
            const string_type & configure_str) {
            typename data::write info(m_data);
            destination_base_type * p = info->name_to_destination[name];
            if ( p)
                p->configure(configure_str);
        }

        void format_string(const string_type & str) {
            { typename data::write info(m_data);
              info->format_string = str;
            }
            compute_write_steps();
        }

        template<class msg_type> void write(msg_type & msg) const {
            typename data::read info(m_data);
            for ( typename write_info::step_array::const_iterator b =
                info->write_steps.begin(), e = info->write_steps.end(); b != e ; ++b)
                (**b)(msg);
        }

    private:
        // non-generic
        template<class destination_type> void add_impl(const string_type & name,
            destination_type dest, const boost::false_type& ) {
            typename data::write info(m_data);
            destination_base_type * p = info->destinations.append(dest);
            info->name_to_destination[name] = p;
        }
        // generic manipulator
        template<class destination_type> void add_impl(const string_type & name,
            destination_type dest, const boost::true_type& ) {
            typedef hpx::util::logging::manipulator::detail
                ::generic_holder<destination_type,destination_base_type> holder;
            add_impl( name, holder(dest), boost::false_type() );
        }

        // recomputes the write steps - note taht this takes place after each operation
        // for instance, the user might have first set the string and
        // later added the formatters
        void compute_write_steps() {
            typename data::write info(m_data);
            info->write_steps.clear();

            std::basic_istringstream<char_type> in(info->format_string);
            string_type word;
            while ( in >> word) {
                if ( word[0] == '+')
                    word.erase( word.begin());
                else if ( word[0] == '-')
                    // ignore this word
                    continue;

                if ( info->name_to_destination.find(word) !=
                    info->name_to_destination.end())
                    info->write_steps.push_back( info->
                        name_to_destination.find(word)->second);
            }
        }

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
template<class destination_base = default_, class lock_resource = default_ >
struct named_t : is_generic, non_const_context<detail
    ::named_context<lock_resource,destination_base> > {
    typedef non_const_context< detail::named_context<lock_resource,destination_base>
    > non_const_context_base;
    typedef hold_string_type string_type;

    /**
        @brief constructs the named destination

        @param named_name name of the named
        @param set [optional] named settings - see named_settings class,
        and @ref dealing_with_flags
    */
    named_t(const string_type & format_string = string_type() ) {
        non_const_context_base::context().format_string( format_string);
    }
    template<class msg_type> void operator()(const msg_type & msg) const { //-V659
        non_const_context_base::context().write(msg);
    }


    named_t & string(const string_type & str) {
        non_const_context_base::context().format_string(str);
        return *this;
    }

    template<class destination> named_t & add(const string_type & name,
        destination dest) {
        non_const_context_base::context().add(name, dest);
        return *this;
    }

    void del(const string_type & name) {
        non_const_context_base::context().del(name);
    }

    void configure_inner(const string_type & name, const string_type & configure_str) {
        non_const_context_base::context().configure(name, configure_str);
    }

    bool operator==(const named_t & other) const {
        return &( non_const_context_base::context()) ==
            &( other.non_const_context_base::context());
    }
};

/** @brief named_t with default values. See named_t

@copydoc named_t
*/
typedef named_t<> named;

}}}}

#endif

