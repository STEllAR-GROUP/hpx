//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/performance_counters/counter_parser.hpp>
#include <hpx/performance_counters/counters.hpp>

#define BOOST_SPIRIT_USE_PHOENIX_V3
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/spirit/include/qi_auxiliary.hpp>
#include <boost/spirit/include/qi_char.hpp>
#include <boost/spirit/include/qi_directive.hpp>
#include <boost/spirit/include/qi_nonterminal.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
#include <boost/spirit/include/qi_operator.hpp>
#include <boost/spirit/include/qi_parse.hpp>
#include <boost/spirit/include/qi_string.hpp>

#include <string>

BOOST_FUSION_ADAPT_STRUCT(hpx::performance_counters::instance_name,
    (std::string, name_)(std::string, index_)(bool, basename_))

BOOST_FUSION_ADAPT_STRUCT(hpx::performance_counters::instance_elements,
    (hpx::performance_counters::instance_name, parent_)(
        hpx::performance_counters::instance_name, child_)(
        hpx::performance_counters::instance_name, subchild_))

BOOST_FUSION_ADAPT_STRUCT(hpx::performance_counters::path_elements,
    (std::string, object_)(hpx::performance_counters::instance_elements,
        instance_)(std::string, counter_)(std::string, parameters_))

namespace {
    ///
    ///    /objectname{parentinstancename#parentindex/instancename#instanceindex}
    ///       /countername#parameters
    ///    /objectname{parentinstancename#*/instancename#*}/countername#parameters
    ///    /objectname{/basecounter}/countername,parameters
    ///
    namespace qi = boost::spirit::qi;

    template <typename Iterator>
    struct path_parser
      : qi::grammar<Iterator, hpx::performance_counters::path_elements()>
    {
        path_parser()
          : path_parser::base_type(start)
        {
            start = -qi::lit(hpx::performance_counters::counter_prefix) >>
                '/' >> +~qi::char_("/{#@") >> -instance >>
                -('/' >> +~qi::char_("#}@")) >> -('@' >> +qi::char_);
            instance =
                '{' >> parent >> -('/' >> child) >> -('/' >> subchild) >> '}';
            parent = &qi::lit('/') >> qi::raw[start] >> qi::attr(-1) >>
                    qi::attr(true)
                // base counter
                | +~qi::char_("#/}") >>
                    ('#' >> raw_uint    // counter parent-instance name
                        | -qi::string(
                              "#*")    // counter parent-instance skeleton name
                        ) >>
                    qi::attr(false);
            child = +~qi::char_("#/}") >>
                (qi::char_('#') >> +~qi::char_("/}")    // counter instance name
                    | -qi::string("#*")    // counter instance skeleton name
                    ) >>
                qi::attr(false);
            subchild = +~qi::char_("#}") >>
                ('#' >> raw_uint    // counter (sub-)instance name
                    |
                    -qi::string("#*")    // counter (sub-)instance skeleton name
                    ) >>
                qi::attr(false);
            raw_uint = qi::raw[qi::uint_];
        }

        qi::rule<Iterator, hpx::performance_counters::path_elements()> start;
        qi::rule<Iterator, hpx::performance_counters::instance_elements()>
            instance;
        qi::rule<Iterator, hpx::performance_counters::instance_name()> parent;
        qi::rule<Iterator, hpx::performance_counters::instance_name()> child;
        qi::rule<Iterator, hpx::performance_counters::instance_name()> subchild;
        qi::rule<Iterator, std::string()> raw_uint;
    };
}    // namespace

namespace hpx { namespace performance_counters {
    bool parse_counter_name(std::string const& name, path_elements& elements)
    {
        path_parser<std::string::const_iterator> p;

        // parse the full name
        std::string::const_iterator begin = name.begin();
        return qi::parse(begin, name.end(), p, elements) && begin == name.end();
    }
}}    // namespace hpx::performance_counters
