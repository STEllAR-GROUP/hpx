////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <fstream>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/home/support/iterators/istream_iterator.hpp>

#include <examples/math/csv/parse.hpp>

namespace hpx { namespace math { namespace csv 
{

template <typename Iterator>
struct parser
    : boost::spirit::qi::grammar<
        Iterator
      , std::vector<std::vector<double> >()
      , boost::spirit::qi::space_type
    >
{
    boost::spirit::qi::rule<
        Iterator
      , std::vector<std::vector<double> >()
      , boost::spirit::qi::space_type
    >
        file;

    boost::spirit::qi::rule<
        Iterator
      , std::vector<double>()
      , boost::spirit::qi::space_type
    >
        line;

    parser() : parser::base_type(file)
    {
        using boost::spirit::qi::double_;

        file = *line;

        line = double_ % ','; 
    }
};

parse_result parse(std::string const& filename, ast& a)
{
    boost::filesystem::path ip(filename);

    if (!boost::filesystem::exists(ip))
        return path_does_not_exist; 

    else if (boost::filesystem::is_directory(ip))
        return path_is_directory;

    std::ifstream ifs(filename.c_str(), std::ifstream::in);  

    // no white space skipping in the stream
    ifs.unsetf(std::ios::skipws);

    typedef boost::spirit::basic_istream_iterator<char> iterator;

    typedef parser<iterator> parser_type;

    iterator first(ifs), last;

    parser_type p;

    bool r = boost::spirit::qi::phrase_parse
        (first, last, p, boost::spirit::qi::space, a);

    if (r)
        return parse_succeeded;
    else
        return parse_failed;
} 

}}}

