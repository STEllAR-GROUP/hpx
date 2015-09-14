////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/exception.hpp>
#include <hpx/util/sed_transform.hpp>
#include <boost/regex.hpp>
#include <boost/make_shared.hpp>

namespace hpx { namespace util
{

bool parse_sed_expression(
    std::string const& input
  , std::string& search
  , std::string& replace
    )
{
    if (input.size() <= 2)
        return false;

    // s/search/replace/
    // ^^
    if ('s' != input.at(0) || '/' != input.at(1))
        return false;

    std::string::size_type const search_begin = 2;
    std::string::size_type search_end = search_begin;

    // s/search/replace/
    //   ^^^^^^
    while (true)
    {
        // See if we're at the end of the string.
        if ((input.size() - 1) < search_end)
            return false;

        // If this is an escape, then don't check the next character.
        if ('\\' == input.at(search_end++))
        {
            // See if we're at the end of the string.
            if ((input.size() - 1) < ++search_end)
                return false;
        }

        if ('/' == input.at(search_end))
        {
            --search_end;
            break;
        }
    }

    // s/search/replace/
    //         ^
    if ('/' != input.at(search_end + 1))
        return false;

    std::string::size_type const replace_begin = search_end + 2;
    std::string::size_type replace_end = input.size() - 1;

    // s/search/replace/
    //                 ^ (optional)
    if ('/' == input.at(replace_end))
        --replace_end;

    search = input.substr(search_begin, (search_end - search_begin) + 1);
    replace = input.substr(replace_begin, (replace_end - replace_begin) + 1);

    return true;
}

struct sed_transform::command
{
    command(
        std::string const& search
      , std::string const& replace
        )
      : search_(search)
      , replace_(replace)
    {}

    boost::regex search_;
    std::string replace_;
};

sed_transform::sed_transform(
    std::string const& search
  , std::string const& replace
    )
  : command_(boost::make_shared<command>(search, replace))
{}

sed_transform::sed_transform(
    std::string const& expression
    )
{
    std::string search, replace;

    if (parse_sed_expression(expression, search, replace))
        command_ = boost::make_shared<command>(search, replace);
}

std::string sed_transform::operator()(
    std::string const& input
    ) const
{
    if (!command_)
        return input;

    return boost::regex_replace(input
                              , command_->search_
                              , command_->replace_
                              , boost::match_default | boost::format_sed);
}

}}


