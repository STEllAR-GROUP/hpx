//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/program_options/detail/cmdline.hpp>
#include <hpx/program_options/errors.hpp>

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace hpx::program_options {

    error::error(std::string const& xwhat)
      : std::logic_error(xwhat)
    {
    }

    too_many_positional_options_error::too_many_positional_options_error()
      : error("too many positional options have been specified on the "
              "command line")
    {
    }

    invalid_command_line_style::invalid_command_line_style(
        std::string const& msg)
      : error(msg)
    {
    }

    reading_file::reading_file(char const* filename)
      : error(std::string("can not read options configuration file '")
                .append(filename)
                .append("'"))
    {
    }

    error_with_option_name::error_with_option_name(std::string const& template_,
        std::string const& option_name, std::string const& original_token,
        int option_style)
      : error(template_)
      , m_option_style(option_style)
      , m_error_template(template_)
    {
        //     parameter            |     placeholder               |   value
        //     ---------            |     -----------               |   -----
        set_substitute_default(
            "canonical_option", "option '%canonical_option%'", "option");
        set_substitute_default("value", "argument ('%value%')", "argument");
        set_substitute_default("prefix", "%prefix%", "");
        m_substitutions["option"] = option_name;
        m_substitutions["original_token"] = original_token;
    }

    char const* error_with_option_name::what() const noexcept
    {
        // will substitute tokens each time what is run()
        substitute_placeholders(m_error_template);

        return m_message.c_str();
    }

    void error_with_option_name::replace_token(
        std::string const& from, std::string const& to) const
    {
        for (;;)
        {
            std::size_t const pos = m_message.find(from);
            // not found: all replaced
            if (pos == std::string::npos)
                return;
            m_message.replace(pos, from.length(), to);
        }
    }

    std::string error_with_option_name::get_canonical_option_prefix() const
    {
        switch (m_option_style)
        {
        case command_line_style::allow_dash_for_short:
            [[fallthrough]];
        case command_line_style::allow_long_disguise:
            return "-";
        case command_line_style::allow_slash_for_short:
            return "/";
        case command_line_style::allow_long:
            return "--";
        case 0:
            return "";
        default:
            break;
        }
        throw std::logic_error(
            "error_with_option_name::m_option_style can only be one of [0, "
            "allow_dash_for_short, allow_slash_for_short, allow_long_disguise "
            "or allow_long]");
    }

    inline std::string strip_prefixes(std::string const& text)
    {
        // "--foo-bar" -> "foo-bar"
        std::string::size_type const i = text.find_first_not_of("-/");
        if (i == std::string::npos)
        {
            return text;
        }
        return text.substr(i);
    }

    std::string error_with_option_name::get_canonical_option_name() const
    {
        auto const option_it = m_substitutions.find("option");
        auto const original_it = m_substitutions.find("original_token");
        if (option_it != m_substitutions.end() && !option_it->second.length())
        {
            return original_it != m_substitutions.end() ? original_it->second :
                                                          std::string();
        }

        std::string original_token;
        if (original_it != m_substitutions.end())
            original_token = strip_prefixes(original_it->second);

        std::string option_name;
        if (option_it != m_substitutions.end())
            option_name = strip_prefixes(option_it->second);

        //  For long options, use option name
        if (m_option_style == command_line_style::allow_long ||
            m_option_style == command_line_style::allow_long_disguise)
            return get_canonical_option_prefix() + option_name;

        //  For short options use first letter of original_token
        if (m_option_style && original_token.length())
            return get_canonical_option_prefix() + original_token[0];

        // no prefix
        return option_name;
    }

    void error_with_option_name::substitute_placeholders(
        std::string const& error_template) const
    {
        m_message = error_template;
        std::map<std::string, std::string> substitutions(m_substitutions);
        substitutions["canonical_option"] = get_canonical_option_name();
        substitutions["prefix"] = get_canonical_option_prefix();

        //
        //  replace placeholder with defaults if values are missing
        //
        for (auto const& substitution_default : m_substitution_defaults)
        {
            // missing parameter: use default
            if (substitutions.count(substitution_default.first) == 0 ||
                substitutions[substitution_default.first].length() == 0)
            {
                replace_token(substitution_default.second.first,
                    substitution_default.second.second);
            }
        }

        //
        //  replace placeholder with values
        //  placeholder are denoted by surrounding '%'
        //
        for (auto& substitution : substitutions)
            replace_token('%' + substitution.first + '%', substitution.second);
    }

    invalid_config_file_syntax::invalid_config_file_syntax(
        std::string const& invalid_line, kind_t kind)
      : invalid_syntax(kind)
    {
        m_substitutions["invalid_line"] = invalid_line;
    }

    /** Convenience functions for backwards compatibility */
    [[nodiscard]] std::string invalid_config_file_syntax::tokens() const
    {
        if (auto const it = m_substitutions.find("invalid_line");
            it != m_substitutions.end())
        {
            return it->second;
        }
        return "<unknown>";
    }

    multiple_values::multiple_values()
      : error_with_option_name(
            "option '%canonical_option%' only takes a single argument")
    {
    }

    multiple_occurrences::multiple_occurrences()
      : error_with_option_name("option '%canonical_option%' cannot be "
                               "specified more than once")
    {
    }

    required_option::required_option(std::string const& option_name)
      : error_with_option_name(
            "the option '%canonical_option%' is required but missing", "",
            option_name)
    {
    }

    error_with_no_option_name::error_with_no_option_name(
        std::string const& template_, std::string const& original_token)
      : error_with_option_name(template_, "", original_token)
    {
    }

    unknown_option::unknown_option(std::string const& original_token)
      : error_with_no_option_name(
            "unrecognized option '%canonical_option%'", original_token)
    {
    }

    ambiguous_option::ambiguous_option(std::vector<std::string> xalternatives)
      : error_with_no_option_name("option '%canonical_option%' is ambiguous")
      , m_alternatives(HPX_MOVE(xalternatives))
    {
    }

    invalid_syntax::invalid_syntax(kind_t kind, std::string const& option_name,
        std::string const& original_token, int option_style)
      : error_with_option_name(
            get_template(kind), option_name, original_token, option_style)
      , m_kind(kind)
    {
    }

    invalid_command_line_syntax::invalid_command_line_syntax(kind_t kind,
        std::string const& option_name, std::string const& original_token,
        int option_style)
      : invalid_syntax(kind, option_name, original_token, option_style)
    {
    }

    validation_error::validation_error(kind_t kind,
        std::string const& option_name, std::string const& original_token,
        int option_style)
      : error_with_option_name(
            get_template(kind), option_name, original_token, option_style)
      , m_kind(kind)
    {
    }
}    // namespace hpx::program_options
