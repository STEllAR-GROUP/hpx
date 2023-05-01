//  Copyright Vladimir Prus 2002-2004.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/program_options/config.hpp>
#include <hpx/program_options/detail/cmdline.hpp>
#include <hpx/program_options/option.hpp>

#include <functional>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::program_options {

    class options_description;
    class positional_options_description;

    /** Results of parsing an input source.
        The primary use of this class is passing information from parsers
        component to value storage component. This class does not makes
        much sense itself.
    */
    template <typename Char>
    class basic_parsed_options
    {
    public:
        explicit basic_parsed_options(
            options_description const* xdescription, int options_prefix = 0)
          : description(xdescription)
          , m_options_prefix(options_prefix)
        {
        }

        /** Options found in the source. */
        std::vector<basic_option<Char>> options;
        /** Options description that was used for parsing.
            Parsers should return pointer to the instance of
            option_description passed to them, and issues of lifetime are
            up to the caller. Can be NULL.
         */
        options_description const* description;

        /** Mainly used for the diagnostic messages in exceptions.
         *  The canonical option prefix  for the parser which generated these results,
         *  depending on the settings for basic_command_line_parser::style() or
         *  cmdline::style(). In order of precedence of command_line_style enums:
         *      allow_long
         *      allow_long_disguise
         *      allow_dash_for_short
         *      allow_slash_for_short
        */
        int m_options_prefix;
    };

    /** Specialization of basic_parsed_options which:
        - provides convenient conversion from basic_parsed_options<char>
        - stores the passed char-based options for later use.
    */
    template <>
    class HPX_CORE_EXPORT basic_parsed_options<wchar_t>
    {
    public:
        /** Constructs wrapped options from options in UTF8 encoding. */
        explicit basic_parsed_options(basic_parsed_options<char> const& po);

        std::vector<basic_option<wchar_t>> options;
        options_description const* description;

        /** Stores UTF8 encoded options that were passed to constructor,
            to avoid reverse conversion in some cases. */
        basic_parsed_options<char> utf8_encoded_options;

        /** Mainly used for the diagnostic messages in exceptions.
         *  The canonical option prefix  for the parser which generated these results,
         *  depending on the settings for basic_command_line_parser::style() or
         *  cmdline::style(). In order of precedence of command_line_style enums:
         *      allow_long
         *      allow_long_disguise
         *      allow_dash_for_short
         *      allow_slash_for_short
        */
        int m_options_prefix;
    };

    using parsed_options = basic_parsed_options<char>;
    using wparsed_options = basic_parsed_options<wchar_t>;

    /** Augments basic_parsed_options<wchar_t> with conversion from
        'parsed_options' */

    using ext_parser =
        std::function<std::pair<std::string, std::string>(std::string const&)>;

    /** Command line parser.

        The class allows one to specify all the information needed for parsing
        and to parse the command line. It is primarily needed to
        emulate named function parameters \-- a regular function with 5
        parameters will be hard to use and creating overloads with a smaller
        number of parameters will be confusing.

        For the most common case, the function parse_command_line is a better
        alternative.

        There are two typedefs \-- command_line_parser and wcommand_line_parser,
        for charT == char and charT == wchar_t cases.
    */
    template <typename Char>
    class basic_command_line_parser : private detail::cmdline
    {
    public:
        /** Creates a command line parser for the specified arguments
            list. The 'args' parameter should not include program name.
        */
        basic_command_line_parser(
            std::vector<std::basic_string<Char>> const& args);
        /** Creates a command line parser for the specified arguments
            list. The parameters should be the same as passed to 'main'.
        */
        basic_command_line_parser(int argc, Char const* const argv[]);

        /** Sets options descriptions to use. */
        basic_command_line_parser& options(options_description const& desc);
        /** Sets positional options description to use. */
        basic_command_line_parser& positional(
            positional_options_description const& desc);

        /** Sets the command line style. */
        basic_command_line_parser& style(int);
        /** Sets the extra parsers. */
        basic_command_line_parser& extra_parser(ext_parser);

        /** Parses the options and returns the result of parsing.
            Throws on error.
        */
        basic_parsed_options<Char> run();

        /** Specifies that unregistered options are allowed and should
            be passed though. For each command like token that looks
            like an option but does not contain a recognized name, an
            instance of basic_option<charT> will be added to result,
            with 'unrecognized' field set to 'true'. It's possible to
            collect all unrecognized options with the 'collect_unrecognized'
            function.
        */
        basic_command_line_parser& allow_unregistered();

        using detail::cmdline::style_parser;

        basic_command_line_parser& extra_style_parser(style_parser s);
    };

    using command_line_parser = basic_command_line_parser<char>;
    using wcommand_line_parser = basic_command_line_parser<wchar_t>;

    /** Creates instance of 'command_line_parser', passes parameters to it,
        and returns the result of calling the 'run' method.
     */
    template <typename Char>
    [[nodiscard]] basic_parsed_options<Char> parse_command_line(int argc,
        Char const* const argv[], options_description const&, int style = 0,
        std::function<std::pair<std::string, std::string>(std::string const&)>
            ext = ext_parser());

    /** Parse a config file.

        Read from given stream.
    */
    template <typename Char>
    [[nodiscard]] HPX_CORE_EXPORT basic_parsed_options<Char> parse_config_file(
        std::basic_istream<Char>&, options_description const&,
        bool allow_unregistered = false);

    /** Parse a config file.

        Read from file with the given name. The character type is
        passed to the file stream.
    */
    template <typename Char = char>
    [[nodiscard]] HPX_CORE_EXPORT basic_parsed_options<Char> parse_config_file(
        char const* filename, options_description const&,
        bool allow_unregistered = false);

    /** Controls if the 'collect_unregistered' function should
        include positional options, or not. */
    enum collect_unrecognized_mode
    {
        include_positional,
        exclude_positional
    };

    /** Collects the original tokens for all named options with
        'unregistered' flag set. If 'mode' is 'include_positional'
        also collects all positional options.
        Returns the vector of original tokens for all collected
        options.
    */
    template <typename Char>
    [[nodiscard]] std::vector<std::basic_string<Char>> collect_unrecognized(
        std::vector<basic_option<Char>> const& options,
        enum collect_unrecognized_mode mode);

    /** Parse environment.

        For each environment variable, the 'name_mapper' function is called to
        obtain the option name. If it returns empty string, the variable is
        ignored.

        This is done since naming of environment variables is typically
        different from the naming of command line options.
    */
    [[nodiscard]] HPX_CORE_EXPORT parsed_options parse_environment(
        options_description const&,
        std::function<std::string(std::string)> const& name_mapper);

    /** Parse environment.

        Takes all environment variables which start with 'prefix'. The option
        name is obtained from variable name by removing the prefix and
        converting the remaining string into lower case.
    */
    [[nodiscard]] HPX_CORE_EXPORT parsed_options parse_environment(
        options_description const&, std::string const& prefix);

    /** @overload
        This function exists to resolve ambiguity between the two above
        functions when second argument is of 'char*' type. There's implicit
        conversion to both std::function and string.
    */
    [[nodiscard]] HPX_CORE_EXPORT parsed_options parse_environment(
        options_description const&, char const* prefix);

    /** Splits a given string to a collection of single strings which
        can be passed to command_line_parser. The second parameter is
        used to specify a collection of possible separator chars used
        for splitting. The separator is defaulted to space " ".
        Splitting is done in a unix style way, with respect to quotes '"'
        and escape characters '\'
    */
    [[nodiscard]] HPX_CORE_EXPORT std::vector<std::string> split_unix(
        std::string const& cmdline, std::string const& separator = " \t",
        std::string const& quote = "'\"", std::string const& escape = "\\");

    /** @overload */
    [[nodiscard]] HPX_CORE_EXPORT std::vector<std::wstring> split_unix(
        std::wstring const& cmdline, std::wstring const& separator = L" \t",
        std::wstring const& quote = L"'\"", std::wstring const& escape = L"\\");

#ifdef HPX_WINDOWS
    /** Parses the char* string which is passed to WinMain function on
        windows. This function is provided for convenience, and because it's
        not clear how to portably access split command line string from
        runtime library and if it always exists.
        This function is available only on Windows.
    */
    [[nodiscard]] HPX_CORE_EXPORT std::vector<std::string> split_winmain(
        std::string const& cmdline);

    /** @overload */
    [[nodiscard]] HPX_CORE_EXPORT std::vector<std::wstring> split_winmain(
        std::wstring const& cmdline);
#endif
}    // namespace hpx::program_options

#include <hpx/config/warnings_suffix.hpp>

#include <hpx/program_options/detail/parsers.hpp>
