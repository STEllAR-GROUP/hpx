//  Copyright Vladimir Prus 2002-2004.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/program_options/config.hpp>
#include <hpx/program_options/cmdline.hpp>
#include <hpx/program_options/errors.hpp>
#include <hpx/program_options/option.hpp>
#include <hpx/program_options/options_description.hpp>
#include <hpx/program_options/positional_options.hpp>

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::program_options::detail {

    /** Command line parser class. Main requirements were:
        - Powerful enough to support all common uses.
        - Simple and easy to learn/use.
        - Minimal code size and external dependencies.
        - Extensible for custom syntaxes.

        First all options are registered. After that, elements of command line
        are extracted using operator++.

        For each element, user can find
        - if it's an option or an argument
        - name of the option
        - index of the option
        - option value(s), if any

        Sometimes the registered option name is not equal to the encountered
        one, for example, because name abbreviation is supported.  Therefore
        two option names can be obtained:
        - the registered one
        - the one found at the command line

        There are lot of style options, which can be used to tune the command
        line parsing. In addition, it's possible to install additional parser
        which will process custom option styles.

        @todo minimal match length for guessing?
    */
    class HPX_CORE_EXPORT cmdline
    {
    public:
        using style_t = ::hpx::program_options::command_line_style::style_t;

        using additional_parser =
            std::function<std::pair<std::string, std::string>(
                std::string const&)>;

        using style_parser =
            std::function<std::vector<option>(std::vector<std::string>&)>;

        /// Constructs a command line parser for (argc, argv) pair. Uses style
        /// options passed in 'style', which should be binary or'ed values of
        /// style_t enum. It can also be zero, in which case a "default" style
        /// will be used. If 'allow_unregistered' is true, then allows
        /// unregistered options. They will be assigned index 1 and are assumed
        /// to have optional parameter.
        ///
        explicit cmdline(std::vector<std::string> const& args);

        /** @overload */
        cmdline(int argc, char const* const* argv);

        void style(int style) noexcept;

        /**
         * returns the canonical option prefix associated with the
         * command_line_style
         *  In order of precedence:
         *      allow_long           : allow_long
         *      allow_long_disguise  : allow_long_disguise
         *      allow_dash_for_short : allow_short | allow_dash_for_short
         *      allow_slash_for_short: allow_short | allow_slash_for_short
         *
         *      This is mainly used for the diagnostic messages in exceptions
         */
        [[nodiscard]] int get_canonical_option_prefix() const noexcept;

        void allow_unregistered() noexcept;

        void set_options_description(options_description const& desc) noexcept;
        void set_positional_options(
            positional_options_description const& m_positional) noexcept;

        std::vector<option> run();

        std::vector<option> parse_long_option(
            std::vector<std::string>& args) const;
        std::vector<option> parse_short_option(
            std::vector<std::string>& args) const;
        std::vector<option> parse_dos_option(
            std::vector<std::string>& args) const;
        std::vector<option> parse_disguised_long_option(
            std::vector<std::string>& args) const;
        static std::vector<option> parse_terminator(
            std::vector<std::string>& args);
        std::vector<option> handle_additional_parser(
            std::vector<std::string>& args) const;

        /// Set additional parser. This will be called for each token of command
        /// line. If first string in pair is not empty, then the token is
        /// considered matched by this parser, and the first string will be
        /// considered an option name (which can be long or short), while the
        /// second will be option's parameter (if not empty). Note that
        /// additional parser can match only one token.
        void set_additional_parser(additional_parser p) noexcept;

        void extra_style_parser(style_parser s) noexcept;

        static void check_style(int style);

        [[nodiscard]] bool is_style_active(style_t style) const noexcept;

        void init(std::vector<std::string> const& args);

        void finish_option(option& opt, std::vector<std::string>& other_tokens,
            std::vector<style_parser> const& style_parsers) const;

        // Copies of input.
        std::vector<std::string> m_args;
        style_t m_style = command_line_style::default_style;
        bool m_allow_unregistered = false;

        options_description const* m_desc = nullptr;
        positional_options_description const* m_positional = nullptr;

        additional_parser m_additional_parser;
        style_parser m_style_parser;
    };
}    // namespace hpx::program_options::detail

#include <hpx/config/warnings_suffix.hpp>
