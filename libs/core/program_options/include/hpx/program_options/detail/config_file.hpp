//  Copyright Vladimir Prus 2002-2004.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/program_options/config.hpp>
#include <hpx/program_options/detail/convert.hpp>
#include <hpx/program_options/eof_iterator.hpp>
#include <hpx/program_options/option.hpp>

#include <iosfwd>
#include <istream>    // std::getline
#include <memory>
#include <set>
#include <string>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::program_options::detail {

    /** Standalone parser for config files in ini-line format.
        The parser is a model of single-pass lvalue iterator, and
        default constructor creates past-the-end-iterator. The typical usage is:
        config_file_iterator i(is, ... set of options ...), e;
        for(; i !=e; ++i) {
            *i;
        }

        Syntax conventions:

        - config file can not contain positional options
        - '#' is comment character: it is ignored together with
          the rest of the line.
        - variable assignments are in the form
          name '=' value.
          spaces around '=' are trimmed.
        - Section names are given in brackets.

         The actual option name is constructed by combining current section
         name and specified option name, with dot between. If section_name
         already contains dot at the end, new dot is not inserted. For example:
         @verbatim
         [gui.accessibility]
         visual_bell=yes
         @endverbatim
         will result in option "gui.accessibility.visual_bell" with value
         "yes" been returned.

         TODO: maybe, we should just accept a pointer to options_description
         class.
     */
    class HPX_CORE_EXPORT common_config_file_iterator
      : public eof_iterator<common_config_file_iterator, option>
    {
    public:
        common_config_file_iterator()
          : m_allow_unregistered(false)
        {
            found_eof();
        }

        explicit common_config_file_iterator(
            std::set<std::string> const& allowed_options,
            bool allow_unregistered = false);

        common_config_file_iterator(
            common_config_file_iterator const&) = default;
        common_config_file_iterator(common_config_file_iterator&&) = default;
        common_config_file_iterator& operator=(
            common_config_file_iterator const&) = default;
        common_config_file_iterator& operator=(
            common_config_file_iterator&&) = default;

        virtual ~common_config_file_iterator() = default;

    public:    // Method required by eof_iterator
        void get();

#if defined(HPX_MSVC) && HPX_MSVC <= 1900
        constexpr void decrement() noexcept {}
        constexpr void advance(difference_type) noexcept {}
#endif

    protected:    // Stubs for derived classes
        // Obtains next line from the config file Note: really, this design is a
        // bit ugly The most clean thing would be to pass 'line_iterator' to
        // constructor of this class, but to avoid templating this class we'd
        // need polymorphic iterator, which does not exist yet.
        virtual bool getline(std::string&)
        {
            return false;
        }

    private:
        /// Adds another allowed option. If the 'name' ends with '*', then all
        /// options with the same prefix are allowed. For example, if 'name' is
        /// 'foo*', then 'foo1' and 'foo_bar' are allowed.
        ///
        void add_option(char const* name);

        // Returns true if 's' is a registered option name.
        [[nodiscard]] bool allowed_option(std::string const& s) const;

        // That's probably too much data for iterator, since
        // it will be copied, but let's not bother for now.
        std::set<std::string> allowed_options;
        // Invariant: no element is prefix of other element.
        std::set<std::string> allowed_prefixes;
        std::string m_prefix;
        bool m_allow_unregistered;
    };

    template <class Char>
    class basic_config_file_iterator : public common_config_file_iterator
    {
    public:
        basic_config_file_iterator()
        {
            found_eof();
        }

        /** Creates a config file parser for the specified stream.
        */
        basic_config_file_iterator(std::basic_istream<Char>& is,
            std::set<std::string> const& allowed_options,
            bool allow_unregistered = false);

    private:    // base overrides
        bool getline(std::string&) override;

    private:    // internal data
        std::shared_ptr<std::basic_istream<Char>> is;
    };

    using config_file_iterator = basic_config_file_iterator<char>;
    using wconfig_file_iterator = basic_config_file_iterator<wchar_t>;

    struct null_deleter
    {
        constexpr void operator()(void const*) const noexcept {}
    };

    template <class Char>
    basic_config_file_iterator<Char>::basic_config_file_iterator(
        std::basic_istream<Char>& is,
        std::set<std::string> const& allowed_options, bool allow_unregistered)
      : common_config_file_iterator(allowed_options, allow_unregistered)
    {
        this->is.reset(&is, null_deleter());
        get();
    }

    // Specializing this function for wchar_t causes problems on borland and
    // vc7, as well as on metrowerks. On the first two I don't know a
    // workaround, so make use of 'to_internal' to avoid specialization.
    template <typename Char>
    bool basic_config_file_iterator<Char>::getline(std::string& s)
    {
        std::basic_string<Char> in;
        if (std::getline(*is, in))
        {
            s = to_internal(in);
            return true;
        }
        return false;
    }
}    // namespace hpx::program_options::detail

#include <hpx/config/warnings_suffix.hpp>
