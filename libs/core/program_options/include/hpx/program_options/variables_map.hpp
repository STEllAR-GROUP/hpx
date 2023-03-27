//  Copyright Vladimir Prus 2002-2004.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/program_options/config.hpp>
#include <hpx/datastructures/any.hpp>

#include <map>
#include <memory>
#include <set>
#include <string>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::program_options {

    template <typename Char>
    class basic_parsed_options;

    class value_semantic;
    class variables_map;

    // forward declaration

    /** Stores in 'm' all options that are defined in 'options'.
        If 'm' already has a non-defaulted value of an option, that value
        is not changed, even if 'options' specify some value.
    */
    HPX_CORE_EXPORT
    void store(basic_parsed_options<char> const& options, variables_map& m,
        bool utf8 = false);

    /** Stores in 'm' all options that are defined in 'options'.
        If 'm' already has a non-defaulted value of an option, that value
        is not changed, even if 'options' specify some value.
        This is wide character variant.
    */
    HPX_CORE_EXPORT
    void store(basic_parsed_options<wchar_t> const& options, variables_map& m);

    /** Runs all 'notify' function for options in 'm'. */
    HPX_CORE_EXPORT void notify(variables_map& m);

    /** Class holding value of option. Contains details about how the
        value is set and allows to conveniently obtain the value.
    */
    class HPX_CORE_EXPORT variable_value
    {
    public:
        variable_value() = default;
        variable_value(hpx::any_nonser const& xv, bool xdefaulted)
          : v(xv)
          , m_defaulted(xdefaulted)
        {
        }

        /** If stored value if of type T, returns that value. Otherwise,
            throws boost::bad_any_cast exception. */
        template <class T>
        T const& as() const
        {
            return hpx::any_cast<T const&>(v);
        }
        /** @overload */
        template <class T>
        T& as()
        {
            return hpx::any_cast<T&>(v);
        }

        /// Returns true if no value is stored.
        bool empty() const noexcept;
        /** Returns true if the value was not explicitly
            given, but has default value. */
        bool defaulted() const noexcept;
        /** Returns the contained value. */
        hpx::any_nonser const& value() const noexcept;

        /** Returns the contained value. */
        hpx::any_nonser& value() noexcept;

    private:
        hpx::any_nonser v;
        bool m_defaulted = false;
        // Internal reference to value semantic. We need to run
        // notifications when *final* values of options are known, and
        // they are known only after all sources are stored. By that
        // time options_description for the first source might not
        // be easily accessible, so we need to store semantic here.
        std::shared_ptr<value_semantic const> m_value_semantic;

        friend HPX_CORE_EXPORT void store(
            basic_parsed_options<char> const& options, variables_map& m, bool);

        friend class HPX_CORE_EXPORT variables_map;
    };

    /** Implements string->string mapping with convenient value casting
        facilities. */
    class HPX_CORE_EXPORT abstract_variables_map
    {
    public:
        abstract_variables_map();
        explicit abstract_variables_map(abstract_variables_map const* next);

        virtual ~abstract_variables_map() = default;

        /** Obtains the value of variable 'name', from *this and
            possibly from the chain of variable maps.

            - if there's no value in *this.
                - if there's next variable map, returns value from it
                - otherwise, returns empty value

            - if there's defaulted value
                - if there's next variable map, which has a non-defaulted
                  value, return that
                - otherwise, return value from *this

            - if there's a non-defaulted value, returns it.
        */
        virtual variable_value const& operator[](std::string const& name) const;

        /** Sets next variable map, which will be used to find
           variables not found in *this. */
        void next(abstract_variables_map* next);

    private:
        /** Returns value of variable 'name' stored in *this, or
            empty value otherwise. */
        virtual variable_value const& get(std::string const& name) const = 0;

        abstract_variables_map const* m_next;
    };

    /** Concrete variables map which store variables in real map.

        This class is derived from std::map<std::string, variable_value>,
        so you can use all map operators to examine its content.
    */
    class HPX_CORE_EXPORT variables_map
      : public abstract_variables_map
      , public std::map<std::string, variable_value>
    {
    public:
        variables_map();
        explicit variables_map(abstract_variables_map const* next);

        // Resolve conflict between inherited operators.
        variable_value const& operator[](std::string const& name) const override
        {
            return abstract_variables_map::operator[](name);
        }

        // Override to clear some extra fields.
        void clear();

        void notify();

    private:
        /** Implementation of abstract_variables_map::get
            which does 'find' in *this. */
        variable_value const& get(std::string const& name) const override;

        /** Names of option with 'final' values \-- which should not
            be changed by subsequence assignments. */
        std::set<std::string> m_final;

        friend HPX_CORE_EXPORT void store(
            basic_parsed_options<char> const& options, variables_map& xm,
            bool utf8);

        /** Names of required options, filled by parser which has
            access to options_description.
            The map values are the "canonical" names for each corresponding option.
            This is useful in creating diagnostic messages when the option is absent. */
        std::map<std::string, std::string> m_required;
    };

    /*
     * Templates/inlines
     */
    [[nodiscard]] inline bool variable_value::empty() const noexcept
    {
        return !v.has_value();
    }

    [[nodiscard]] inline bool variable_value::defaulted() const noexcept
    {
        return m_defaulted;
    }

    [[nodiscard]] inline hpx::any_nonser const& variable_value::value()
        const noexcept
    {
        return v;
    }

    [[nodiscard]] inline hpx::any_nonser& variable_value::value() noexcept
    {
        return v;
    }
}    // namespace hpx::program_options

#include <hpx/config/warnings_suffix.hpp>
