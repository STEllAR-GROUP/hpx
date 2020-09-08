// Copyright Vladimir Prus 2002-2004.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/program_options/config.hpp>

#if defined(HPX_PROGRAM_OPTIONS_HAVE_BOOST_PROGRAM_OPTIONS_COMPATIBILITY)
// hpxinspect:nodeprecatedinclude:boost/program_options/parsers.hpp
// hpxinspect:nodeprecatedinclude:boost/program_options/variables_map.hpp
// hpxinspect:nodeprecatedinclude:boost/program_options/value_semantic.hpp

#include <boost/program_options/parsers.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/variables_map.hpp>

namespace hpx { namespace program_options {

    using boost::program_options::abstract_variables_map;
    using boost::program_options::notify;
    using boost::program_options::store;
    using boost::program_options::variable_value;
    using boost::program_options::variables_map;

}}    // namespace hpx::program_options

#else

#include <hpx/datastructures/any.hpp>

#include <map>
#include <memory>
#include <set>
#include <string>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace program_options {

    template <class Char>
    class basic_parsed_options;

    class value_semantic;
    class variables_map;

    // forward declaration

    /** Stores in 'm' all options that are defined in 'options'.
        If 'm' already has a non-defaulted value of an option, that value
        is not changed, even if 'options' specify some value.
    */
    HPX_EXPORT
    void store(const basic_parsed_options<char>& options, variables_map& m,
        bool utf8 = false);

    /** Stores in 'm' all options that are defined in 'options'.
        If 'm' already has a non-defaulted value of an option, that value
        is not changed, even if 'options' specify some value.
        This is wide character variant.
    */
    HPX_EXPORT
    void store(const basic_parsed_options<wchar_t>& options, variables_map& m);

    /** Runs all 'notify' function for options in 'm'. */
    HPX_EXPORT void notify(variables_map& m);

    /** Class holding value of option. Contains details about how the
        value is set and allows to conveniently obtain the value.
    */
    class HPX_EXPORT variable_value
    {
    public:
        variable_value()
          : m_defaulted(false)
        {
        }
        variable_value(const hpx::util::any_nonser& xv, bool xdefaulted)
          : v(xv)
          , m_defaulted(xdefaulted)
        {
        }

        /** If stored value if of type T, returns that value. Otherwise,
            throws boost::bad_any_cast exception. */
        template <class T>
        const T& as() const
        {
            return hpx::util::any_cast<const T&>(v);
        }
        /** @overload */
        template <class T>
        T& as()
        {
            return hpx::util::any_cast<T&>(v);
        }

        /// Returns true if no value is stored.
        bool empty() const;
        /** Returns true if the value was not explicitly
            given, but has default value. */
        bool defaulted() const;
        /** Returns the contained value. */
        const hpx::util::any_nonser& value() const;

        /** Returns the contained value. */
        hpx::util::any_nonser& value();

    private:
        hpx::util::any_nonser v;
        bool m_defaulted;
        // Internal reference to value semantic. We need to run
        // notifications when *final* values of options are known, and
        // they are known only after all sources are stored. By that
        // time options_description for the first source might not
        // be easily accessible, so we need to store semantic here.
        std::shared_ptr<const value_semantic> m_value_semantic;

        friend HPX_EXPORT void store(
            const basic_parsed_options<char>& options, variables_map& m, bool);

        friend class HPX_EXPORT variables_map;
    };

    /** Implements string->string mapping with convenient value casting
        facilities. */
    class HPX_EXPORT abstract_variables_map
    {
    public:
        abstract_variables_map();
        abstract_variables_map(const abstract_variables_map* next);

        virtual ~abstract_variables_map() {}

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
        const variable_value& operator[](const std::string& name) const;

        /** Sets next variable map, which will be used to find
           variables not found in *this. */
        void next(abstract_variables_map* next);

    private:
        /** Returns value of variable 'name' stored in *this, or
            empty value otherwise. */
        virtual const variable_value& get(const std::string& name) const = 0;

        const abstract_variables_map* m_next;
    };

    /** Concrete variables map which store variables in real map.

        This class is derived from std::map<std::string, variable_value>,
        so you can use all map operators to examine its content.
    */
    class HPX_EXPORT variables_map
      : public abstract_variables_map
      , public std::map<std::string, variable_value>
    {
    public:
        variables_map();
        variables_map(const abstract_variables_map* next);

        // Resolve conflict between inherited operators.
        const variable_value& operator[](const std::string& name) const
        {
            return abstract_variables_map::operator[](name);
        }

        // Override to clear some extra fields.
        void clear();

        void notify();

    private:
        /** Implementation of abstract_variables_map::get
            which does 'find' in *this. */
        const variable_value& get(const std::string& name) const override;

        /** Names of option with 'final' values \-- which should not
            be changed by subsequence assignments. */
        std::set<std::string> m_final;

        friend HPX_EXPORT void store(const basic_parsed_options<char>& options,
            variables_map& xm, bool utf8);

        /** Names of required options, filled by parser which has
            access to options_description.
            The map values are the "canonical" names for each corresponding option.
            This is useful in creating diagnostic messages when the option is absent. */
        std::map<std::string, std::string> m_required;
    };

    /*
     * Templates/inlines
     */

    inline bool variable_value::empty() const
    {
        return !v.has_value();
    }

    inline bool variable_value::defaulted() const
    {
        return m_defaulted;
    }

    inline const hpx::util::any_nonser& variable_value::value() const
    {
        return v;
    }

    inline hpx::util::any_nonser& variable_value::value()
    {
        return v;
    }

}}    // namespace hpx::program_options

#include <hpx/config/warnings_suffix.hpp>

#endif
