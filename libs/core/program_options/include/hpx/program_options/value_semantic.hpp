//  Copyright Vladimir Prus 2004.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/program_options/config.hpp>
#include <hpx/datastructures/any.hpp>
#include <hpx/program_options/errors.hpp>
#include <hpx/util/to_string.hpp>

#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace hpx::program_options {

    /** Class which specifies how the option's value is to be parsed
        and converted into C++ types.
    */
    class HPX_CORE_EXPORT value_semantic
    {
    public:
        /** Returns the name of the option. The name is only meaningful
            for automatic help message.
         */
        virtual std::string name() const = 0;

        /** The minimum number of tokens for this option that
            should be present on the command line. */
        virtual unsigned min_tokens() const noexcept = 0;

        /** The maximum number of tokens for this option that
            should be present on the command line. */
        virtual unsigned max_tokens() const noexcept = 0;

        /** Returns true if values from different sources should be composed.
            Otherwise, value from the first source is used and values from
            other sources are discarded.
        */
        virtual bool is_composing() const = 0;

        /** Returns true if value must be given. Non-optional value

        */
        virtual bool is_required() const = 0;

        /** Parses a group of tokens that specify a value of option.
            Stores the result in 'value_store', using whatever representation
            is desired. May be be called several times if value of the same
            option is specified more than once.
        */
        virtual void parse(hpx::any_nonser& value_store,
            std::vector<std::string> const& new_tokens, bool utf8) const = 0;

        /** Called to assign default value to 'value_store'. Returns
            true if default value is assigned, and false if no default
            value exists. */
        virtual bool apply_default(hpx::any_nonser& value_store) const = 0;

        /** Called when final value of an option is determined.
        */
        virtual void notify(hpx::any_nonser const& value_store) const = 0;

        virtual ~value_semantic() {}
    };

    /** Helper class which perform necessary character conversions in the
        'parse' method and forwards the data further.
    */
    template <typename Char>
    class value_semantic_codecvt_helper
    {
        // Nothing here. Specializations to follow.
    };

    /** Helper conversion class for values that accept ascii
        strings as input.
        Overrides the 'parse' method and defines new 'xparse'
        method taking std::string. Depending on whether input
        to parse is ascii or UTF8, will pass it to xparse unmodified,
        or with UTF8->ascii conversion.
    */
    template <>
    class HPX_CORE_EXPORT value_semantic_codecvt_helper<char>
      : public value_semantic
    {
    private:    // base overrides
        void parse(hpx::any_nonser& value_store,
            std::vector<std::string> const& new_tokens,
            bool utf8) const override;

    protected:    // interface for derived classes.
        virtual void xparse(hpx::any_nonser& value_store,
            std::vector<std::string> const& new_tokens) const = 0;
    };

    /** Helper conversion class for values that accept ascii
        strings as input.
        Overrides the 'parse' method and defines new 'xparse'
        method taking std::wstring. Depending on whether input
        to parse is ascii or UTF8, will recode input to Unicode, or
        pass it unmodified.
    */
    template <>
    class HPX_CORE_EXPORT value_semantic_codecvt_helper<wchar_t>
      : public value_semantic
    {
    private:    // base overrides
        void parse(hpx::any_nonser& value_store,
            std::vector<std::string> const& new_tokens,
            bool utf8) const override;

    protected:    // interface for derived classes.
        virtual void xparse(hpx::any_nonser& value_store,
            std::vector<std::wstring> const& new_tokens) const = 0;
    };

    /** Class which specifies a simple handling of a value: the value will
        have string type and only one token is allowed. */
    class HPX_CORE_EXPORT untyped_value
      : public value_semantic_codecvt_helper<char>
    {
    public:
        explicit untyped_value(bool zero_tokens = false)
          : m_zero_tokens(zero_tokens)
        {
        }

        std::string name() const override;

        unsigned min_tokens() const noexcept override;
        unsigned max_tokens() const noexcept override;

        bool is_composing() const noexcept override
        {
            return false;
        }

        bool is_required() const noexcept override
        {
            return false;
        }

        /** If 'value_store' is already initialized, or new_tokens
            has more than one elements, throws. Otherwise, assigns
            the first string from 'new_tokens' to 'value_store', without
            any modifications.
         */
        void xparse(hpx::any_nonser& value_store,
            std::vector<std::string> const& new_tokens) const override;

        /** Does nothing. */
        bool apply_default(hpx::any_nonser&) const override
        {
            return false;
        }

        /** Does nothing. */
        void notify(hpx::any_nonser const&) const override {}

    private:
        bool m_zero_tokens;
    };

    /** Base class for all option that have a fixed type, and are
        willing to announce this type to the outside world.
        Any 'value_semantics' for which you want to find out the
        type can be dynamic_cast-ed to typed_value_base. If conversion
        succeeds, the 'type' method can be called.
    */
    class typed_value_base
    {
    public:
        // Returns the type of the value described by this
        // object.
        virtual std::type_info const& value_type() const = 0;
        // Not really needed, since deletion from this
        // class is silly, but just in case.
        virtual ~typed_value_base() = default;
    };

    /** Class which handles value of a specific type. */
    template <typename T, typename Char = char>
    class typed_value
      : public value_semantic_codecvt_helper<Char>
      , public typed_value_base
    {
    public:
        /** Ctor. The 'store_to' parameter tells where to store
            the value when it's known. The parameter can be NULL. */
        explicit typed_value(T* store_to)
          : m_store_to(store_to)
          , m_composing(false)
          , m_implicit(false)
          , m_multitoken(false)
          , m_zero_tokens(false)
          , m_required(false)
        {
        }

        /** Specifies default value, which will be used
            if none is explicitly specified. The type 'T' should
            provide operator<< for ostream.
        */
        typed_value* default_value(T const& v)
        {
            m_default_value = hpx::any_nonser(v);
            m_default_value_as_text = hpx::util::to_string(v);
            return this;
        }

        /** Specifies default value, which will be used
            if none is explicitly specified. Unlike the above overload,
            the type 'T' need not provide operator<< for ostream,
            but textual representation of default value must be provided
            by the user.
        */
        typed_value* default_value(T const& v, std::string const& textual)
        {
            m_default_value = hpx::any_nonser(v);
            m_default_value_as_text = textual;
            return this;
        }

        /** Specifies an implicit value, which will be used
            if the option is given, but without an adjacent value.
            Using this implies that an explicit value is optional,
        */
        typed_value* implicit_value(T const& v)
        {
            m_implicit_value = hpx::any_nonser(v);
            m_implicit_value_as_text = hpx::util::to_string(v);
            return this;
        }

        /** Specifies the name used to to the value in help message.  */
        typed_value* value_name(std::string const& name)
        {
            m_value_name = name;
            return this;
        }

        /** Specifies an implicit value, which will be used
            if the option is given, but without an adjacent value.
            Using this implies that an explicit value is optional, but if
            given, must be strictly adjacent to the option, i.e.: '-ovalue'
            or '--option=value'.  Giving '-o' or '--option' will cause the
            implicit value to be applied.
            Unlike the above overload, the type 'T' need not provide
            operator<< for ostream, but textual representation of default
            value must be provided by the user.
        */
        typed_value* implicit_value(T const& v, std::string const& textual)
        {
            m_implicit_value = hpx::any_nonser(v);
            m_implicit_value_as_text = textual;
            return this;
        }

        /** Specifies a function to be called when the final value
            is determined. */
        typed_value* notifier(std::function<void(T const&)> f)
        {
            m_notifier = f;
            return this;
        }

        /** Specifies that the value is composing. See the 'is_composing'
            method for explanation.
        */
        typed_value* composing() noexcept
        {
            m_composing = true;
            return this;
        }

        /** Specifies that the value can span multiple tokens.
        */
        typed_value* multitoken() noexcept
        {
            m_multitoken = true;
            return this;
        }

        /** Specifies that no tokens may be provided as the value of
            this option, which means that only presence of the option
            is significant. For such option to be useful, either the
            'validate' function should be specialized, or the
            'implicit_value' method should be also used. In most
            cases, you can use the 'bool_switch' function instead of
            using this method. */
        typed_value* zero_tokens() noexcept
        {
            m_zero_tokens = true;
            return this;
        }

        /** Specifies that the value must occur. */
        typed_value* required() noexcept
        {
            m_required = true;
            return this;
        }

    public:    // value semantic overrides
        std::string name() const override;

        bool is_composing() const noexcept override
        {
            return m_composing;
        }

        unsigned min_tokens() const noexcept override
        {
            if (m_zero_tokens || m_implicit_value.has_value())
            {
                return 0;
            }
            else
            {
                return 1;
            }
        }

        unsigned max_tokens() const noexcept override
        {
            if (m_multitoken)
            {
                return (std::numeric_limits<unsigned>::max)();
            }
            else if (m_zero_tokens)
            {
                return 0;
            }
            else
            {
                return 1;
            }
        }

        bool is_required() const noexcept override
        {
            return m_required;
        }

        /** Creates an instance of the 'validator' class and calls
            its operator() to perform the actual conversion. */
        void xparse(hpx::any_nonser& value_store,
            std::vector<std::basic_string<Char>> const& new_tokens)
            const override;

        /** If default value was specified via previous call to
            'default_value', stores that value into 'value_store'.
            Returns true if default value was stored.
        */
        bool apply_default(hpx::any_nonser& value_store) const override
        {
            if (!m_default_value.has_value())
            {
                return false;
            }

            value_store = m_default_value;
            return true;
        }

        /** If an address of variable to store value was specified
            when creating *this, stores the value there. Otherwise,
            does nothing. */
        void notify(hpx::any_nonser const& value_store) const override;

    public:    // typed_value_base overrides
        std::type_info const& value_type() const override
        {
            return typeid(T);
        }

    private:
        T* m_store_to;

        // Default value is stored as hpx::any_nonser and not
        // as boost::optional to avoid unnecessary instantiations.
        std::string m_value_name;
        hpx::any_nonser m_default_value;
        std::string m_default_value_as_text;
        hpx::any_nonser m_implicit_value;
        std::string m_implicit_value_as_text;
        bool m_composing, m_implicit, m_multitoken, m_zero_tokens, m_required;
        std::function<void(T const&)> m_notifier;
    };

    /** Creates a typed_value<T> instance. This function is the primary
        method to create value_semantic instance for a specific type, which
        can later be passed to 'option_description' constructor.
        The second overload is used when it's additionally desired to store the
        value of option into program variable.
    */
    template <typename T>
    typed_value<T>* value();

    /** @overload
    */
    template <typename T>
    typed_value<T>* value(T* v);

    /** Creates a typed_value<T> instance. This function is the primary
        method to create value_semantic instance for a specific type, which
        can later be passed to 'option_description' constructor.
    */
    template <typename T>
    typed_value<T, wchar_t>* wvalue();

    /** @overload
    */
    template <typename T>
    typed_value<T, wchar_t>* wvalue(T* v);

    /** Works the same way as the 'value<bool>' function, but the created
        value_semantic won't accept any explicit value. So, if the option
        is present on the command line, the value will be 'true'.
    */
    HPX_CORE_EXPORT typed_value<bool>* bool_switch();

    /** @overload
    */
    HPX_CORE_EXPORT typed_value<bool>* bool_switch(bool* v);
}    // namespace hpx::program_options

#include <hpx/program_options/detail/value_semantic.hpp>
