// Copyright Vladimir Prus 2004.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

// This file defines template functions that are declared in
// ../value_semantic.hpp.

#include <hpx/program_options/config.hpp>

#if defined(HPX_PROGRAM_OPTIONS_HAVE_BOOST_PROGRAM_OPTIONS_COMPATIBILITY)
// hpxinspect:nodeprecatedinclude:boost/program_options/detail/value_semantic.hpp

#include <boost/program_options/detail/value_semantic.hpp>

namespace hpx { namespace program_options {

    using boost::program_options::arg;

    namespace validators {

        using boost::program_options::validators::check_first_occurrence;
        using boost::program_options::validators::get_single_string;
    }    // namespace validators

    using namespace validators;

    using boost::program_options::validate;

}}    // namespace hpx::program_options

#else

#include <hpx/assert.hpp>
#include <hpx/datastructures/any.hpp>
#include <hpx/datastructures/optional.hpp>
#include <hpx/program_options/errors.hpp>
#include <hpx/util/from_string.hpp>

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

namespace hpx { namespace program_options {

    extern HPX_EXPORT std::string arg;

    template <class T, class Char>
    std::string typed_value<T, Char>::name() const
    {
        std::string const& var = (m_value_name.empty() ? arg : m_value_name);
        if (m_implicit_value.has_value() && !m_implicit_value_as_text.empty())
        {
            std::string msg =
                "[=" + var + "(=" + m_implicit_value_as_text + ")]";
            if (m_default_value.has_value() && !m_default_value_as_text.empty())
                msg += " (=" + m_default_value_as_text + ")";
            return msg;
        }
        else if (m_default_value.has_value() &&
            !m_default_value_as_text.empty())
        {
            return var + " (=" + m_default_value_as_text + ")";
        }
        else
        {
            return var;
        }
    }

    template <class T, class Char>
    void typed_value<T, Char>::notify(const hpx::any_nonser& value_store) const
    {
        const T* value = hpx::any_cast<T>(&value_store);
        if (m_store_to)
        {
            *m_store_to = *value;
        }
        if (m_notifier)
        {
            m_notifier(*value);
        }
    }

    namespace validators {

        /* If v.size() > 1, throw validation_error.
           If v.size() == 1, return v.front()
           Otherwise, returns a reference to a statically allocated
           empty string if 'allow_empty' and throws validation_error
           otherwise. */
        template <class Char>
        const std::basic_string<Char>& get_single_string(
            const std::vector<std::basic_string<Char>>& v,
            bool allow_empty = false)
        {
            static std::basic_string<Char> empty;
            if (v.size() > 1)
                throw validation_error(
                    validation_error::multiple_values_not_allowed);
            else if (v.size() == 1)
                return v.front();
            else if (!allow_empty)
                throw validation_error(
                    validation_error::at_least_one_value_required);
            return empty;
        }

        /* Throws multiple_occurrences if 'value' is not empty. */
        HPX_EXPORT void check_first_occurrence(const hpx::any_nonser& value);
    }    // namespace validators

    using namespace validators;

    /** Validates 's' and updates 'v'.
        @pre 'v' is either empty or in the state assigned by the previous
        invocation of 'validate'.
        The target type is specified via a parameter which has the type of
        pointer to the desired type. This is workaround for compilers without
        partial template ordering, just like the last 'long/int' parameter.
    */
    template <class T, class Char>
    void validate(hpx::any_nonser& v,
        const std::vector<std::basic_string<Char>>& xs, T*, long)
    {
        validators::check_first_occurrence(v);
        std::basic_string<Char> s(validators::get_single_string(xs));
        try
        {
            v = hpx::any_nonser(hpx::util::from_string<T>(s));
        }
        catch (const hpx::util::bad_lexical_cast&)
        {
            throw invalid_option_value(s);
        }
    }

    HPX_EXPORT void validate(
        hpx::any_nonser& v, const std::vector<std::string>& xs, bool*, int);

    HPX_EXPORT void validate(
        hpx::any_nonser& v, const std::vector<std::wstring>& xs, bool*, int);

    // For some reason, this declaration, which is require by the standard,
    // cause msvc 7.1 to not generate code to specialization defined in
    // value_semantic.cpp
    HPX_EXPORT void validate(hpx::any_nonser& v,
        const std::vector<std::string>& xs, std::string*, int);
    HPX_EXPORT void validate(hpx::any_nonser& v,
        const std::vector<std::wstring>& xs, std::string*, int);

    /** Validates sequences. Allows multiple values per option occurrence
       and multiple occurrences. */
    template <class T, class Char>
    void validate(hpx::any_nonser& v,
        const std::vector<std::basic_string<Char>>& s, std::vector<T>*, int)
    {
        if (!v.has_value())
        {
            v = hpx::any_nonser(std::vector<T>());
        }
        std::vector<T>* tv = hpx::any_cast<std::vector<T>>(&v);
        HPX_ASSERT(nullptr != tv);
        for (std::size_t i = 0; i < s.size(); ++i)
        {
            try
            {
                /* We call validate so that if user provided
                   a validator for class T, we use it even
                   when parsing vector<T>.  */
                hpx::any_nonser a;
                std::vector<std::basic_string<Char>> cv;
                cv.push_back(s[i]);
                validate(a, cv, (T*) nullptr, 0);
                tv->push_back(hpx::any_cast<T>(a));
            }
            catch (const hpx::util::bad_lexical_cast& /*e*/)
            {
                throw invalid_option_value(s[i]);
            }
        }
    }

    /** Validates optional arguments. */
    template <class T, class Char>
    void validate(hpx::any_nonser& v,
        const std::vector<std::basic_string<Char>>& s, hpx::util::optional<T>*,
        int)
    {
        validators::check_first_occurrence(v);
        validators::get_single_string(s);
        hpx::any_nonser a;
        validate(a, s, (T*) nullptr, 0);
        v = hpx::any_nonser(hpx::util::optional<T>(hpx::any_cast<T>(a)));
    }

    template <class T, class Char>
    void typed_value<T, Char>::xparse(hpx::any_nonser& value_store,
        const std::vector<std::basic_string<Char>>& new_tokens) const
    {
        // If no tokens were given, and the option accepts an implicit
        // value, then assign the implicit value as the stored value;
        // otherwise, validate the user-provided token(s).
        if (new_tokens.empty() && m_implicit_value.has_value())
            value_store = m_implicit_value;
        else
            validate(value_store, new_tokens, (T*) nullptr, 0);
    }

    template <class T>
    typed_value<T>* value()
    {
        // Explicit qualification is vc6 workaround.
        return hpx::program_options::value<T>(nullptr);
    }

    template <class T>
    typed_value<T>* value(T* v)
    {
        typed_value<T>* r = new typed_value<T>(v);

        return r;
    }

    template <class T>
    typed_value<T, wchar_t>* wvalue()
    {
        return wvalue<T>(nullptr);
    }

    template <class T>
    typed_value<T, wchar_t>* wvalue(T* v)
    {
        typed_value<T, wchar_t>* r = new typed_value<T, wchar_t>(v);

        return r;
    }

}}    // namespace hpx::program_options

#endif
