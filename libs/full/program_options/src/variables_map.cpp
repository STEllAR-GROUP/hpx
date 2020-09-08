//  Copyright Vladimir Prus 2002-2004.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/program_options/config.hpp>

#if !defined(HPX_PROGRAM_OPTIONS_HAVE_BOOST_PROGRAM_OPTIONS_COMPATIBILITY)
#include <hpx/assert.hpp>
#include <hpx/datastructures/any.hpp>
#include <hpx/program_options/options_description.hpp>
#include <hpx/program_options/parsers.hpp>
#include <hpx/program_options/value_semantic.hpp>
#include <hpx/program_options/variables_map.hpp>

#include <cstddef>
#include <map>
#include <set>
#include <string>

namespace hpx { namespace program_options {

    using namespace std;

    // First, performs semantic actions for 'oa'.
    // Then, stores in 'm' all options that are defined in 'desc'.
    HPX_EXPORT
    void store(const parsed_options& options, variables_map& xm, bool utf8)
    {
        // TODO: what if we have different definition
        // for the same option name during different calls
        // 'store'.
        HPX_ASSERT(options.description);
        const options_description& desc = *options.description;

        // We need to access map's operator[], not the overridden version
        // variables_map. Ehmm.. messy.
        std::map<std::string, variable_value>& m = xm;

        std::set<std::string> new_final;

        // Declared once, to please Intel in VC++ mode;
        std::size_t i;

        // Declared here so can be used to provide context for exceptions
        string option_name;
        string original_token;

        try
        {
            // First, convert/store all given options
            for (i = 0; i < options.options.size(); ++i)
            {
                auto const& opts = options.options[i];
                option_name = opts.string_key;
                // Skip positional options without name
                if (option_name.empty())
                    continue;

                // Ignore unregistered option. The 'unregistered'
                // field can be true only if user has explicitly asked
                // to allow unregistered options. We can't store them
                // to variables map (lacking any information about paring),
                // so just ignore them.
                if (opts.unregistered)
                    continue;

                // If option has final value, skip this assignment
                if (xm.m_final.count(option_name))
                    continue;

                original_token = !opts.original_tokens.empty() ?
                    opts.original_tokens[0] :
                    "";
                const option_description& d =
                    desc.find(option_name, false, false, false);

                variable_value& v = m[option_name];
                if (v.defaulted())
                {
                    // Explicit assignment here erases defaulted value
                    v = variable_value();
                }

                d.semantic()->parse(v.value(), opts.value, utf8);

                v.m_value_semantic = d.semantic();

                // The option is not composing, and the value is explicitly
                // provided. Ignore values of this option for subsequent
                // calls to 'store'. We store this to a temporary set,
                // so that several assignment inside *this* 'store' call
                // are allowed.
                if (!d.semantic()->is_composing())
                    new_final.insert(option_name);
            }
        }
        catch (error_with_option_name& e)
        {
            // add context and rethrow
            e.add_context(
                option_name, original_token, options.m_options_prefix);
            throw;
        }

        xm.m_final.insert(new_final.begin(), new_final.end());

        // Second, apply default values and store required options.
        const vector<shared_ptr<option_description>>& all = desc.options();
        for (i = 0; i < all.size(); ++i)
        {
            const option_description& d = *all[i];
            string key = d.key("");
            // FIXME: this logic relies on knowledge of option_description
            // internals.
            // The 'key' is empty if options description contains '*'.
            // In that
            // case, default value makes no sense at all.
            if (key.empty())
            {
                continue;
            }
            if (m.count(key) == 0)
            {
                hpx::util::any_nonser def;
                if (d.semantic()->apply_default(def))
                {
                    m[key] = variable_value(def, true);
                    m[key].m_value_semantic = d.semantic();
                }
            }

            // add empty value if this is an required option
            if (d.semantic()->is_required())
            {
                // For option names specified in multiple ways, e.g. on the command line,
                // config file etc, the following precedence rules apply:
                //  "--"  >  ("-" or "/")  >  ""
                //  Precedence is set conveniently by a single call to length()
                string canonical_name =
                    d.canonical_display_name(options.m_options_prefix);
                if (canonical_name.length() > xm.m_required[key].length())
                    xm.m_required[key] = canonical_name;
            }
        }
    }

    void store(const wparsed_options& options, variables_map& m)
    {
        store(options.utf8_encoded_options, m, true);
    }

    void notify(variables_map& vm)
    {
        vm.notify();
    }

    abstract_variables_map::abstract_variables_map()
      : m_next(nullptr)
    {
    }

    abstract_variables_map::abstract_variables_map(
        const abstract_variables_map* next)
      : m_next(next)
    {
    }

    const variable_value& abstract_variables_map::operator[](
        const std::string& name) const
    {
        const variable_value& v = get(name);
        if (v.empty() && m_next)
            return (*m_next)[name];
        else if (v.defaulted() && m_next)
        {
            const variable_value& v2 = (*m_next)[name];
            if (!v2.empty() && !v2.defaulted())
                return v2;
            else
                return v;
        }
        else
        {
            return v;
        }
    }

    void abstract_variables_map::next(abstract_variables_map* next)
    {
        m_next = next;
    }

    variables_map::variables_map() {}

    variables_map::variables_map(const abstract_variables_map* next)
      : abstract_variables_map(next)
    {
    }

    void variables_map::clear()
    {
        std::map<std::string, variable_value>::clear();
        m_final.clear();
        m_required.clear();
    }

    const variable_value& variables_map::get(const std::string& name) const
    {
        static variable_value empty;
        const_iterator i = this->find(name);
        if (i == this->end())
            return empty;
        else
            return i->second;
    }

    void variables_map::notify()
    {
        // This checks if all required options occur
        for (map<string, string>::const_iterator r = m_required.begin();
             r != m_required.end(); ++r)
        {
            const string& opt = r->first;
            const string& display_opt = r->second;
            map<string, variable_value>::const_iterator iter = find(opt);
            if (iter == end() || iter->second.empty())
            {
                throw required_option(display_opt);
            }
        }

        // Lastly, run notify actions.
        for (auto& k : *this)
        {
            /* Users might wish to use variables_map to store their own values
               that are not parsed, and therefore will not have value_semantics
               defined. Do not crash on such values. In multi-module programs,
               one module might add custom values, and the 'notify' function
               will be called after that, so we check that value_sematics is
               not NULL. See:
                   https://svn.boost.org/trac/boost/ticket/2782
            */
            if (k.second.m_value_semantic)
                k.second.m_value_semantic->notify(k.second.value());
        }
    }

}}    // namespace hpx::program_options

#endif
