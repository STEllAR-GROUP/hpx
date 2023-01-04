//  Copyright Vladimir Prus 2002-2004.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/program_options/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/datastructures/any.hpp>
#include <hpx/program_options/options_description.hpp>
#include <hpx/program_options/parsers.hpp>
#include <hpx/program_options/value_semantic.hpp>
#include <hpx/program_options/variables_map.hpp>

#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace hpx::program_options {

    // First, performs semantic actions for 'oa'. Then, stores in 'm' all
    // options that are defined in 'desc'.
    HPX_CORE_EXPORT
    void store(parsed_options const& options, variables_map& xm, bool utf8)
    {
        // TODO: what if we have different definition for the same option name
        // during different calls 'store'.
        HPX_ASSERT(options.description);
        options_description const& desc = *options.description;

        // We need to access map's operator[], not the overridden version
        // variables_map. Ehmm.. messy.
        std::map<std::string, variable_value>& m = xm;

        std::set<std::string> new_final;

        // Declared once, to please Intel in VC++ mode;
        std::size_t i;

        // Declared here so can be used to provide context for exceptions
        std::string option_name;
        std::string original_token;

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

                // Ignore unregistered option. The 'unregistered' field can be
                // true only if user has explicitly asked to allow unregistered
                // options. We can't store them to variables map (lacking any
                // information about paring), so just ignore them.
                if (opts.unregistered)
                    continue;

                // If option has final value, skip this assignment
                if (xm.m_final.count(option_name))
                    continue;

                original_token = !opts.original_tokens.empty() ?
                    opts.original_tokens[0] :
                    "";
                option_description const& d =
                    desc.find(option_name, false, false, false);

                variable_value& v = m[option_name];
                if (v.defaulted())
                {
                    // Explicit assignment here erases defaulted value
                    v = variable_value();
                }

                auto const& semantic = d.semantic();
                semantic->parse(v.value(), opts.value, utf8);

                v.m_value_semantic = semantic;

                // The option is not composing, and the value is explicitly
                // provided. Ignore values of this option for subsequent calls
                // to 'store'. We store this to a temporary set, so that several
                // assignment inside *this* 'store' call are allowed.
                if (!semantic->is_composing())
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
        std::vector<std::shared_ptr<option_description>> const& all =
            desc.options();
        for (i = 0; i < all.size(); ++i)
        {
            option_description const& d = *all[i];
            std::string key = d.key("");
            // FIXME: this logic relies on knowledge of option_description
            // internals. The 'key' is empty if options description contains
            // '*'. In that case, default value makes no sense at all.
            if (key.empty())
            {
                continue;
            }
            if (m.count(key) == 0)
            {
                hpx::any_nonser def;
                if (d.semantic()->apply_default(def))
                {
                    m[key] = variable_value(def, true);
                    m[key].m_value_semantic = d.semantic();
                }
            }

            // add empty value if this is an required option
            if (d.semantic()->is_required())
            {
                // For option names specified in multiple ways, e.g. on the
                // command line, config file etc, the following precedence rules
                // apply: "--"  >  ("-" or "/")  >  ""
                // Precedence is set conveniently by a single call to length()
                std::string canonical_name =
                    d.canonical_display_name(options.m_options_prefix);
                if (canonical_name.length() > xm.m_required[key].length())
                    xm.m_required[key] = HPX_MOVE(canonical_name);
            }
        }
    }

    void store(wparsed_options const& options, variables_map& m)
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
        abstract_variables_map const* next)
      : m_next(next)
    {
    }

    variable_value const& abstract_variables_map::operator[](
        std::string const& name) const
    {
        variable_value const& v = get(name);
        if (v.empty() && m_next)
            return (*m_next)[name];
        else if (v.defaulted() && m_next)
        {
            variable_value const& v2 = (*m_next)[name];
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

    variables_map::variables_map(abstract_variables_map const* next)
      : abstract_variables_map(next)
    {
    }

    void variables_map::clear()
    {
        std::map<std::string, variable_value>::clear();
        m_final.clear();
        m_required.clear();
    }

    variable_value const& variables_map::get(std::string const& name) const
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
        for (std::map<std::string, std::string>::const_iterator r =
                 m_required.begin();
             r != m_required.end(); ++r)
        {
            std::string const& opt = r->first;
            std::string const& display_opt = r->second;
            std::map<std::string, variable_value>::const_iterator iter =
                find(opt);
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
}    // namespace hpx::program_options
