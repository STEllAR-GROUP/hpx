//  deprecated_name_check implementation  -----------------------------------//

//  Copyright Beman Dawes   2002.
//  Copyright Gennaro Prota 2006.
//  Copyright Hartmut Kaiser 2016.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/util/to_string.hpp>

#include <algorithm>

#include "boost/regex.hpp"
#include "deprecated_name_check.hpp"
#include "function_hyper.hpp"

#include <set>
#include <string>
#include <vector>

namespace boost { namespace inspect {
    deprecated_names const names[] = {
        // boost::xyz
        {"(\\bboost\\s*::\\s*move\\b)", "std::move"},
        {"(\\bboost\\s*::\\s*forward\\b)", "std::forward"},
        {"(\\bboost\\s*::\\s*noncopyable\\b)", "HPX_NON_COPYABLE"},
        {"(\\bboost\\s*::\\s*result_of\\b)", "std::result_of"},
        {"(\\bboost\\s*::\\s*decay\\b)", "std::decay"},
        {"(\\bboost\\s*::\\s*enable_if\\b)", "std::enable_if"},
        {"(\\bboost\\s*::\\s*disable_if\\b)", "std::enable_if"},
        {"(\\bboost\\s*::\\s*enable_if_c\\b)", "std::enable_if"},
        {"(\\bboost\\s*::\\s*disable_if_c\\b)", "std::enable_if"},
        {"(\\bboost\\s*::\\s*lazy_enable_if\\b)", "hpx::util::lazy_enable_if"},
        {"(\\bboost\\s*::\\s*lazy_disable_if\\b)", "hpx::util::lazy_enable_if"},
        {"(\\bboost\\s*::\\s*lazy_enable_if_c\\b)",
            "hpx::util::lazy_enable_if"},
        {"(\\bboost\\s*::\\s*lazy_disable_if_c\\b)",
            "hpx::util::lazy_enable_if"},
        {"(\\bboost\\s*::\\s*mpl\\b)", "no specific replacement"},
        {"(\\bboost\\s*::\\s*(is_[^\\s]*?\\b))", "std::\\2"},
        {"(\\bboost\\s*::\\s*(add_[^\\s]*?\\b))", "std::\\2"},
        {"(\\bboost\\s*::\\s*(remove_[^\\s]*?\\b))", "std::\\2"},
        {"(\\bboost\\s*::\\s*(((false)|(true))_type\\b))", "std::\\2"},
        {"(\\bboost\\s*::\\s*lock_guard\\b)", "std::lock_guard"},
        {"(\\bboost\\s*::\\s*unordered_map\\b)", "std::unordered_map"},
        {"(\\bboost\\s*::\\s*unordered_multimap\\b)",
            "std::unordered_multimap"},
        {"(\\bboost\\s*::\\s*unordered_set\\b)", "std::unordered_set"},
        {"(\\bboost\\s*::\\s*unordered_multiset\\b)",
            "std::unordered_multiset"},
        {"(\\bboost\\s*::\\s*detail\\s*::\\s*atomic_count\\b)",
            "hpx::util::atomic_count"},
        {"(\\bboost\\s*::\\s*function\\b)", "hpx::function"},
        {R"((\bboost\s*::\s*intrusive_ptr\b))", "hpx::intrusive_ptr"},
        {"(\\bboost\\s*::\\s*shared_ptr\\b)", "std::shared_ptr"},
        {"(\\bboost\\s*::\\s*make_shared\\b)", "std::make_shared"},
        {"(\\bboost\\s*::\\s*enable_shared_from_this\\b)",
            "std::enable_shared_from_this"},
        {"(\\bboost\\s*::\\s*bind\\b)", "hpx::bind"},
        {"(\\bboost\\s*::\\s*unique_lock\\b)", "std::unique_lock"},
        {"(\\bboost\\s*::\\s*chrono\\b)", "std::chrono"},
        {"(\\bboost\\s*::\\s*reference_wrapper\\b)", "std::reference_wrapper"},
        {"(\\bboost\\s*::\\s*(c?ref)\\b)", "std::\\2"},
        {"(\\bboost\\s*::\\s*(u?int[0-9]+_t)\\b)", "std::\\2"},
        {"(\\bboost\\s*::\\s*exception_ptr\\b)", "std::exception_ptr"},
        {"(\\bboost\\s*::\\s*copy_exception\\b)", "std::make_exception_ptr"},
        {"(\\bboost\\s*::\\s*current_exception\\b)", "std::current_exception"},
        {"(\\bboost\\s*::\\s*rethrow_exception\\b)", "std::rethrow_exception"},
        {"(\\bboost\\s*::\\s*enable_error_info\\b)", "hpx::throw_with_info"},
        {"(\\bboost\\s*::\\s*iterator_range\\b)", "hpx::util::iterator_range"},
        {"(\\bboost\\s*::\\s*make_iterator_range\\b)",
            "hpx::util::make_iterator_range"},
        {"(\\bboost\\s*::\\s*atomic_flag\\b)", "std::atomic_flag"},
        {"(\\bboost\\s*::\\s*atomic\\b)", "std::atomic"},
        {"(\\bboost\\s*::\\s*memory_order_((relaxed)|(acquire)|(release)|"
         "(acq_rel)|(seq_cst))\\b)",
            "std::memory_order_\\2"},
        {"(\\bboost\\s*::\\s*random\\s*::\\s*([^\\s]*)\\b)", "std::\\2"},
        {"(\\bboost\\s*::\\s*format\\b)", "hpx::util::format[_to]"},
        {"(\\bboost\\s*::\\s*(regex[^\\s]*)\\b)", "std::\\2"},
        {"(\\bboost\\s*::\\s*lexical_cast\\b)",
            "hpx::util::((from_string)|(to_string))"},
        {"(\\bboost\\s*::\\s*system\\s*::\\s*error_code\\b)",
            "std::error_code"},
        {"(\\bboost\\s*::\\s*system\\s*::\\s*error_condition\\b)",
            "std::error_condition"},
        {"(\\bboost\\s*::\\s*system\\s*::\\s*error_category\\b)",
            "std::error_category"},
        {"(\\bboost\\s*::\\s*system\\s*::\\s*system_error\\b)",
            "std::system_error"},
        /////////////////////////////////////////////////////////////////////////
        {"((\\bhpx::\\b)?\\btraits\\s*::\\bis_callable\\b)",
            "\\2is_invocable[_r]"},
        {"((\\bhpx::\\b)?\\butil\\s*::\\bresult_of\\b)",
            "\\2util::invoke_result"},
        {"((\\bhpx::\\b)?\\butil\\s*::\\bdecay\\b)", "std::decay"},
        {"(\\bNULL\\b)", "nullptr"},
        // Boost preprocessor macros
        {"\\b(BOOST_PP_CAT)\\b", "HPX_PP_CAT"},
        {"\\b(BOOST_PP_STRINGIZE)\\b", "HPX_PP_STRINGIZE"},
        {"\\b(BOOST_STRINGIZE)\\b", "HPX_PP_STRINGIZE(HPX_PP_EXPAND())"},
        {"\\b(BOOST_ASSERT)\\b", "HPX_ASSERT"}, {nullptr, nullptr}};

    //  deprecated_name_check constructor  --------------------------------- //

    deprecated_name_check::deprecated_name_check()
      : m_errors(0)
    {
        // C/C++ source code...
        register_signature(".c");
        register_signature(".cpp");
        register_signature(".cu");
        register_signature(".cxx");
        register_signature(".h");
        register_signature(".hpp");
        register_signature(".hxx");
        register_signature(".inc");
        register_signature(".ipp");
        register_signature(".ixx");

        for (deprecated_names const* names_it = &names[0];
             names_it->name_regex != nullptr; ++names_it)
        {
            std::string rx(names_it->name_regex);
            rx += "|"    // or (ignored)
                  "("
                  "//[^\\n]*"    // single line comments (//)
                  "|"
                  "/\\*.*?\\*/"    // multi line comments (/**/)
                  "|"
                  "\"([^\"\\\\]|\\\\.)*\""    // string literals
                  ")";
            regex_data.push_back(deprecated_names_regex_data(names_it, rx));
        }
    }

    //  inspect ( C++ source files )  ---------------------------------------//

    void deprecated_name_check::inspect(const string& library_name,
        const path& full_path,     // example: c:/foo/boost/filesystem/path.hpp
        const string& contents)    // contents of file to be inspected
    {
        std::string::size_type p = contents.find("hpxinspect:"
                                                 "nodeprecatedname");
        if (p != string::npos)
        {
            // ignore this directive here (it is handled below) if it is followed
            // by a ':'
            if (p == contents.size() - 27 ||
                (contents.size() > p + 27 && contents[p + 27] != ':'))
            {
                return;
            }
        }

        std::set<std::string> found_names;

        // for all given names, check whether any is used
        for (deprecated_names_regex_data const& d : regex_data)
        {
            boost::sregex_iterator cur(
                contents.begin(), contents.end(), d.pattern),
                end;
            for (/**/; cur != end; ++cur)
            {
                auto m = *cur;
                if (m[1].matched)
                {
                    // avoid errors to be reported twice
                    std::string found_name(m[1].first, m[1].second);

                    if (found_names.find(found_name) == found_names.end())
                    {
                        std::string tag("hpxinspect:"
                                        "nodeprecatedname:" +
                            found_name);
                        if (contents.find(tag) != string::npos)
                            continue;

                        // name was found
                        found_names.insert(found_name);

                        auto it = contents.begin();
                        auto match_it = m[1].first;
                        auto line_start = it;

                        string::size_type line_number = 1;
                        for (/**/; it != match_it; ++it)
                        {
                            if (string::traits_type::eq(*it, '\n'))
                            {
                                ++line_number;
                                line_start = it + 1;    // could be end()
                            }
                        }

                        ++m_errors;
                        error(library_name, full_path,
                            string(name()) + " deprecated name (" + found_name +
                                ") on line " +
                                linelink(full_path,
                                    hpx::util::to_string(line_number)) +
                                ", use " + m.format(d.data->use_instead) +
                                " instead");
                    }
                }
            }
        }
    }

}}    // namespace boost::inspect
