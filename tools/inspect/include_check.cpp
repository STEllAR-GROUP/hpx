//  include_check implementation  --------------------------------------------//

//  Copyright Beman Dawes   2002.
//  Copyright Gennaro Prota 2006.
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


#include <algorithm>

#include "include_check.hpp"
#include "boost/regex.hpp"
#include "boost/lexical_cast.hpp"
#include "function_hyper.hpp"

namespace boost
{
  namespace inspect
  {
    boost::regex include_regex(
      "^\\s*#\\s*include\\s*<([^\n>]*)>(\\s|//[^\\n]*|/\\*.*?\\*/)*$"     // # include <foobar>
      "|"
      "^\\s*#\\s*include\\s*\"([^\n\"]*)\"(\\s|//[^\\n]*|/\\*.*?\\*/)*$"  // # include "foobar"
      , boost::regex::normal);

    names_includes const names[] =
    {
      { "(\\bstd\\s*::\\s*make_shared\\b)", "std::make_shared", "memory" },
      { "(\\bstd\\s*::\\s*((map)|(set))\\b)", "std::\\2", "\\2" },
      { "(\\bstd\\s*::\\s*(multi((map)|(set)))\\b)", "std::\\2", "\\3" },
      { "(\\bstd\\s*::\\s*((shared|unique)_ptr)\\b)", "std::\\2", "memory" },
      { "(\\bstd\\s*::\\s*(unordered_((map|set)))\\b)", "std::\\2", "unordered_\\3" },
      { "(\\bstd\\s*::\\s*(unordered_multi((map)|(set)))\\b)", "std::\\2", "unordered_\\3" },
      { "(\\bstd\\s*::\\s*list\\b)", "std::list", "list" },
      { "(\\bstd\\s*::\\s*string\\b)", "std::string", "string" },
      { "(\\bstd\\s*::\\s*vector\\b)", "std::vector", "vector" },
      { "(\\bstd\\s*::\\s*reference_wrapper\\b)", "std::reference_wrapper", "functional" },
      { "(\\bstd\\s*::\\s*(c?ref)\\b)", "std::\\2", "functional" },
      { "(\\bstd\\s*::\\s*(u?int[0-9]+_t)\\b)", "std::\\2", "cstdint" },
      { "(\\bstd\\s*::\\s*((i|o)?stringstream)\\b)", "std::\\2", "sstream" },
      // cstddef
      { "(\\bstd\\s*::\\s*size_t\\b)", "std::size_t", "cstddef" },
      { "(\\bstd\\s*::\\s*ptrdiff_t\\b)", "std::ptrdiff_t", "cstddef" },
      { "(\\bstd\\s*::\\s*nullptr_t\\b)", "std::nullptr_t", "cstddef" },
      // type_traits
      { "(\\bstd\\s*::\\s*true_type\\b)", "std::true_type", "type_traits" },
      { "(\\bstd\\s*::\\s*false_type\\b)", "std::false_type", "type_traits" },
      { "(\\bstd\\s*::\\s*integral_constant\\b)", "std::integral_constant", "type_traits" },
      { "(\\bstd\\s*::\\s*bool_constant\\b)", "std::bool_constant", "type_traits" },
      { "(\\bstd\\s*::\\s*is_void\\b)", "std::is_void", "type_traits" },
      { "(\\bstd\\s*::\\s*is_null_pointer\\b)", "std::is_null_pointer", "type_traits" },
      { "(\\bstd\\s*::\\s*is_integral\\b)", "std::is_integral", "type_traits" },
      { "(\\bstd\\s*::\\s*is_floating_point\\b)", "std::is_floating_point", "type_traits" },
      { "(\\bstd\\s*::\\s*is_array\\b)", "std::is_array", "type_traits" },
      { "(\\bstd\\s*::\\s*is_enum\\b)", "std::is_enum", "type_traits" },
      { "(\\bstd\\s*::\\s*is_union\\b)", "std::is_union", "type_traits" },
      { "(\\bstd\\s*::\\s*is_class\\b)", "std::is_class", "type_traits" },
      { "(\\bstd\\s*::\\s*is_function\\b)", "std::is_function", "type_traits" },
      { "(\\bstd\\s*::\\s*is_pointer\\b)", "std::is_pointer", "type_traits" },
      { "(\\bstd\\s*::\\s*is_lvalue_reference\\b)", "std::is_lvalue_reference", "type_traits" },
      { "(\\bstd\\s*::\\s*is_rvalue_reference\\b)", "std::is_rvalue_reference", "type_traits" },
      { "(\\bstd\\s*::\\s*is_member_object_pointer\\b)", "std::is_member_object_pointer", "type_traits" },
      { "(\\bstd\\s*::\\s*is_member_function_pointer\\b)", "std::is_member_function_pointer", "type_traits" },
      { "(\\bstd\\s*::\\s*is_fundamental\\b)", "std::is_fundamental", "type_traits" },
      { "(\\bstd\\s*::\\s*is_arithmetic\\b)", "std::is_arithmetic", "type_traits" },
      { "(\\bstd\\s*::\\s*is_scalar\\b)", "std::is_scalar", "type_traits" },
      { "(\\bstd\\s*::\\s*is_object\\b)", "std::is_object", "type_traits" },
      { "(\\bstd\\s*::\\s*is_compound\\b)", "std::is_compound", "type_traits" },
      { "(\\bstd\\s*::\\s*is_reference\\b)", "std::is_reference", "type_traits" },
      { "(\\bstd\\s*::\\s*is_member_pointer\\b)", "std::is_member_pointer", "type_traits" },
      { "(\\bstd\\s*::\\s*is_const\\b)", "std::is_const", "type_traits" },
      { "(\\bstd\\s*::\\s*is_volatile\\b)", "std::is_volatile", "type_traits" },
      { "(\\bstd\\s*::\\s*is_trivial\\b)", "std::is_trivial", "type_traits" },
      { "(\\bstd\\s*::\\s*is_trivially_copyable\\b)", "std::is_trivially_copyable", "type_traits" },
      { "(\\bstd\\s*::\\s*is_standard_layout\\b)", "std::is_standard_layout", "type_traits" },
      { "(\\bstd\\s*::\\s*is_pod\\b)", "std::is_pod", "type_traits" },
      { "(\\bstd\\s*::\\s*is_literal_type\\b)", "std::is_literal_type", "type_traits" },
      { "(\\bstd\\s*::\\s*is_empty\\b)", "std::is_empty", "type_traits" },
      { "(\\bstd\\s*::\\s*is_polymorphic\\b)", "std::is_polymorphic", "type_traits" },
      { "(\\bstd\\s*::\\s*is_abstract\\b)", "std::is_abstract", "type_traits" },
      { "(\\bstd\\s*::\\s*is_signed\\b)", "std::is_signed", "type_traits" },
      { "(\\bstd\\s*::\\s*is_unsigned\\b)", "std::is_unsigned", "type_traits" },
      { "(\\bstd\\s*::\\s*is_constructible\\b)", "std::is_constructible", "type_traits" },
      { "(\\bstd\\s*::\\s*is_trivially_constructible\\b)", "std::is_trivially_constructible", "type_traits" },
      { "(\\bstd\\s*::\\s*is_nothrow_constructible\\b)", "std::is_nothrow_constructible", "type_traits" },
      { "(\\bstd\\s*::\\s*is_default_constructible\\b)", "std::is_default_constructible", "type_traits" },
      { "(\\bstd\\s*::\\s*is_trivially_default_constructible\\b)", "std::is_trivially_default_constructible", "type_traits" },
      { "(\\bstd\\s*::\\s*is_nothrow_default_constructible\\b)", "std::is_nothrow_default_constructible", "type_traits" },
      { "(\\bstd\\s*::\\s*is_copy_constructible\\b)", "std::is_copy_constructible", "type_traits" },
      { "(\\bstd\\s*::\\s*is_trivially_copy_constructible\\b)", "std::is_trivially_copy_constructible", "type_traits" },
      { "(\\bstd\\s*::\\s*is_nothrow_copy_constructible\\b)", "std::is_nothrow_copy_constructible", "type_traits" },
      { "(\\bstd\\s*::\\s*is_move_constructible\\b)", "std::is_move_constructible", "type_traits" },
      { "(\\bstd\\s*::\\s*is_trivially_move_constructible\\b)", "std::is_trivially_move_constructible", "type_traits" },
      { "(\\bstd\\s*::\\s*is_nothrow_move_constructible\\b)", "std::is_nothrow_move_constructible", "type_traits" },
      { "(\\bstd\\s*::\\s*is_assignable\\b)", "std::is_assignable", "type_traits" },
      { "(\\bstd\\s*::\\s*is_trivially_assignable\\b)", "std::is_trivially_assignable", "type_traits" },
      { "(\\bstd\\s*::\\s*is_nothrow_assignable\\b)", "std::is_nothrow_assignable", "type_traits" },
      { "(\\bstd\\s*::\\s*is_copy_assignable\\b)", "std::is_copy_assignable", "type_traits" },
      { "(\\bstd\\s*::\\s*is_trivially_copy_assignable\\b)", "std::is_trivially_copy_assignable", "type_traits" },
      { "(\\bstd\\s*::\\s*is_nothrow_copy_assignable\\b)", "std::is_nothrow_copy_assignable", "type_traits" },
      { "(\\bstd\\s*::\\s*is_move_assignable\\b)", "std::is_move_assignable", "type_traits" },
      { "(\\bstd\\s*::\\s*is_trivially_move_assignable\\b)", "std::is_trivially_move_assignable", "type_traits" },
      { "(\\bstd\\s*::\\s*is_nothrow_move_assignable\\b)", "std::is_nothrow_move_assignable", "type_traits" },
      { "(\\bstd\\s*::\\s*is_destructible\\b)", "std::is_destructible", "type_traits" },
      { "(\\bstd\\s*::\\s*is_trivially_destructible\\b)", "std::is_trivially_destructible", "type_traits" },
      { "(\\bstd\\s*::\\s*is_same\\b)", "std::is_same", "type_traits" },
      { "(\\bstd\\s*::\\s*is_base_of\\b)", "std::is_base_of", "type_traits" },
      { "(\\bstd\\s*::\\s*is_convertible\\b)", "std::is_convertible", "type_traits" },
      { "(\\bstd\\s*::\\s*has_virtual_destructor\\b)", "std::has_virtual_destructor", "type_traits" },
      { "(\\bstd\\s*::\\s*alignment_of\\b)", "std::alignment_of", "type_traits" },
      { "(\\bstd\\s*::\\s*aligned_storage\\b)", "std::aligned_storage", "type_traits" },
      { "(\\bstd\\s*::\\s*aligned_union\\b)", "std::aligned_union", "type_traits" },
      { "(\\bstd\\s*::\\s*rank\\b)", "std::rank", "type_traits" },
      { "(\\bstd\\s*::\\s*extent\\b)", "std::extent", "type_traits" },
      { "(\\bstd\\s*::\\s*add_cv\\b)", "std::add_cv", "type_traits" },
      { "(\\bstd\\s*::\\s*add_const\\b)", "std::add_const", "type_traits" },
      { "(\\bstd\\s*::\\s*add_pointer\\b)", "std::add_pointer", "type_traits" },
      { "(\\bstd\\s*::\\s*add_volatile\\b)", "std::add_volatile", "type_traits" },
      { "(\\bstd\\s*::\\s*add_lvalue_reference\\b)", "std::add_lvalue_reference", "type_traits" },
      { "(\\bstd\\s*::\\s*add_rvalue_reference\\b)", "std::add_rvalue_reference", "type_traits" },
      { "(\\bstd\\s*::\\s*make_signed\\b)", "std::make_signed", "type_traits" },
      { "(\\bstd\\s*::\\s*make_unsigned\\b)", "std::make_unsigned", "type_traits" },
      { "(\\bstd\\s*::\\s*remove_cv\\b)", "std::remove_cv", "type_traits" },
      { "(\\bstd\\s*::\\s*remove_const\\b)", "std::remove_const", "type_traits" },
      { "(\\bstd\\s*::\\s*remove_volatile\\b)", "std::remove_volatile", "type_traits" },
      { "(\\bstd\\s*::\\s*remove_reference\\b)", "std::remove_reference", "type_traits" },
      { "(\\bstd\\s*::\\s*remove_pointer\\b)", "std::remove_pointer", "type_traits" },
      { "(\\bstd\\s*::\\s*remove_extent\\b)", "std::remove_extent", "type_traits" },
      { "(\\bstd\\s*::\\s*remove_all_extents\\b)", "std::remove_all_extents", "type_traits" },
      { "(\\bstd\\s*::\\s*decay\\b)", "std::decay", "type_traits" },
      { "(\\bstd\\s*::\\s*enable_if\\b)", "std::enable_if", "type_traits" },
      { "(\\bstd\\s*::\\s*conditional\\b)", "std::conditional", "type_traits" },
      { "(\\bstd\\s*::\\s*integral_constant\\b)", "std::integral_constant", "type_traits" },
      { "(\\bstd\\s*::\\s*common_type\\b)", "std::common_type", "type_traits" },
      { "(\\bstd\\s*::\\s*underlying_type\\b)", "std::underlying_type", "type_traits" },
      { "(\\bstd\\s*::\\s*result_of\\b)", "std::result_of", "type_traits" },
      // cstring
      { "(\\bstd\\s*::\\s*(mem((set)|(cpy)|(move)))\\b)", "std::\\2", "cstring" },
      // utility
      { "(\\bstd\\s*::\\s*swap\\b)", "std::swap", "utility" },
      { "(\\bstd\\s*::\\s*move\\b)", "std::move", "utility" },
      { "(\\bstd\\s*::\\s*forward\\b)", "std::forward", "utility" },
      { "(\\bstd\\s*::\\s*declval\\b)", "std::declval", "utility" },
      { "(\\bstd\\s*::\\s*pair\\b)", "std::pair", "utility" },
      { "(\\bstd\\s*::\\s*make_pair\\b)", "std::make_pair", "utility" },
      // algorithm
      { "(\\bstd\\s*::\\s*swap_ranges\\b)", "std::swap_ranges", "algorithm" },
      { "(\\bstd\\s*::\\s*iter_swap\\b)", "std::iter_swap", "algorithm" },
      // boost
      { "(\\bboost\\s*::\\s*atomic\\b)", "boost::atomic", "boost/atomic.hpp" },
      { "(\\bboost\\s*::\\s*(((exception)|(intrusive))_ptr)\\b)", "boost::\\3_ptr", "boost/\\3_ptr.hpp" },
      { nullptr, nullptr, nullptr }
    };

    //  include_check constructor  -------------------------------------------//

    include_check::include_check()
      : m_errors(0)
    {
      // C/C++ source code...
      register_signature( ".c" );
      register_signature( ".cpp" );
      register_signature( ".cxx" );
      register_signature( ".h" );
      register_signature( ".hpp" );
      register_signature( ".hxx" );
      register_signature( ".inc" );
      register_signature( ".ipp" );

      for (names_includes const* names_it = &names[0];
           names_it->name_regex != nullptr;
           ++names_it)
      {
        std::string rx(names_it->name_regex);
        rx +=
          "|"                   // or (ignored)
          "("
          "//[^\\n]*"           // single line comments (//)
          "|"
          "/\\*.*?\\*/"         // multi line comments (/**/)
          "|"
          "\"([^\"\\\\]|\\\\.)*\"" // string literals
          ")";
        regex_data.push_back(names_regex_data(names_it, rx));
      }
    }

    //  inspect ( C++ source files )  ---------------------------------------//

    void include_check::inspect(
      const string & library_name,
      const path & full_path,      // example: c:/foo/boost/filesystem/path.hpp
      const string & contents)     // contents of file to be inspected
    {
      if (contents.find( "hpxinspect:" "noinclude" ) != string::npos) return;

      // first, collect all #includes in this file
      std::set<std::string> includes;

      boost::sregex_iterator cur(contents.begin(), contents.end(), include_regex), end;

      for( ; cur != end; ++cur /*, ++m_errors*/ )
      {
        auto m = *cur;
        if (m[1].matched)
          includes.insert(std::string(m[1].first, m[1].second));
        else if (m[2].matched)
          includes.insert(std::string(m[2].first, m[2].second));
      }

      // for all given names, check whether corresponding include was found
      std::set<std::string> checked_includes;
      std::set<std::string> found_names;
      for (names_regex_data const& d : regex_data)
      {
        boost::sregex_iterator cur(contents.begin(), contents.end(), d.pattern), end;
        for(/**/; cur != end; ++cur)
        {
          auto m = *cur;
          if (m[1].matched)
          {
            // avoid checking the same include twice
            auto checked_includes_it =
                checked_includes.find(m.format(d.data->include));
            if (checked_includes_it != checked_includes.end())
               continue;

            // avoid errors to be reported twice
            std::string found_name(m[1].first, m[1].second);
            if (found_names.find(found_name) != found_names.end())
                continue;
            found_names.insert(found_name);

            auto include_it = includes.find(m.format(d.data->include));
            if (include_it == includes.end())
            {
              // include is missing
              auto it = contents.begin();
              auto match_it = m[1].first;
              auto line_start = it;

              string::size_type line_number = 1;
              for (/**/; it != match_it; ++it)
              {
                if (string::traits_type::eq(*it, '\n'))
                {
                  ++line_number;
                  line_start = it + 1; // could be end()
                }
              }

              ++m_errors;
              error(library_name, full_path, string(name())
                  + " missing #include ("
                  + m.format(d.data->include)
                  + ") for symbol "
                  + m.format(d.data->name) + " on line "
                  + linelink(full_path, boost::lexical_cast<string>(line_number)));
            }
            checked_includes.insert(m.format(d.data->include));
          }
        }
      }
    }

  } // namespace inspect
} // namespace boost

