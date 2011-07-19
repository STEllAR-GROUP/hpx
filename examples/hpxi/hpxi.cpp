///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// NOTE: crude alpha, based on the phxi.cpp code from Prana

#include <phxpr/config.hpp>

#include <ios>
#include <iostream>
#include <sstream>
#include <fstream>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/fusion/include/at_c.hpp>

#include <prana/version.hpp>
#include <prana/utree/io.hpp>

#include <sheol/adt/dynamic_array.hpp>

#include <prana/parse/parse_tree.hpp>
#include <prana/parse/grammar/sexpr.hpp>
#include <prana/generate/generate_sexpr.hpp>
    
#include <phxpr/version.hpp>
#include <phxpr/intrinsics/assert.hpp>
#include <phxpr/intrinsics/basic_arithmetic.hpp>
#include <phxpr/intrinsics/basic_io.hpp>
#include <phxpr/intrinsics/comparison_predicates.hpp>
#include <phxpr/intrinsics/equivalence_predicates.hpp>
#include <phxpr/intrinsics/type_predicates.hpp>
#include <phxpr/evaluator.hpp>

#include "local_eager_future.hpp" // Temporary hackage until the runtime linker
                                  // is completed.

#include "serialize_utree.hpp" // Move this jazz to Prana?

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::positional_options_description;
using boost::program_options::value;
using boost::program_options::store;
using boost::program_options::command_line_parser;
using boost::program_options::notify;

using boost::fusion::at_c;

using boost::spirit::nil;
using boost::spirit::utree;
using boost::spirit::utree_type;

using boost::filesystem::path;
using boost::filesystem::exists;
using boost::filesystem::is_directory;

using sheol::adt::dynamic_array;  

using prana::parse_tree;
using prana::tag::sexpr;
using prana::generate_sexpr;

using phxpr::evaluator;
using phxpr::evaluate;

using phxpr::addition;
using phxpr::subtraction;
using phxpr::multiplication;
using phxpr::division;

using phxpr::less_predicate;
using phxpr::less_equal_predicate;
using phxpr::greater_predicate;
using phxpr::greater_equal_predicate;

using phxpr::equal_predicate;

using phxpr::boolean_predicate;
using phxpr::symbol_predicate;
using phxpr::procedure_predicate;
using phxpr::list_predicate;
using phxpr::number_predicate;
using phxpr::string_predicate;
using phxpr::nil_predicate;
using phxpr::invalid_predicate;

using phxpr::display;
using phxpr::newline;

using phxpr::assertion;

using hpx::init;
using hpx::finalize;

int hpx_main(variables_map& vm) {
  std::string input("");

  else if (vm.count("version")) {
    std::cout <<
      "hpxi, a parallel lisp interpreter (phxpr v" PHXPR_VERSION_STR ")\n"
      "Prana v" PRANA_VERSION_STR "\n"
      "HPX v" HPX_VERSION_STR "\n"
      "\n"
      "Copyright (c) 2010-2011 Bryce Lelbach, Joel de Guzman and Hartmut Kaiser\n"
      "\n" 
      "Distributed under the Boost Software License, Version 1.0. (See accompanying\n"
      "file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)\n";
    return 0;
  }

  if (!vm.count("input")) {
    std::cerr << "error: no file specified\n";
    return 1; 
  }

  else {
    input = vm["input"].as<std::string>();

    path ip(input);

    if (!exists(ip)) {
      std::cerr << "error: phxpr input '" << input << "' does not exist\n";
      return 1;
    }

    else if (is_directory(ip)) {
      std::cerr << "error: phxpr input '" << input << "' is a directory\n";
      return 1;
    }
  }

  std::ifstream ifs(input.c_str(), std::ifstream::in);  
  evaluator e;

  const bool hide_pointers = vm.count("hide-pointers");

  // globals
  e.define_global("nil", nil);

  // basic arithmetic 
  e.define_intrinsic("+", addition());
  e.define_intrinsic("-", subtraction());
  e.define_intrinsic("*", multiplication());
  e.define_intrinsic("/", division());

  // comparison predicates
  e.define_intrinsic("<", less_predicate());
  e.define_intrinsic("<=", less_equal_predicate());
  e.define_intrinsic(">", greater_predicate());
  e.define_intrinsic(">=", greater_equal_predicate());
 
  // equivalence predicates 
  e.define_intrinsic("=", equal_predicate());

  // type predicates
  e.define_intrinsic("boolean?", boolean_predicate());
  e.define_intrinsic("symbol?", symbol_predicate());
  e.define_intrinsic("procedure?", procedure_predicate());
  e.define_intrinsic("list?", list_predicate());
  e.define_intrinsic("number?", number_predicate());
  e.define_intrinsic("string?", string_predicate());
  e.define_intrinsic("null?", nil_predicate());
  e.define_intrinsic("unspecified?", invalid_predicate());

  // basic io
  e.define_intrinsic("display", display(std::cout, !hide_pointers));
  e.define_intrinsic("newline", newline(std::cout));
  
  // debugging 
  e.define_intrinsic("assert", assertion());

  dynamic_array<boost::shared_ptr<parse_tree<sexpr> > > asts(16);

  if (vm.count("print-return")) {
    utree r;

    // interpreter file REL
    while (ifs.good()) {
      asts.push_back(boost::shared_ptr<parse_tree<sexpr> >());

      // read
      asts.back().reset(new parse_tree<sexpr>(ifs));

      // eval
      if (!ifs.good()) {
        r = evaluate(*asts.back(), e);
        break;
      }

      // invoke for side effects
      else 
        evaluate(*asts.back(), e);
    }

    std::cout << "(return-value ";
    generate_sexpr(r, std::cout, !hide_pointers);
    std::cout << ")\n";
  }

  else {
    // interpreter file REL
    while (ifs.good()) {
      asts.push_back(boost::shared_ptr<parse_tree<sexpr> >());

      // read
      asts.back().reset(new parse_tree<sexpr>(ifs));

      // eval
      if (!ifs.good()) {
        evaluate(*asts.back(), e);
        break;
      }

      // invoke for side effects
      else 
        evaluate(*asts.back(), e);
    }
  }

  return 0; 
}

int main (int argc, char** argv) { 
  options_description desc_commandline("Usage: hpxi [options]");
   
  desc_commandline.add_options()
    ("help,h", "display this message")
    ("print-return", "print return value after evaluation")
    ("hide-pointers",
     "do not show C++ pointer values when displaying objects and procedures")
    ("version,v", "display the version and copyright information")
    ("input,i", value<std::string>(), 
     "file to read and execute phxpr code from") 
  ;

  // Initialize and run HPX
  return init(desc_commandline, argc, argv);
}

