//  Copyright (c) 2006, Giovanni P. Deretta
//
//  This code may be used under either of the following two licences:
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy 
//  of this software and associated documentation files (the "Software"), to deal 
//  in the Software without restriction, including without limitation the rights 
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
//  copies of the Software, and to permit persons to whom the Software is 
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in 
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
//  THE SOFTWARE. OF SUCH DAMAGE.
//
//  Or:
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <boost/variant.hpp>
#include <boost/variant/get.hpp>
#include <boost/variant/recursive_variant.hpp>
#include <boost/variant/recursive_wrapper.hpp>
#include <boost/optional.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/coroutine/generator.hpp>
#include <boost/bind.hpp>
#include <boost/mpl/at.hpp>
/**
 * Solve the 'same fringe problem' using coroutines.
 * Given two binary trees, they have the same fringe
 * if all leafs, read from left to right are equals.
 * This is the classical coroutine demonstration problem,
 * because it is hard to solve in O(N) (with best case O(1)) 
 * without using coroutines.
 * see http://c2.com/cgi/wiki?CoRoutine
 * NOTE: this solution is an almost verbatim port of the lua solution from
 * the wiki.
 *
 * This is a cleaned up version of samefringe.cpp. It uses generators instead
 * of corutines. The variant is much more cleaner and 
 * same_fringe has been simplified.
 */
namespace coroutines = boost::coroutines;
using coroutines::generator;

namespace meta {
  /**
   * Compact compile-time description of binary trees of ints.
   */
  template<typename Left, typename Right>
  class node{
    typedef Left left;
    typedef Right right;
  };
  
  template<int A>
  struct leaf {
    enum {value = A};
  };

  typedef 
  node<node<leaf<0>, leaf<1> >, node<leaf<0>, node<leaf<5>, leaf<7> > > >
  tree_a; // fringe: 0 1 0 5 7

  typedef 
  node<leaf<0>, node<leaf<1>, node<node<leaf<0>, leaf<5> >, leaf<7> > > >
  tree_b; // fringe: 0 1 0 5 7

  typedef 
  node<leaf<1>, node<leaf<7>, node<node<leaf<5>, leaf<4> >, leaf<7> > > >
  tree_c; // fringe: 1 7 5 4 7
}

// This is ugly, but at least on GCC it is not fount if it is in the the global 
// namespace. It could  work both in boost and in std:
// - in std   is found through ADL from pair,
// - in boost is found through ADL from pair's template argument 
//   (boost::variant)
// Adding overloads in std is not portable so we fallback to the boost 
// namespace.
namespace boost {
  template<typename Element>
  std::ostream& operator<<(std::ostream& out, 
			   const std::pair<Element, Element>& x) {
    out << "("<< x.first << ", " << x.second<<")";
    return out;
  }
}


typedef int leaf;
typedef boost::make_recursive_variant<leaf, 
				      std::pair<boost::recursive_variant_, 
						boost::recursive_variant_> 
>::type element;
typedef std::pair<element, element> node;

bool is_leaf(const element& x) {
  return x.which() == 0;
}


template<typename Left, typename Right>
element make_tree(meta::node<Left, Right> const&) {
  return node(make_tree(Left()), 
	      make_tree(Right()));
}

template<int A>
element make_tree(meta::leaf<A> const&) {
  return leaf (A);
}

typedef generator<leaf> generator_type;

leaf
tree_leaves(generator_type::self& self, element& tree) {
  if (is_leaf(tree)) {
    self.yield(boost::get<leaf>(tree));
  } else {
    tree_leaves(self, boost::get<node>(tree).first);
    tree_leaves(self, boost::get<node>(tree).second);
  }
  self.exit();
  return 0;
}

bool same_fringe(const element& tree1, const element& tree2) {
  generator_type tree_leaves_a(boost::bind(tree_leaves, _1, tree1));
  generator_type tree_leaves_b(boost::bind(tree_leaves, _1, tree2));
  while(tree_leaves_a && tree_leaves_b) {
    if(tree_leaves_a() != tree_leaves_b())
      return false;
  }
  return true && (!tree_leaves_b && !tree_leaves_a);
}

int main() {
  std::cout << "fringe a: " << make_tree(meta::tree_a()) << "\n";
  std::cout << "fringe b: " << make_tree(meta::tree_b()) << "\n";
  std::cout << "frinte c: " << make_tree(meta::tree_c()) << "\n";
  std::cout<<"same_fringe (tree_a, tree_a) -> "<<same_fringe
    (make_tree(meta::tree_a()), make_tree(meta::tree_a()))<<"\n";
  std::cout<<"same_fringe (tree_a, tree_b) -> "<<same_fringe
    (make_tree(meta::tree_a()), make_tree(meta::tree_b()))<<"\n";
  std::cout<<"same_fringe (tree_a, tree_c) -> "<<same_fringe
    (make_tree(meta::tree_a()), make_tree(meta::tree_c()))<<"\n";
}
