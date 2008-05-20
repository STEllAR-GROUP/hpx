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
#include <boost/coroutine/coroutine.hpp>

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
 */
namespace coroutines = boost::coroutines;
using coroutines::coroutine;

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

typedef int leaf;
struct node;
typedef boost::variant<boost::recursive_wrapper<node>,  leaf> tree_element;

typedef boost::tuple<tree_element, tree_element> node_;
struct node : node_ {
  template<typename A, typename B>
  node(const A& a, const B& b) :
    node_(a, b) {}
};

bool is_leaf(const tree_element& x) {
  return x.which() == 1;
}

template<typename Left, typename Right>
tree_element make_tree(meta::node<Left, Right> const&) {
  return tree_element(node(make_tree(Left()), 
			   make_tree(Right())));
}

template<int A>
tree_element make_tree(meta::leaf<A> const&) {
  return tree_element(A);
}

typedef coroutine<boost::optional<leaf>(tree_element)> coroutine_type;

boost::optional<leaf>  tree_leaves(coroutine_type::self& self, tree_element tree) {
  if (is_leaf(tree)) {
    self.yield(boost::get<leaf>(tree));
  } else {
    tree_leaves(self, boost::get<node>(tree).get<0>());
    tree_leaves(self, boost::get<node>(tree).get<1>());
  }
  return boost::optional<leaf>();
}

bool same_fringe(tree_element tree1, tree_element tree2) {
  coroutine_type tree_leaves_a(tree_leaves);
  coroutine_type tree_leaves_b(tree_leaves);
  boost::optional<leaf> tmp1, tmp2;
  while ((tmp1 = tree_leaves_a(tree1)) && (tmp2=tree_leaves_b(tree2))) 
    if (!(tmp1 == tmp2)) return false;
  return true;
}

int main() {
  std::cout<<"same_fringe tree_a, tree_a "<<same_fringe
    (make_tree(meta::tree_a()), make_tree(meta::tree_a()))<<"\n";
  std::cout<<"same_fringe tree_a, tree_b "<<same_fringe
    (make_tree(meta::tree_a()), make_tree(meta::tree_b()))<<"\n";
  std::cout<<"same_fringe tree_a, tree_c "<<same_fringe
    (make_tree(meta::tree_a()), make_tree(meta::tree_c()))<<"\n";
}
