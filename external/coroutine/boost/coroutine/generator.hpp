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

#ifndef BOOST_COROUTINE_GENERATOR_HPP_20060812
#define BOOST_COROUTINE_GENERATOR_HPP_20060812
#include <iterator>
#include <boost/optional.hpp>
#include <boost/none.hpp>
#include <boost/coroutine/shared_coroutine.hpp>
namespace boost { namespace coroutines {

  namespace detail {
    template<typename T>
    class tag {};

    class empty {};

    template<typename Category, typename ValueType>
    struct make_std_iterator {
      typedef std::iterator<Category,
			    ValueType,
			    std::ptrdiff_t,
			    BOOST_DEDUCED_TYPENAME boost::remove_reference<ValueType>::type *, //pointer
			    BOOST_DEDUCED_TYPENAME boost::remove_reference<ValueType>::type & //reference
			    > type;
    };
  }

  // This simple class implement generators (a simple
  // subset of coroutines) in the form of an InputIterator
  // interface. It also models to the AdaptableGenerator concept.
  // Finally it is ConvertibleToBool.
  template<typename ValueType, 
	   typename Coroutine = 
	   shared_coroutine<ValueType()> >
  class generator : public boost::mpl::eval_if<boost::is_same<ValueType, void>,
					       boost::mpl::identity<detail::empty>,
					       detail::make_std_iterator<
    std::input_iterator_tag, 
    typename Coroutine::result_type> >::type {
    typedef void(generator::*safe_bool)();
    typedef ValueType internal_value_type;

  public:
    typedef Coroutine coroutine_type;
    typedef BOOST_DEDUCED_TYPENAME 
    coroutine_type::result_type result_type;
    typedef result_type value_type;

    typedef BOOST_DEDUCED_TYPENAME coroutine_type::self self;

    generator() {}

    template<typename Functor>
    generator(Functor f) :
      m_coro(f),
      m_val(assign(detail::tag<result_type>())){};

    generator(const generator& rhs) :
      m_coro(rhs.m_coro),
      m_val(rhs.m_val) {}

    value_type operator*() {
      return *m_val;
    }

    generator& operator++() {
      m_val = assign(detail::tag<result_type>());
      return *this;
    }

    generator operator++(int) {
      generator t(*this);
      ++(*this);
      return t;
    }

    friend
    bool operator==(const generator& lhs, const generator& rhs) {
      return lhs.m_val == rhs.m_val;
    }

    friend 
    bool operator != (const generator& lhs, const generator & rhs) {
      return !(lhs == rhs);
    }

    operator safe_bool () const {
      return m_val? &generator::safe_bool_true: 0;
    }

    value_type operator()() {
      return *(*this)++;
    }

  private:
    void safe_bool_true () {};

    // hack to handle correctly void result types.
    struct optional_void {
      optional_void() : m_result(true) {}
      optional_void(boost::none_t) : m_result(false) {}

      void operator*() const {}
      operator bool() const { return m_result; };
      bool m_result;
    };

    typedef BOOST_DEDUCED_TYPENAME
    boost::mpl::if_<boost::is_same<value_type, void>,
		    optional_void,
		    boost::optional<value_type> >::type optional_type;

    template<typename T>
    optional_type assign(detail::tag<T>) {
      return m_coro? m_coro(std::nothrow): boost::none;
    }

    optional_type assign(detail::tag<void>) {
      return m_coro? (m_coro(std::nothrow), optional_type()): boost::none;
    }

    // There is a possible EBO here. May be use compressed pair.
    coroutine_type m_coro;
    optional_type m_val;

  };

} }
#endif
