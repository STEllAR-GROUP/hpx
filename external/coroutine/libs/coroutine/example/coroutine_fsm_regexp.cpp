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
#include <string>
#include <algorithm>
#include <boost/coroutine/shared_coroutine.hpp>

namespace coroutines = boost::coroutines;
using coroutines::shared_coroutine;

typedef shared_coroutine<void(char)> coroutine_type;

void fsm_regexp(coroutine_type::self& self, char) {
  while(true) {
    if(self.result() == '0') {
      std::cout << '0';
      self.yield();
      if(self.result() == '1') {
	std::cout << '0';
	self.yield();
	while(self.result() == '1') {
	  std::cout << '0';
	  self.yield();
	}
	std::cout <<'0';
	self.yield();
	if(self.result() == '1') {
	  std::cout << '0';
	  self.yield();
	  if(self.result() == '0') {
	    std::cout << '1';
	    self.yield();
	  } else {
	    std::cout <<'0';
	    self.yield();
	  }
	} else {
	  std::cout <<'0';
	  self.yield();
	}
      } else {
	std::cout << '0';
	self.yield();
      }
    } else {
      std::cout <<'0';
      self.yield();
    }
  } 
}

int main() {
  {
    std::string input ("0110100010010001101001000111110010011001");
    std::for_each(input.begin(), input.end(), coroutine_type(fsm_regexp));
    std::cout <<"\n";
  }
  {
    std::string input ("011011010");
    std::for_each(input.begin(), input.end(), coroutine_type(fsm_regexp));
    std::cout <<"\n";
  }

}
