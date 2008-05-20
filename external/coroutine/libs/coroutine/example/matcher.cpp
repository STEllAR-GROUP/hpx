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
#include <boost/coroutine/coroutine.hpp>

namespace coroutines = boost::coroutines;
using coroutines::coroutine;

bool matcher(coroutine<bool(char)>::self& self, char c) { 
  while(true) { 
    if(c == 'h') { 
      c = self.yield(false); 
      if(c == 'e') { 
	c = self.yield(false); 
	if(c == 'l') { 
	  c = self.yield(false); 
	  if(c == 'l') { 
	    c = self.yield(false); 
	    if(c == 'o') { 
	      c = self.yield(true); 
	    } continue; 
	  } else continue; 
	} else continue; 
      } else continue; 
    } c = self.yield(false); 
  } 
} 

int main(int, char**) { 
  coroutine<bool(char)> match(matcher); 
  std::string buffer = "hello to everybody, this is not an hello world program.";
  for(std::string::iterator i = buffer.begin(); 
      i != buffer.end(); 
      ++i) { 
    if(match(*i)) std::cout<< "match\n"; 
  } 
} 
 
