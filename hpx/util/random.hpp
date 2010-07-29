//  Copyright (c) 2010-2011 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_RANDOM_JUN_29_2010_1132PM)
#define HPX_UTIL_RANDOM_JUN_29_2010_1132PM

#include <stdlib.h>

namespace hpx { namespace util { namespace random
{
  class random_generator
  {
  public:
    typedef std::size_t size_type;
    typedef int32_t random_value_type;

    random_generator(size_type state_len=1000)
      : state_len_(state_len)
    {
      seed(0);
    }

    ~random_generator(void)
    {
      delete[] state_;
    }

    random_value_type operator()(void)
    {
      random_value_type tmp;
      random_r(&ctx_, &tmp);

      return tmp;
    }

    void seed(unsigned int seed)
    {
      state_ = new char[state_len_];
      initstate_r(seed, state_, state_len_, &ctx_);
    }

  private:
    char* state_;
    size_type state_len_;

    random_data ctx_;
  };

  random_generator::random_value_type get_max(void)
  {
    return RAND_MAX;
  }
}}}

#endif
