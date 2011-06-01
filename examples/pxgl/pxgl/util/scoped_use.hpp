// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_UTIL_SCOPED_USE_20110217T1513)
#define PXGL_UTIL_SCOPED_USE_20110217T1513

#include <hpx/hpx.hpp>
#include <hpx/hpx_fwd.hpp>

////////////////////////////////////////////////////////////////////////////////
namespace pxgl { namespace util { 
  //////////////////////////////////////////////////////////////////////////////
  struct scoped_use
  {
    typedef int result_type;
    typedef boost::exception_ptr error_type;
    typedef boost::variant<result_type, error_type> feb_data_type;
    typedef hpx::util::full_empty<feb_data_type> feb_type;

    scoped_use(feb_type & feb)
      : feb_(feb)
    {
      feb_.read_and_empty(d_);

      if (1 == d_.which())
      {
        error_type e = boost::get<error_type>(d_);
        boost::rethrow_exception(e);
      }
    }

    ~scoped_use()
    {
      feb_.write(d_);
    }

  private:
    feb_type & feb_;
    feb_data_type d_;
  };
}}

#endif

