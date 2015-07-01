//  Copyright (c) 2011 Matt Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef FNAME_H
#define FNAME_H

#if defined(sgi) || defined(SGI) || defined(__sgi__) || defined(__SGI__)
#define FNAME(n_) n_##_
#elif defined(__INTEL_COMPILER)
#define FNAME(n_) n_##_
#elif defined(__GNUC__) && !defined(__INTEL_COMPILER)
#define FNAME(n_) n_##_
#elif defined(__PGI)
#define FNAME(n_) n_##_
#elif defined(_MSC_VER)
#define FNAME(n_) n_
#else
#error "Unknown Fortran name mangling convention"
#endif

#endif

