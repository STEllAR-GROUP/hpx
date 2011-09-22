!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
! Copyright (c) 2011 Bryce Adelstein-Lelbach
!
! Distributed under the Boost Software License, Version 1.0. (See accompanying 
! file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 

      program gfortran_version
        write (*, '(3i2.2)', advance='no')
     *      __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__
      end program gfortran_version

