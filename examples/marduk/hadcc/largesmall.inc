c.......................................................................
c
c Copyright (c) 2011 Matthew Anderson
c
c largesmall.inc:   Parameters for real*8 number comparison
c                   when small precision mistakes on different
c                   architectures can have unintended consequences
c
c
c.......................................................................

      !
      ! Used for certain types of masking of values:
      !
      real(kind=8)        LARGENUMBER
      parameter     (     LARGENUMBER    =  9.9d99  )

      !
      ! Used as a fuzz factor (see below)
      !
      real(kind=8)        SMALLNUMBER
      parameter     (     SMALLNUMBER    =  1.0d-12  )

      !
      ! Used for testing whether a number is LESS    than a largenumber
      !                                   or GE      than a largenumber
      !
      real(kind=8)        LARGE_M_SMALL
      parameter     (     LARGE_M_SMALL  = LARGENUMBER - SMALLNUMBER )

      !
      ! Used for testing whether a number is GREATER than a largenumber
      !                                   or LE      than a largenumber
      !
      real(kind=8)        LARGE_P_SMALL
      parameter     (     LARGE_P_SMALL  = LARGENUMBER + SMALLNUMBER )
