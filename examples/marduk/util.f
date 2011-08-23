c
c Copyright (c) 2011 Steve Liebling
c Copyright (c) 2011 Matt Anderson
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc                                                                            cc
cc  find_bbox:                                                                cc
cc             Find bounding box for a binary [0,1] array.                    cc
cc             Note: finds bounding box within *existing* box                 cc
cc                   defined by [bbminx,bbmaxx][][].                          cc
cc             Likewise computes:                                             cc
cc                   sigi, sigj, sigk  ---Signature lines                     cc
cc                                                                            cc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine find_bbox( flagarray,
     *                      sigi,        sigj,       sigk,
     *                      bbminx,      bbminy,     bbminz,
     *                      bbmaxx,      bbmaxy,     bbmaxz,
     *                      nx,          ny,         nz,
     *                      efficiency                           )
      implicit    none
      integer     nx,         ny,         nz,
     *            bbminx,     bbminy,     bbminz,
     *            bbmaxx,     bbmaxy,     bbmaxz
      real(kind=8)      flagarray(nx,ny,nz),
     *            sigi(nx),   sigj(ny),   sigk(nz),
     *            efficiency
      integer     i,          j,          k,
     *            bbminx_tmp, bbminy_tmp, bbminz_tmp,
     *            bbmaxx_tmp, bbmaxy_tmp, bbmaxz_tmp
      logical     ltrace
      parameter ( ltrace = .false. )

c     if (ltrace) then
c        write(*,*) 'find_bbox: bbminx, bbmaxx =',bbminx,bbmaxx
c        write(*,*) 'find_bbox: bbminy, bbmaxy =',bbminy,bbmaxy
c        write(*,*) 'find_bbox: bbminz, bbmaxz =',bbminz,bbmaxz
c        write(*,*) 'find_bbox: nx,ny,nz=',nx,ny,nz
c     end if
      !
      ! Zero out signature:
      !   (only need to zero where maximal bounding box exists:
      !
      do i = bbminx, bbmaxx
         sigi(i) = 0.d0
      end do
      do j = bbminy, bbmaxy
         sigj(j) = 0.d0
      end do
      do k = bbminz, bbmaxz
         sigk(k) = 0.d0
      end do

      !
      ! Define temporary bounding box:
      !
      bbminx_tmp = nx
      bbminy_tmp = ny
      bbminz_tmp = nz
      bbmaxx_tmp = 1
      bbmaxy_tmp = 1
      bbmaxz_tmp = 1

      efficiency = 0.d0

      do k = bbminz, bbmaxz
         do j = bbminy, bbmaxy
            do i = bbminx, bbmaxx
               if ( flagarray(i,j,k) .gt. 0 ) then
                  efficiency = efficiency + 1.d0
                  sigi(i)    = sigi(i) + 1.d0
                  sigj(j)    = sigj(j) + 1.d0
                  sigk(k)    = sigk(k) + 1.d0
                  if (i.gt.bbmaxx_tmp) bbmaxx_tmp = i
                  if (i.lt.bbminx_tmp) bbminx_tmp = i
                  if (j.gt.bbmaxy_tmp) bbmaxy_tmp = j
                  if (j.lt.bbminy_tmp) bbminy_tmp = j
                  if (k.gt.bbmaxz_tmp) bbmaxz_tmp = k
                  if (k.lt.bbminz_tmp) bbminz_tmp = k
               end if
            end do
         end do
c        if (ltrace) write(*,*) k, sigk(k)
      end do

      !
      ! replace input bbox with  the bounding box found here:
      !
      bbminx = bbminx_tmp
      bbminy = bbminy_tmp
      bbminz = bbminz_tmp
      bbmaxx = bbmaxx_tmp
      bbmaxy = bbmaxy_tmp
      bbmaxz = bbmaxz_tmp

c     if (ltrace) write(*,*) 'find_bbox: numpoints = ',efficiency

      efficiency = efficiency / (bbmaxx - bbminx + 1)
     *                        / (bbmaxy - bbminy + 1)
     *                        / (bbmaxz - bbminz + 1)

      if (ltrace) then
         write(*,*) 'find_bbox: efficiency     = ',efficiency
         write(*,*) 'find_bbox: bbminx, bbmaxx =',bbminx,bbmaxx
         write(*,*) 'find_bbox: bbminy, bbmaxy =',bbminy,bbmaxy
         write(*,*) 'find_bbox: bbminz, bbmaxz =',bbminz,bbmaxz
         do i = bbminx, bbmaxx
            write(*,*) i, sigi(i)
         end do
      end if

      return
      end     ! END: find_bbox

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc                                                                            cc
cc  compute_disallowed:                                                       cc
cc             Compute signatures and Laplacians for disallowed points        cc
cc             in a given cluster.                                            cc
cc                                                                            cc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine compute_disallowed( flagarray,
     *                      sigi,        sigj,       sigk,
     *                      lapi,        lapj,       lapk,
     *                      bbminx,      bbminy,     bbminz,
     *                      bbmaxx,      bbmaxy,     bbmaxz,
     *                      nx,          ny,         nz,
     *                      disallowedlogical                    )
      implicit    none
      integer     nx,         ny,         nz,
     *            bbminx,     bbminy,     bbminz,
     *            bbmaxx,     bbmaxy,     bbmaxz
      real(kind=8)      flagarray(nx,ny,nz),
     *            sigi(nx),   sigj(ny),   sigk(nz),
     *            lapi(nx),   lapj(ny),   lapk(nz)
      logical     disallowedlogical
      integer     i,          j,          k
      integer     lengthx,    lengthy,    lengthz
      logical     ltrace
      parameter ( ltrace = .false. )

      if (ltrace) then
         write(*,*) 'compute_disallowed: bbminx/maxx: ',bbminx,bbmaxx
         write(*,*) 'compute_disallowed: bbminy/maxy: ',bbminy,bbmaxy
         write(*,*) 'compute_disallowed: bbminz/maxz: ',bbminz,bbmaxz
         write(*,*) 'compute_disallowed: nx/y/z:      ',nx,ny,nz
      end if

      !
      ! Zero out signature:
      !   (only need to zero where bounding box exists:
      !
      do i = bbminx, bbmaxx
         sigi(i) = 0.d0
      end do
      do j = bbminy, bbmaxy
         sigj(j) = 0.d0
      end do
      do k = bbminz, bbmaxz
         sigk(k) = 0.d0
      end do


      disallowedlogical = .false.

      do k = bbminz, bbmaxz
         do j = bbminy, bbmaxy
            do i = bbminx, bbmaxx
               if ( flagarray(i,j,k) .lt. 0 ) then
                  disallowedlogical = .true.
                  sigi(i)    = sigi(i) + 1.d0
                  sigj(j)    = sigj(j) + 1.d0
                  sigk(k)    = sigk(k) + 1.d0
               end if
            end do
         end do
      end do

      !
      ! Only compute lapi/j/k if disallowed points are here:
      !
      if (disallowedlogical) then
         lengthx = bbmaxx-bbminx+1
         lengthy = bbmaxy-bbminy+1
         lengthz = bbmaxz-bbminz+1
         !write(*,*) lengthx,lengthy, lengthz
         call  compute_lap(lapi(bbminx), sigi(bbminx), lengthx )
         call  compute_lap(lapj(bbminy), sigj(bbminy), lengthy )
         call  compute_lap(lapk(bbminz), sigk(bbminz), lengthz )
      end if

      if (ltrace) then
         write(*,*) 'compute_disallowed: disallowedlogical: ',
     *                                   disallowedlogical
      end if

      return
      end     ! END: compute_disallowed

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc                                                                            cc
cc  find_inflect:                                                             cc
cc          finds largest inflection point in 1d vector                       cc
cc          Used for clustering ala Berger and Rigtousis '91.                 cc
cc                                                                            cc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine find_inflect( u1, max, max_i, nx)
      implicit    none
      integer     max_i, nx
      real(kind=8)      max, u1(nx)
      integer     i
      real(kind=8)      tmp, tmp2
      logical     ltrace
      parameter ( ltrace      = .false. )


      max     = 0.d0
      tmp     = u1(1)

      do i = 2, nx
         if ( u1(i)*tmp .lt. 0.d0 ) then
            !
            ! If opposite sign, then an inflection point:
            !
            tmp2            = abs(u1(i)-tmp)
            if (ltrace) write(*,*) ' Inflection pt found: ',i,tmp2
            if (tmp2 .gt. max) then
               if (ltrace) write(*,*) ' New max'
               max   = tmp2
               max_i = i - 1
            end if
         end if
         tmp = u1(i)
      end do


      return
      end      ! END: find_inflect

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc                                                                            cc
cc  load_scal1d:                                                              cc
cc                                                                            cc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine load_scal1d( field, scalar, nx)
      implicit   none
      integer    nx
      real(kind=8)     field(nx), scalar
      integer    i

            do i = 1, nx
               field(i) = scalar
            end do

      return
      end    ! END: load_scal1d
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc                                                                            cc
cc  find_min1d:                                                               cc
cc          finds minimum of 1d vector                                        cc
cc          If multiple regions of the same minimum value,                    cc
cc          it finds the one with the largest span.                           cc
cc             min   = minimum value found                                    cc
cc             min_i = index of minimum where first found                     cc
cc             span  = number of successive points at this minimum            cc
cc                                                                            cc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine find_min1d( u1, min, min_i, span, nx)
      implicit    none
      integer     min_i,   span, nx
      real(kind=8)      u1(nx),  min
      integer     min_i_tmp,   span_tmp
      real(kind=8)      min_tmp
      integer     i
      !real(kind=8)      LARGENUMBER
      !parameter ( LARGENUMBER = 9.d98 )
      include       'largesmall.inc'
      logical     within_span, within_span_tmp
      logical     ltrace
      parameter ( ltrace      = .false. )

      if (ltrace) then
         write(*,*) 'find_min1d: nx     = ',nx
         write(*,*) 'find_min1d: u1(1)  = ',u1(1)
         write(*,*) 'find_min1d: u1(nx) = ',u1(nx)
      end if

      if (nx.le.0) then
            min   = LARGENUMBER
            min_i = 0
            span  = 0
            return
      end if

      min     = LARGENUMBER
      min_tmp = LARGENUMBER + 1

      do i = 1, nx
         if (ltrace) write(*,*) '      i, u1(i)=',i,u1(i)
         if ( u1(i) .lt. min ) then
            min             = u1(i)
            min_i           = i
            span            = 1
            within_span     = .true.
            within_span_tmp = .false.
         else if ( (u1(i).eq.min) .and. within_span) then
            span            = span + 1
         else if (        (u1(i).eq.min)
     *             .and. .not.within_span
     *             .and. .not.within_span_tmp) then
            min_tmp         = min
            min_i_tmp       = i
            span_tmp        = 1
            within_span_tmp = .true.
         else if ( (u1(i).eq.min) .and. within_span_tmp) then
            span_tmp        = span_tmp + 1
         else                !   (u1(i) .ne. min)
            within_span     = .false.
            within_span_tmp = .false.
            if ( (min .eq. min_tmp) .and. (span_tmp.gt.span) ) then
               span  = span_tmp
               min_i = min_i_tmp
            end if
         end if
      end do

      if (ltrace) then
         write(*,*) 'find_min1d: min   = ', min
         write(*,*) 'find_min1d: min_i = ', min_i
         write(*,*) 'find_min1d: span  = ', span
      end if

      return
      end      ! END: find_min1d
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc                                                                            cc
cc  compute_lap:                                                              cc
cc          computes Laplacian terms from signature arrays...                 cc
cc          Used for clustering ala Berger and Rigtousis '91.                 cc
cc                                                                            cc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine compute_lap( lap, sig, nx)
      implicit    none
      integer     nx
      real(kind=8)      lap(nx), sig(nx)
      integer     i

      lap(1)  = 0.d0
      lap(nx) = 0.d0
      do i = 2, nx-1
         lap(i) = 2.d0*sig(i) - sig(i-1) - sig(i+1)
      end do

      return
      end      ! END: compute_lap


cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc                                                                            cc
cc    grow_bbox:                                                              cc
cc              Expand bboxes to include a ghostwidth but                     cc
cc              be careful not to:                                            cc
cc                  1) include any disallowed points                          cc
cc                  2) go past the bounds of the grid                         cc
cc                                                                            cc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine grow_bbox( flagarray, value, width, bmini,bmaxi,
     *                      bminj,bmaxj,bmink,bmaxk, nx,ny,nz)
      implicit     none
      integer      nx, ny, nz, width, bmini,bmaxi,
     *             bminj,bmaxj, bmink,bmaxk
      real(kind=8) flagarray(nx,ny,nz), value
      !
      integer     i, j, k, num
      logical      double_equal
      external     double_equal
      logical      allowed
      logical      ltrace
      parameter (  ltrace = .false. )

      if (ltrace) then
         write(*,*) 'grow_bbox: value:     ',value
         write(*,*) 'grow_bbox: width:     ',width
         write(*,*) 'grow_bbox: bmin/axi:  ',bmini,bmaxi
         write(*,*) 'grow_bbox: bmin/axj:  ',bminj,bmaxj
         write(*,*) 'grow_bbox: bmin/axk:  ',bmink,bmaxk
         write(*,*) 'grow_bbox: nx/y/z:    ',nx,ny,nz
      end if

      if (.false.  ) then
         !
         ! No disallowed points to worry about:
         !
         bmini = bmini - width
         bmaxi = bmaxi + width
         if (bmini .lt.  1) bmini =  1
         if (bmaxi .gt. nx) bmaxi = nx
         !
         bminj = bminj - width
         bmaxj = bmaxj + width
         if (bminj .lt.  1) bminj =  1
         if (bmaxj .gt. ny) bmaxj = ny
         !
         bmink = bmink - width
         bmaxk = bmaxk + width
         if (bmink .lt.  1) bmink =  1
         if (bmaxk .gt. nz) bmaxk = nz
         !
      else
         !
         ! Try to expand out "width" points
         ! but do not include disallowed points
         !
         ! X min:
         !
         num     = 0
   10    i       = bmini - num - 1
         if (i.lt. 1) goto 15
         allowed = .true.
         do k = bmink, bmaxk
            do j = bminj, bmaxj
               if (double_equal(flagarray(i,j,k),value)) then
                  allowed = .false.
               end if
            end do
         end do
         if (allowed) then
            ! Increase the number by which we can grow box:
            num = num + 1
            if (num .lt. width) goto 10
         end if
   15    continue
         bmini = bmini - num
         if (ltrace) write(*,*) 'grow_bbox: Changing bmini: ',bmini,num
         !
         ! X max:
         !
         num     = 0
   20    i       = bmaxi + num + 1
         if (i.gt.nx) goto 25
         allowed = .true.
         do k = bmink, bmaxk
            do j = bminj, bmaxj
               if (double_equal(flagarray(i,j,k),value)) then
                  allowed = .false.
               end if
            end do
         end do
         if (allowed) then
            ! Increase the number by which we can grow box:
            num = num + 1
            if (num .lt. width) goto 20
         end if
   25    continue
         bmaxi = bmaxi + num
         if (ltrace) write(*,*) 'grow_bbox: Changing bmaxi: ',bmaxi,num
         !
         ! Y min:
         !
         num     = 0
   30    j       = bminj - num - 1
         if (j.lt. 1) goto 35
         allowed = .true.
         do k = bmink, bmaxk
            do i = bmini, bmaxi
               if (double_equal(flagarray(i,j,k),value)) then
                  allowed = .false.
               end if
            end do
         end do
         if (allowed) then
            ! Increase the number by which we can grow box:
            num = num + 1
            if (num .lt. width) goto 30
         end if
   35    continue
         bminj = bminj - num
         if (ltrace) write(*,*) 'grow_bbox: Changing bminj: ',bminj,num
         !  
         ! Y max:
         !
         num     = 0
   40    j       = bmaxj + num + 1
         if (j.gt.ny) goto 45
         allowed = .true.
         do k = bmink, bmaxk
            do i = bmini, bmaxi
               if (double_equal(flagarray(i,j,k),value)) then
                  allowed = .false.
               end if
            end do
         end do
         if (allowed) then
            ! Increase the number by which we can grow box:
            num = num + 1
            if (num .lt. width) goto 40
         end if
   45    continue
         bmaxj = bmaxj + num
         if (ltrace) write(*,*) 'grow_bbox: Changing bmaxj: ',bmaxj,num
         !  
         ! Z min:
         !
         num     = 0
   50    k       = bmink - num - 1
         if (k.lt. 1) goto 55
         allowed = .true.
         do j = bminj, bmaxj
            do i = bmini, bmaxi
               if (double_equal(flagarray(i,j,k),value)) then
                  allowed = .false.
               end if
            end do
         end do
         if (allowed) then
            ! Increase the number by which we can grow box:
            num = num + 1
            if (num .lt. width) goto 50
         end if
   55    continue
         bmink = bmink - num
         if (ltrace) write(*,*) 'grow_bbox: Changing bmink: ',bmink,num
         !
         ! Z max:
         !
         num     = 0
   60    k       = bmaxk + num + 1
         if (k.gt.nz) goto 65
         allowed = .true.
         do j = bminj, bmaxj
            do i = bmini, bmaxi
               if (double_equal(flagarray(i,j,k),value)) then
                  allowed = .false.
               end if
            end do
         end do
         if (allowed) then
            ! Increase the number by which we can grow box:
            num = num + 1
            if (num .lt. width) goto 60
         end if
   65    continue
         bmaxk = bmaxk + num
         if (ltrace) write(*,*) 'grow_bbox: Changing bmaxk: ',bmaxk,num
         !
      end if

      return
      end      ! END: grow_bbox

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc                                                                            cc
cc  double_equal:                                                             cc
cc                                                                            cc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      logical function double_equal( first, second )
      implicit    none
      real(kind=8)      first, second
      include       'largesmall.inc'
      !real(kind=8)      SMALLNUMBER
      !parameter       ( SMALLNUMBER = 1.0d-14)
      logical           ltrace
      parameter       ( ltrace      = .false. )

      double_equal = abs(first-second) .lt. SMALLNUMBER

      if (ltrace) then
         write(*,*)'double_equal:first,second',first,second,double_equal
         write(*,*)'double_equal:diff ',abs(first-second),SMALLNUMBER
         write(*,*)'double_equal:log',abs(first-second).lt.SMALLNUMBER
      end if

      return
      end    ! END: double_equal

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc                                                                            cc
cc  load_scal_mult3d:                                                         cc
cc                    Load scalar*fieldin to fieldout.                        cc
cc                                                                            cc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine load_scal_mult3d( fieldout, fieldin, scalar, nx,ny,nz)
      implicit   none
      integer    nx, ny, nz
      real(kind=8)     fieldout(nx,ny,nz), fieldin(nx,ny,nz), scalar
      integer    i,j,k

      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               fieldout(i,j,k) = scalar * fieldin(i,j,k)
            end do
         end do
      end do

      return
      end    ! END: load_scal_mult3d


cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc                                                                            cc
cc  level_makeflag:                                                           cc
cc                 Make flag array for level:                                 cc
cc                     (1) Initialize flag array to DISALLOW                  cc
cc                     (2) If error exceeds threshold, flag                   cc
cc                     (3) Everywhere a refined grid exists, flag             cc
cc                     (4) If any points are flagged, flag where masked       cc
cc                     (5) Add a buffer region of flagged points              cc
cc                     (6) Disallow ghostregion of level                      cc
cc                                                                            cc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine level_makeflag_simple( flag, error, level,
     *                           minx,miny,minz, h,nx,ny,nz,
     *                           ethreshold,ghostwidth)
      implicit none
      integer       nx,ny,nz, level, ghostwidth
      real(kind=8)  minx,miny,minz, h, ethreshold
      real(kind=8)  flag(nx,ny,nz), error(nx,ny,nz)
      include      'largesmall.inc'
      integer       gi, numpoints
      integer       i, j, k, l
      integer       mini,maxi, minj,maxj, mink,maxk
      integer       bi,ei, bj,ej, bk,ek
      integer       resolution, izero,jzero, kzero
      logical       inmaskedregion, unresolved, cancellevel
      logical       childlevelexists
      real(kind=8)  r1_2, x,y,z
      logical       double_equal
      external      double_equal
      real(kind=8)  myl2norm3d, errornorm
      external      myl2norm3d

      !
      ! Minimum number of grid points in any direction for a masked region:
      !    (in units of number of grid points)
      !    (set to 0 to turn off this "feature")
      !
      integer       MINRESOLUTION
      parameter (   MINRESOLUTION = 13 )
      !parameter (   MINRESOLUTION = 8 )
      !
      ! Width of buffer region around a masked region to refine:
      !    (in units of number of grid points)
      !
      integer       MASKBUFFER
      parameter (   MASKBUFFER = 5 )

      logical      ltrace
      parameter (  ltrace  = .false.)
      logical      ltrace2
      parameter (  ltrace2 = .false.)

      real(kind=8)   FLAG_REFINE
      parameter    ( FLAG_REFINE   =  1.d0 )
      !     ...points NOT to be refined
      real(kind=8)   FLAG_NOREFINE
      parameter    ( FLAG_NOREFINE =  0.d0 )
      !     ...points which do not exist on the level
      real(kind=8)   FLAG_DISALLOW
      parameter    ( FLAG_DISALLOW = -1.d0 )

      if (ltrace) write(*,*) 'level_makeflag: Initializing flag: ',level

      call load_scal3d(flag, FLAG_DISALLOW, nx,ny,nz )

      !
      ! Keep track of bounding box for FLAG_REFINE points:
      !
      numpoints = 0

      !
      ! If certain bad conditions, then we will not
      ! create this level
      !
      cancellevel      = .false.
      childlevelexists = .false.

      mini = nx
      minj = ny
      mink = nz
      maxi = 1
      maxj = 1
      maxk = 1

      if (ltrace) write(*,*) 'level_makeflag: Flagging where high error'
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               if (error(i,j,k) .ge. ethreshold-SMALLNUMBER) then
                  flag(i,j,k) = FLAG_REFINE
                  numpoints   = numpoints + 1
                  if(.false.)write(*,*)'level_makeflag: Flagged: ',
     *                                                i,j,k,error(i,j,k)
                  if (i.lt.mini) mini = i
                  if (j.lt.minj) minj = j
                  if (k.lt.mink) mink = k
                  if (i.gt.maxi) maxi = i
                  if (j.gt.maxj) maxj = j
                  if (k.gt.maxk) maxk = k
               else if (error(i,j,k) .ge. 0-SMALLNUMBER) then
                  flag(i,j,k) = FLAG_NOREFINE
               else if (ltrace2) then
                  if (NINT(error(i,j,k)).ne.-1) then
                     write(*,*) 'level_makeflag: Disallowed pt: ',
     *                                         error(i,j,k),i,j,k
                  end if
               end if
            end do
         end do
      end do
      if (ltrace2) then
         write(*,*) 'level_makeflag:   ethreshold: ',ethreshold
         write(*,*) 'level_makeflag:   numpoints flagged: ',numpoints
         write(*,*) 'level_makeflag:   mini/j/k:',mini,minj,mink
         write(*,*) 'level_makeflag:   maxi/j/k:',maxi,maxj,maxk
         write(*,*) 'level_makeflag:   nx/y/z:  ',nx,ny,nz
      end if

      if (ltrace) write(*,*) 'level_makeflag: Done on level ',level

      return
      end        ! END: level_makeflag

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc                                                                            cc
cc  level_makeflag:                                                           cc
cc                 Make flag array for level:                                 cc
cc                     (1) Initialize flag array to DISALLOW                  cc
cc                     (2) If error exceeds threshold, flag                   cc
cc                     (3) Everywhere a refined grid exists, flag             cc
cc                     (4) If any points are flagged, flag where masked       cc
cc                     (5) Add a buffer region of flagged points              cc
cc                     (6) Disallow ghostregion of level                      cc
cc                                                                            cc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine level_makeflag( flag, error, level,
     *                           minx,miny,minz, h,nx,ny,nz,
     *                           ethreshold,num_masks,max_num_masks,
     *                           mask_coords,ghostwidth,
     *                           levelp,allowedl,numboxes,
     *                           gr_minx,gr_maxx,
     *                           gr_miny,gr_maxy,
     *                           gr_minz,gr_maxz,
     *                           gr_sibling,bh_true,
     *                           assume_symmetry,buffer)
      implicit none
      integer       nx,ny,nz, level,ghostwidth
      integer       num_masks,max_num_masks
      integer       allowedl
      integer       levelp(allowedl)
      integer       numboxes,buffer
      integer       assume_symmetry
      logical       bh_true(max_num_masks) 
      integer       gr_sibling(numboxes)
      real(kind=8)  gr_minx(numboxes),gr_maxx(numboxes)
      real(kind=8)  gr_miny(numboxes),gr_maxy(numboxes)
      real(kind=8)  gr_minz(numboxes),gr_maxz(numboxes)
      real(kind=8)  mask_coords(6,max_num_masks)
      real(kind=8)  minx,miny,minz, h, ethreshold
      real(kind=8)  flag(nx,ny,nz), error(nx,ny,nz)
      include       'largesmall.inc'
      integer       gi, numpoints
      integer       i, j, k, l
      integer       mini,maxi, minj,maxj, mink,maxk
      integer       bi,ei, bj,ej, bk,ek
      integer       resolution, izero,jzero, kzero
      logical       inmaskedregion, unresolved, cancellevel
      logical       childlevelexists
      real(kind=8)  r1_2, x,y,z
      logical       double_equal
      external      double_equal

      !
      ! Minimum number of grid points in any direction for a masked region:
      !    (in units of number of grid points)
      !    (set to 0 to turn off this "feature")
      !
      integer       MINRESOLUTION
      parameter (   MINRESOLUTION = 13 )
      !parameter (   MINRESOLUTION = 8 )
      !
      ! Width of buffer region around a masked region to refine:
      !    (in units of number of grid points)
      !
      integer       MASKBUFFER
      parameter (   MASKBUFFER = 5 )

      real(kind=8)   FLAG_REFINE
      parameter    ( FLAG_REFINE   =  1.d0 )
      !     ...points NOT to be refined
      real(kind=8)   FLAG_NOREFINE
      parameter    ( FLAG_NOREFINE =  0.d0 )
      !     ...points which do not exist on the level
      real(kind=8)   FLAG_DISALLOW
      parameter    ( FLAG_DISALLOW = -1.d0 )

      logical      ltrace
      parameter (  ltrace  = .false.)
      logical      ltrace2
      parameter (  ltrace2 = .false.)


      if (ltrace) write(*,*) 'level_makeflag: Initializing flag: ',level

      call load_scal3d(flag, FLAG_DISALLOW, nx,ny,nz )

      !
      ! Keep track of bounding box for FLAG_REFINE points:
      !
      numpoints = 0

      !
      ! If certain bad conditions, then we will not
      ! create this level
      !
      cancellevel      = .false.
      childlevelexists = .false.

      mini = nx
      minj = ny
      mink = nz
      maxi = 1
      maxj = 1
      maxk = 1

      if (ltrace) write(*,*) 'level_makeflag: Flagging where high error'
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               if (error(i,j,k) .ge. ethreshold-SMALLNUMBER) then
                  flag(i,j,k) = FLAG_REFINE
                  numpoints   = numpoints + 1
                  if(.false.)write(*,*)'level_makeflag: Flagged: ',
     *                                                i,j,k,error(i,j,k)
                  if (i.lt.mini) mini = i
                  if (j.lt.minj) minj = j
                  if (k.lt.mink) mink = k
                  if (i.gt.maxi) maxi = i
                  if (j.gt.maxj) maxj = j
                  if (k.gt.maxk) maxk = k
               else if (error(i,j,k) .ge. 0-SMALLNUMBER) then
                  flag(i,j,k) = FLAG_NOREFINE
               else if (ltrace2) then
                  if (NINT(error(i,j,k)).ne.-1) then
                     write(*,*) 'level_makeflag: Disallowed pt: ',
     *                                         error(i,j,k),i,j,k
                  end if
               end if
            end do
         end do
      end do
      bi = mini
      ei = maxi
      bj = minj
      ej = maxj
      bk = mink
      ek = maxk

      if (ltrace) write(*,*) 'level_makeflag: Flagging where refined'
      gi = levelp(level+2)
      if (ltrace2)write(*,*) 'level_makeflag: Starting w/ gi: ',gi
      !
 10   if ( gi .ne. -1) then
         childlevelexists = .true.
         if (ltrace) write(*,*) 'level_makeflag:   ...grid ',gi
         !
         mini = 1 + NINT( (gr_minx(gi)-minx)/h )
         minj = 1 + NINT( (gr_miny(gi)-miny)/h )
         mink = 1 + NINT( (gr_minz(gi)-minz)/h )
         ! Because resolutions are different, need to make sure
         ! that the full grid is "covered":
         if (minx+(mini-1)*h .gt. gr_minx(gi).and.mini.gt.1) mini=mini-1
         if (miny+(minj-1)*h .gt. gr_miny(gi).and.minj.gt.1) minj=minj-1
         if (minz+(mink-1)*h .gt. gr_minz(gi).and.mink.gt.1) mink=mink-1
         !
         maxi = 1 + NINT( (gr_maxx(gi)-minx)/h )
         maxj = 1 + NINT( (gr_maxy(gi)-miny)/h )
         maxk = 1 + NINT( (gr_maxz(gi)-minz)/h )
         if (minx+(maxi-1)*h .lt. gr_maxx(gi).and.maxi.lt.nx)maxi=maxi+1
         if (miny+(maxj-1)*h .lt. gr_maxy(gi).and.maxj.lt.ny)maxj=maxj+1
         if (minz+(maxk-1)*h .lt. gr_maxz(gi).and.maxk.lt.nz)maxk=maxk+1
         !
         if (ltrace) then
             write(*,*) 'level_makeflag:   ...looping'
             write(*,*) 'level_makeflag:   mini/j/k:',mini,minj,mink
             write(*,*) 'level_makeflag:   maxi/j/k:',maxi,maxj,maxk
             write(*,*) 'level_makeflag: h: ',h
             write(*,*) 'level_makeflag:    minx/y/z:',minx,
     *                  miny,minz
             write(*,*) 'level_makeflag: gr_minx/y/z:',gr_minx(gi),
     *                  gr_miny(gi),gr_minz(gi)
             write(*,*) 'level_makeflag: gr_maxx/y/z:',gr_maxx(gi),
     *                  gr_maxy(gi),gr_maxz(gi)
             write(*,*) 'level_makeflag:  lower x:',minx+(mini-1)*h
             write(*,*) 'level_makeflag:  upper x:',minx+(maxi-1)*h
          end if
         do k = mink, maxk
            do j = minj, maxj
               do i = mini, maxi
                  if( NINT(flag(i,j,k)).ne.FLAG_REFINE)
     *                       numpoints   = numpoints + 1
                  flag(i,j,k) = FLAG_REFINE
                  if (i.lt.bi) bi = i
                  if (i.gt.ei) ei = i
                  if (j.lt.bj) bj = j
                  if (j.gt.ej) ej = j
                  if (k.lt.bk) bk = k
                  if (k.gt.ek) ek = k
               end do
            end do
         end do
         !
         gi = gr_sibling(gi)
         goto 10
      end if

      !
      ! Force refinement in masked region(s) if:
      !     ---any points have already been flagged
      !     ---if resolution of masked region too poor
      !
      if (ltrace)write(*,*) 'level_makeflag: Are there masked regions?'
      if (ltrace)write(*,*) 'level_makeflag: num_masks = ',num_masks
      if (num_masks.gt.0) then
         if (ltrace)write(*,*) 'level_makeflag: Masked region(s) exist'
         ! Only flag the masked region if the level has flagged points:
         !
         unresolved = .false.
         do l = 1, max_num_masks
            if (bh_true(l)) then
               resolution = min(
     *                     NINT(mask_coords(2,l)-mask_coords(1,l))/h,
     *                     NINT(mask_coords(4,l)-mask_coords(3,l))/h,
     *                     NINT(mask_coords(6,l)-mask_coords(5,l))/h
     *                          )
               unresolved = unresolved .or. resolution.lt.MINRESOLUTION
               if (numpoints.eq.0.and.unresolved) then
               !if (ltrace.or. (numpoints.eq.0.and.unresolved)) then
               write(*,*) 'level_makeflag: Forcing new level to resolve'
               write(*,*) 'level_makeflag: Hole #:       ',l
               !write(*,*) 'level_makeflag: numpoints:    ',numpoints
               write(*,*) 'level_makeflag: resolution:   ',resolution
               !write(*,*) 'level_makeflag: unresolved:   ',unresolved
               write(*,*) 'level_makeflag: MINRESOLUTION:',MINRESOLUTION
               end if
            end if
         end do
         !
         if ( numpoints .gt. 0 .or. unresolved) then
            if(ltrace)write(*,*)'level_makeFlagged pnts exist',numpoints
            do k = 1, nz
               z = minz + h * (k-1)
            do j = 1, ny
               y = miny + h * (j-1)
            do i = 1, nx
               x = minx + h * (i-1)
               inmaskedregion = .false.
               do l = 1, max_num_masks
                  if (bh_true(l)) then
                     !
                     ! Add buffer region around mask:
                     !
                     inmaskedregion = inmaskedregion .or.
     *          (      ( x .ge. (mask_coords(1,l)-MASKBUFFER*h) ) .and.
     *                 ( x .le. (mask_coords(2,l)+MASKBUFFER*h) ) .and.
     *                 ( y .ge. (mask_coords(3,l)-MASKBUFFER*h) ) .and.
     *                 ( y .le. (mask_coords(4,l)+MASKBUFFER*h) ) .and.
     *                 ( z .ge. (mask_coords(5,l)-MASKBUFFER*h) ) .and.
     *                 ( z .le. (mask_coords(6,l)+MASKBUFFER*h) ) )
                     !
                  end if
               end do
               if (inmaskedregion) then
                  ! This point should       be refined:
                  if( NINT(flag(i,j,k)).ne.FLAG_REFINE)
     *                       numpoints   = numpoints + 1
                  flag(i,j,k) = 1.d0 * FLAG_REFINE
                  !
                  ! If the masked region would occur too close
                  ! to the boundary of a new level then 
                  ! let us not even create the level:
                  !
                  if ( i.le.ghostwidth .or. i.gt.nx-ghostwidth .or.
     *                 j.le.ghostwidth .or. j.gt.ny-ghostwidth .or.
     *                 k.le.ghostwidth .or. k.gt.nz-ghostwidth ) then
                     cancellevel = .true.
                  end if
               end if
            end do
            end do
            end do
            if (ltrace)
     *      write(*,*)'level_makeflag: Total num flagged pts ',numpoints
         end if
      else
         if (ltrace)write(*,*) 'level_makeflag: No masked regions'
      end if

      if (ltrace) write(*,*) 'level_makeflag: Buffering flag array'
      if (ltrace2)write(*,*) 'level_makeflag:     buffer = ',buffer
      !
      ! Use "error" array as temporary storage for this routine:
      !
      call mat_buffer( flag, error,  buffer, nx,ny,nz)

      !
      ! Disallow clusters at the boundaries of the level
      !  NB:  this is where the level gets boundary data from its parent
      !  NB2: 
      !
      if (ltrace) write(*,*) 'level_makeflag: NOrefining boundaries'

      ! x-boundaries
      if(  minz.gt.0  .or.
     *     (assume_symmetry.ne.1.and.assume_symmetry.ne.6) )then
         do k = 1, nz
            do j = 1, ny
               do i = 1, ghostwidth
                  flag(i,j,k) = FLAG_DISALLOW
               end do
            end do
         end do
      else
         if (ltrace) then
         write(*,*)'level_makeflag: Allowing min z boundary'
         write(*,*)'level_makeflag: assume_symmetry=',assume_symmetry
         end if
         ! Allow flagging only two coarse grid points less than zero
         !        (NB: 4 fine grid points for extended boundary)
         izero   = -NINT(minx/h) + 1
         do k = 1, nz
            do j = 1, ny
               do i = 1,izero-3
                  flag(i,j,k) = FLAG_DISALLOW
               end do
            end do
         end do
      end if
      do k = 1, nz
         do j = 1, ny
            do i = nx-ghostwidth+1, nx
               flag(i,j,k) = FLAG_DISALLOW
            end do
         end do
      end do
      ! y-boundaries
      if(  minz.gt.0  .or.
     *     (assume_symmetry.ne.2.and.assume_symmetry.ne.6) )then
         do k = 1, nz
            do i = 1, nx
               do j = 1, ghostwidth
                  flag(i,j,k) = FLAG_DISALLOW
               end do
            end do
         end do
      else
         if (ltrace) then
         write(*,*)'level_makeflag: Allowing min z boundary'
         write(*,*)'level_makeflag: assume_symmetry=',assume_symmetry
         end if
         ! Allow flagging only two coarse grid points less than zero
         !        (NB: 4 fine grid points for extended boundary)
         jzero   = -NINT(miny/h) + 1
         do k = 1, nz
            do i = 1, nx
               do j = 1,jzero-3
                  flag(i,j,k) = FLAG_DISALLOW
               end do
            end do
         end do
      end if
      do k = 1, nz
         do i = 1, nx
            do j = ny-ghostwidth+1, ny
               flag(i,j,k) = FLAG_DISALLOW
            end do
         end do
      end do

      !
      ! z-boundaries
      !
      !    NB: If using reflection symmetry about z
      !        we do not want to disallow:
      !if (.not.(assume_symmetry.eq.3.and.minz .le. 0.d0) )then
      if(  minz.gt.0  .or.
     *     (assume_symmetry.ne.3.and.assume_symmetry.ne.6) )then
         if (ltrace) write(*,*)'level_makeflag: Disallowing min z bndry'
         do j = 1, ny
            do i = 1, nx
               do k = 1, ghostwidth
                  flag(i,j,k) = FLAG_DISALLOW
               end do
            end do
         end do
      else
         if (ltrace) then
         write(*,*)'level_makeflag: Allowing min z boundary'
         write(*,*)'level_makeflag: assume_symmetry=',assume_symmetry
         write(*,*)'level_makeflag: minz           =',minz
         end if
         ! Allow flagging only two coarse grid points less than zero
         !        (NB: 4 fine grid points for extended boundary)
         kzero   = -NINT(minz/h) + 1
         do j = 1, ny
            do i = 1, nx
               do k = 1,kzero-3
                  flag(i,j,k) = FLAG_DISALLOW
               end do
            end do
         end do
      end if

      do j = 1, ny
        do i = 1, nx
          do k = nz-ghostwidth+1, nz
            flag(i,j,k) = FLAG_DISALLOW
          end do
        end do
      end do

      if (cancellevel) then
         if (childlevelexists) then
            write(*,*)'level_makeflag: --------------------------'
            write(*,*)'level_makeflag: level = ',level
            write(*,*)'level_makeflag: Cannot cancel level because'
            write(*,*)'level_makeflag: child level exists, but a masked'
            write(*,*)'level_makeflag: region is too close to boundary'
            write(*,*)'level_makeflag: --------------------------'
         else
            write(*,*) 'level_makeflag: Cancelling, level: ',level
            call load_scal1D(flag,1.d0*FLAG_NOREFINE,nx*ny*nz)
         end if
      end if

      if (ltrace) write(*,*) 'level_makeflag: Done on level ',level

      return
      end        ! END: level_makeflag


cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc                                                                            cc
cc  load_scal3d:                                                              cc
cc                                                                            cc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine load_scal3d( field, scalar, nx, ny, nz)
      implicit   none
      integer    nx, ny, nz
      real(kind=8)     field(nx,ny,nz), scalar
      integer    i,j,k

      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               field(i,j,k) = scalar
            end do
         end do
      end do

      return
      end    ! END: load_scal3d

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc                                                                            cc
cc  mat_buffer:                                                               cc
cc              Takes a flag array, and flags points withing "buffer" pts     cc
cc              NB: if a point is disallowed, it will not get buffered/flaggedcc
cc                                                                            cc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine mat_buffer( flag, work, buffer, nx, ny, nz)
      implicit   none
      integer    nx, ny, nz, buffer
      real(kind=8)     flag(nx,ny,nz), work(nx,ny,nz)
      integer    i,j,k, ii,jj,kk

      !
      ! Copy flag to work array:
      !
      call mat_copy3d( flag, work, nx,ny,nz)

      !
      ! Remake flag array with buffer:
      !
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               if ( NINT(work(i,j,k)) .eq. 1) then
                  do kk = k-buffer, k+buffer
                  do jj = j-buffer, j+buffer
                  do ii = i-buffer, i+buffer
                     if (ii.ge.1 .and. ii.le.nx .and.
     *                   jj.ge.1 .and. jj.le.ny .and.
     *                   kk.ge.1 .and. kk.le.nz ) then
!    *                   NINT(work(ii,jj,kk)).ge.0) then
                         if (NINT(work(ii,jj,kk)).ge.0)
     *                      flag(ii,jj,kk) = work(i,j,k)
                     end if
                  end do
                  end do
                  end do
               end if
            end do
         end do
      end do

      return
      end    ! END: mat_buffer

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc                                                                            cc
cc  mat_copy1d:                                                               cc
cc                                                                            cc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine mat_copy1d( field_n, field_np1, nx)
      implicit   none
      integer    nx
      real(kind=8)     field_n(nx), field_np1(nx)
      integer    i

            do i = 1, nx
               field_np1(i) = field_n(i)
            end do

      return
      end    ! END: mat_copy1d

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc                                                                            cc
cc  mat_copy3d:                                                               cc
cc                                                                            cc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine mat_copy3d( field_n, field_np1, nx, ny, nz)
      implicit   none
      integer    nx, ny, nz
      real(kind=8)     field_n(nx,ny,nz), field_np1(nx,ny,nz)
      integer    i,j,k

      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               field_np1(i,j,k) = field_n(i,j,k)
            end do
         end do
      end do

      return
      end    ! END: mat_copy3d

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc                                                                            cc
cc  level_clusterDD:                                                          cc
cc                  Take as input a the set of subgrids to create             cc
cc                  And output that same set but domain decomposed            cc
cc                  according to the number of processors.                    cc
cc                                                                            cc
cc                  NB: to help ensure that the chopped up grids              cc
cc                      get placed onto all different processors,             cc
cc                      the list should be kept in order instead of           cc
cc                      putting the new blocks at the end of the list.        cc
cc                      Hence the need for temp storage. Reals are used       cc
cc                      because such space is readily available in            cc
cc                      level_refine().                                       cc
cc                                                                            cc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine level_clusterdd(tmp_mini, tmp_maxi, tmp_minj, tmp_maxj,
     *                           tmp_mink, tmp_maxk, b_mini,   b_maxi,
     *                           b_minj,   b_maxj,   b_mink,   b_maxk,
     *                           numbox, nump, maxnum,
     *                           ghostwidth,refine_factor,mindim,
     *                           bound_width)
      implicit     none
      integer      ghostwidth,refine_factor,mindim,bound_width
      integer      numbox, nump, maxnum
      real(kind=8) tmp_mini(maxnum), tmp_maxi(maxnum),
     *             tmp_minj(maxnum), tmp_maxj(maxnum),
     *             tmp_mink(maxnum), tmp_maxk(maxnum)
      integer      b_mini(maxnum),   b_maxi(maxnum),
     *             b_minj(maxnum),   b_maxj(maxnum),
     *             b_mink(maxnum),   b_maxk(maxnum)
      integer      ii,   i, j, k, dims(3), gwc
      integer      lengthi, lengthj, lengthk
      integer      numx, numy, numz
      ! Offsets to overlap decomposed grids:
      integer      minus, plus
      ! Minimum dimension into which to cut:
      integer      minsize
      integer      boxcount, fixcount
      integer      fixminx, fixminy, fixminz, fixmaxx, fixmaxy, fixmaxz
      logical      fixshow

      logical      ltrace
      parameter (  ltrace  = .false. )

      !
      ! Each cluster which we are trying to split among
      ! the procs should have at least ghostwidth+mindim
      ! points in each direction. If we split, then each
      !
      minsize = ghostwidth / refine_factor + mindim

      ! The ghostwidth as it appears on the parent level:
      gwc  = ghostwidth / refine_factor + 1

      if (ltrace) then
         write(*,*) '********************************'
         write(*,*) 'level_clusterDD:  numbox     = ',numbox
         write(*,*) 'level_clusterDD:  mindim     = ',mindim
         write(*,*) 'level_clusterDD:  ghostwidth = ',ghostwidth
         write(*,*) 'level_clusterDD:  gwc        = ',gwc
         write(*,*) 'level_clusterDD:  minsize    = ',minsize
      end if

      !            
      ! Copy entries to temp storage:
      !            
      do ii = 1, numbox
         tmp_mini(ii) = b_mini(ii)
         tmp_maxi(ii) = b_maxi(ii)
         tmp_minj(ii) = b_minj(ii)
         tmp_maxj(ii) = b_maxj(ii)
         tmp_mink(ii) = b_mink(ii)
         tmp_maxk(ii) = b_maxk(ii)
      end do

      call my_dims_create( nump, dims)
      !if (nump .ne. 64) then
      !   dims(1) = 0      !  x: LEFT   and RIGHT
      !   dims(2) = 0      !  y: BACK   and FRONT
      !   dims(3) = 0      !  z: BOTTOM and TOP
      !   call MPI_DIMS_CREATE( nump, 3, dims, ierr )
      !else
      !   ! For some strange reason above function yields 8x4x2
      !   ! instead of 4x4x4, so bypass
      !   dims(1) = 4
      !   dims(2) = 4
      !   dims(3) = 4
      !end if
      if (ltrace) write(*,*) 'level_clusterDD: dims1/2/3:',
     *                             dims(1),dims(2),dims(3)


      !            
      ! For each box, break up into nump pieces:
      !            
      boxcount = 0
      do ii = 1, numbox
         ! Keep track if clustering needs to be "fixed":
         fixminx = 0
         fixminy = 0
         fixminz = 0
         fixmaxx = 0
         fixmaxy = 0
         fixmaxz = 0
         fixcount= boxcount
         lengthi = NINT( (tmp_maxi(ii) - tmp_mini(ii) ) / dims(1) )
         lengthj = NINT( (tmp_maxj(ii) - tmp_minj(ii) ) / dims(2) )
         lengthk = NINT( (tmp_maxk(ii) - tmp_mink(ii) ) / dims(3) )
         if (ltrace) then
            write(*,*) 'level_clusterDD:  Splitting box: ',ii
            write(*,*) 'level_clusterDD:  i: ',tmp_mini(ii),tmp_maxi(ii)
            write(*,*) 'level_clusterDD:  j: ',tmp_minj(ii),tmp_maxj(ii)
            write(*,*) 'level_clusterDD:  k: ',tmp_mink(ii),tmp_maxk(ii)
            write(*,*) 'level_clusterDD:    lengthi/j/k: ',
     *                  lengthi, lengthj, lengthk
         end if
         !
         ! Decide how we should chop up:
         !    (1) Chop as per MPI_DIMS_Create() says if possible
         !        (ie if that would produce sufficient size boxes)
         !    (2) Else chop into "minsize" pieces
         !
         if (lengthi .lt. minsize .and. dims(1).gt.1) then
         !if (lengthi .lt. mindim .and. dims(1).gt.1) then
            numx = (tmp_maxi(ii)-tmp_mini(ii)) / minsize
            if (numx .le. 0) numx = 1
            lengthi = NINT( (tmp_maxi(ii) - tmp_mini(ii) ) / numx )
         else
            numx = dims(1)
         end if
         if (lengthj .lt. minsize .and. dims(2).gt.1) then
            numy = (tmp_maxj(ii)-tmp_minj(ii)) / minsize
            if (numy .le. 0) numy = 1
            lengthj = NINT( (tmp_maxj(ii) - tmp_minj(ii) ) / numy )
         else
            numy = dims(2)
         end if
         if (lengthk .lt. minsize .and. dims(3).gt.1) then
            numz = (tmp_maxk(ii)-tmp_mink(ii)) / minsize
            if (numz .le. 0) numz = 1
            if (numx*numy*numz.gt.nump) numz = nump / numx / numy 
            lengthk = NINT( (tmp_maxk(ii) - tmp_mink(ii) ) / numz )
         else
            numz = dims(3)
         end if
         if (ltrace) write(*,*) '   split numbers:',
     *                       numx,numy,numz,numx*numy*numz,nump
         !
         ! If the boxes need splitting and we have enough
         ! children to accomodate:
         !
         if ( (numx.gt.1.or.numy.gt.1.or.numz.gt.1)
     *                       .and. 
     *         (maxnum .ge. (boxcount+numx*numy*numz)) ) then
         !if (numx.gt.1.or.numy.gt.1.or.numz.gt.1) then
            !
            ! Split up box "ii":
            !
            do k = 1, numz
            do j = 1, numy
            do i = 1, numx      
               boxcount         = boxcount + 1 
               b_mini(boxcount) = tmp_mini(ii) + (i-1) * lengthi
               b_maxi(boxcount) = tmp_mini(ii) +   i   * lengthi
               b_minj(boxcount) = tmp_minj(ii) + (j-1) * lengthj
               b_maxj(boxcount) = tmp_minj(ii) +   j   * lengthj
               b_mink(boxcount) = tmp_mink(ii) + (k-1) * lengthk
               b_maxk(boxcount) = tmp_mink(ii) +   k   * lengthk
               !
               ! Force overlap:
               !    NB: The amount of overlap among the grids
               !    depends on the stencils used and how
               !    the parameter bwidth is set in util.f.
               !    Also, keep in mind that we're working
               !    in the coarse grid space in which the
               !    grids are defined but the grids to be
               !    completed will have greater resolution.
               !    Assuming a minimum refinement factor of 2,
               !    then we need to overlap with 4 of these
               !    coarse points so that the grids themselves
               !    share 2*bwidth points. 
               !
               plus  = bound_width / 2
               if (mod(bound_width,2) .eq. 0) then
                  minus = plus
               else
                  minus = plus + 1
               end if
               b_mini(boxcount) = b_mini(boxcount) - minus
               b_maxi(boxcount) = b_maxi(boxcount) + plus
               b_minj(boxcount) = b_minj(boxcount) - minus
               b_maxj(boxcount) = b_maxj(boxcount) + plus
               b_mink(boxcount) = b_mink(boxcount) - minus
               b_maxk(boxcount) = b_maxk(boxcount) + plus
               !
               ! In case, dimensions don't divide evenly
               ! and to overrule the overlap at the bounds,
               ! make sure we cover the whole box:
               !
               if (i.eq. 1    ) b_mini(boxcount) = tmp_mini(ii)
               if (j.eq. 1    ) b_minj(boxcount) = tmp_minj(ii)
               if (k.eq. 1    ) b_mink(boxcount) = tmp_mink(ii)
               if (i.eq. numx ) b_maxi(boxcount) = tmp_maxi(ii)
               if (j.eq. numy ) b_maxj(boxcount) = tmp_maxj(ii)
               if (k.eq. numz ) b_maxk(boxcount) = tmp_maxk(ii)
               !
               ! Must ensure that grids do not terminate within
               ! the ghostregion for the level...that is, each
               ! grid that does not touch the amr boundary on
               ! one of its faces, should not end *close* to
               ! the amr boundary. If one does, one gets points
               ! that are both injected to the parent, and
               ! those same points on the parent are used to
               ! reset the boundary.
               !
               if(ltrace) then
                  write(*,*) '---boxcount= ',boxcount
                  write(*,*) '    i: ',b_mini(boxcount),b_maxi(boxcount)
                  write(*,*) '    j: ',b_minj(boxcount),b_maxj(boxcount)
                  write(*,*) '    k: ',b_mink(boxcount),b_maxk(boxcount)
                  write(*,*) '  . . . '
               end if
               if (b_mini(boxcount).ne.tmp_mini(ii) .and.
     *             b_mini(boxcount).le.tmp_mini(ii)+gwc-1 ) then
                  if(ltrace)write(*,*)'level_clusterDD: fixing small i',
     *                   b_mini(boxcount),tmp_mini(ii)+gwc
                  if ((tmp_mini(ii)+gwc-b_mini(boxcount)) .gt. fixminx)
     *               fixminx = tmp_mini(ii)+gwc-b_mini(boxcount)
                  b_mini(boxcount) = tmp_mini(ii)+gwc
               end if
               if (b_maxi(boxcount).ne.tmp_maxi(ii) .and.
     *             b_maxi(boxcount).ge.tmp_maxi(ii)-gwc+1 ) then
                  if(ltrace)write(*,*)'level_clusterDD: fixing large i',
     *                   b_maxi(boxcount),tmp_maxi(ii)-gwc
                  if ((b_maxi(boxcount)- (tmp_maxi(ii)-gwc).gt.fixmaxx))
     *               fixmaxx = b_maxi(boxcount)- (tmp_maxi(ii)-gwc)
                  b_maxi(boxcount) = tmp_maxi(ii)-gwc
               end if
               if (b_minj(boxcount).ne.tmp_minj(ii) .and.
     *             b_minj(boxcount).le.tmp_minj(ii)+gwc-1 ) then
                  if(ltrace)write(*,*)'level_clusterDD: fixing small j',
     *                   b_minj(boxcount),tmp_minj(ii)+gwc
                  if ( (tmp_minj(ii)+gwc-b_minj(boxcount)) .gt. fixminy)
     *               fixminy = tmp_minj(ii)+gwc-b_minj(boxcount)
                  b_minj(boxcount) = tmp_minj(ii)+gwc
               end if
               if (b_maxj(boxcount).ne.tmp_maxj(ii) .and.
     *             b_maxj(boxcount).ge.tmp_maxj(ii)-gwc+1 ) then
                  if(ltrace)write(*,*)'level_clusterDD: fixing large j',
     *                   b_maxj(boxcount),tmp_maxj(ii)-gwc
                  if ((b_maxj(boxcount)- (tmp_maxj(ii)-gwc).gt.fixmaxy))
     *               fixmaxy = b_maxj(boxcount)- (tmp_maxj(ii)-gwc)
                  b_maxj(boxcount) = tmp_maxj(ii)-gwc
               end if
               if (b_mink(boxcount).ne.tmp_mink(ii) .and.
     *             b_mink(boxcount).le.tmp_mink(ii)+gwc-1 ) then
                  if(ltrace)write(*,*)'level_clusterDD: fixing small k',
     *                   b_mink(boxcount),tmp_mink(ii)+gwc
                  if ( (tmp_mink(ii)+gwc-b_mink(boxcount)) .gt. fixminz)
     *               fixminz = tmp_mink(ii)+gwc-b_mink(boxcount)
                  b_mink(boxcount) = tmp_mink(ii)+gwc
               end if
               if (b_maxk(boxcount).ne.tmp_maxk(ii) .and.
     *             b_maxk(boxcount).ge.tmp_maxk(ii)-gwc+1 ) then
                  if(ltrace)write(*,*)'level_clusterDD: fixing large k',
     *                   b_maxk(boxcount),tmp_maxk(ii)-gwc
                  if ((b_maxk(boxcount)- (tmp_maxk(ii)-gwc).gt.fixmaxz))
     *               fixmaxz = b_maxk(boxcount)- (tmp_maxk(ii)-gwc)
                  b_maxk(boxcount) = tmp_maxk(ii)-gwc
               end if
               !
               if (ltrace) then
                  write(*,*) 'bound_width = ',bound_width
                  write(*,*) 'plus        = ',plus
                  write(*,*) 'minus       = ',minus
                  write(*,*) 'gwc         = ',gwc
                  write(*,*) '   boxcount= ',boxcount
                  write(*,*) '    i: ',b_mini(boxcount),b_maxi(boxcount)
                  write(*,*) '    j: ',b_minj(boxcount),b_maxj(boxcount)
                  write(*,*) '    k: ',b_mink(boxcount),b_maxk(boxcount)
                  write(*,*) '-----------------------------'
               end if
            end do
            end do
            end do
            if (ltrace) then
            !if (ltrace.or. .true.) then
               write(*,*)'level_clusterDD:fixminx/y/z:',fixminx,fixminy,
     *                        fixminz
               write(*,*)'level_clusterDD:fixmaxx/y/z:',fixmaxx,fixmaxy,
     *                        fixmaxz
            end if
            !
            ! Fix the boxes on the border:
            !
            do k = fixcount+1,boxcount
               fixshow = .false.
               if (ltrace) then
                  write(*,*) 'level_clusterDD: Fixing box: ',k
                  write(*,*) '    i: ',b_mini(k),b_maxi(k)
                  write(*,*) '    j: ',b_minj(k),b_maxj(k)
                  write(*,*) '    k: ',b_mink(k),b_maxk(k)
               end if
               if (fixminx.gt.0 .and. b_mini(k).eq.tmp_mini(ii)) then
                  b_maxi(k) = b_maxi(k) + fixminx
                  if (b_maxi(k).gt.tmp_maxi(ii)) b_maxi(k)=tmp_maxi(ii)
                  if(ltrace)write(*,*)'level_clusterDD: fixed minx'
                  fixshow = .true.
               end if
               if (fixmaxx.gt.0 .and. b_maxi(k).eq.tmp_maxi(ii)) then
                  b_mini(k) = b_mini(k) - fixmaxx
                  if (b_mini(k).lt.tmp_mini(ii)) b_mini(k)=tmp_mini(ii)
                  if(ltrace)write(*,*)'level_clusterDD: fixed maxx'
                  fixshow = .true.
               end if
               if (fixminy.gt.0 .and. b_minj(k).eq.tmp_minj(ii)) then
                  b_maxj(k) = b_maxj(k) + fixminy
                  if (b_maxj(k).gt.tmp_maxj(ii)) b_maxj(k)=tmp_maxj(ii)
                  if(ltrace)write(*,*)'level_clusterDD: fixed miny'
                  fixshow = .true.
               end if
               if (fixmaxy.gt.0 .and. b_maxj(k).eq.tmp_maxj(ii)) then
                  b_minj(k) = b_minj(k) - fixmaxy
                  if (b_minj(k).lt.tmp_minj(ii)) b_minj(k)=tmp_minj(ii)
                  if(ltrace)write(*,*)'level_clusterDD: fixed maxy'
                  fixshow = .true.
               end if
               if (fixminz.gt.0 .and. b_mink(k).eq.tmp_mink(ii)) then
                  b_maxk(k) = b_maxk(k) + fixminz
                  if (b_maxk(k).gt.tmp_maxk(ii)) b_maxk(k)=tmp_maxk(ii)
                  if(ltrace)write(*,*)'level_clusterDD: fixed minz'
                  fixshow = .true.
               end if
               if (fixmaxz.gt.0 .and. b_maxk(k).eq.tmp_maxk(ii)) then
                  b_mink(k) = b_mink(k) - fixmaxz
                  if (b_mink(k).lt.tmp_mink(ii)) b_mink(k)=tmp_mink(ii)
                  if(ltrace)write(*,*)'level_clusterDD: fixed maxz'
                  fixshow = .true.
               end if
               if (ltrace .and. fixshow) then
                  write(*,*) 'level_clusterDD: After FIX box: ',k
                  write(*,*) '    i: ',b_mini(k),b_maxi(k)
                  write(*,*) '    j: ',b_minj(k),b_maxj(k)
                  write(*,*) '    k: ',b_mink(k),b_maxk(k)
                  write(*,*) ' ---------------------'
               end if
            end do
         else
            if (ltrace) write(*,*) ' Not cutting box'
            if ( maxnum .lt. (boxcount+numx*numy*numz)) then
               write(*,*) ' Not decomposing anymore grids because of '
               write(*,*) ' hardcoded limit to number of child grids.'
               write(*,*) ' Current limit: maxnum =',maxnum
               write(*,*) ' Consider increasing in had/include/glob.inc'
               write(*,*) '       *  *  *  *  *  *  *'
            end if
            boxcount         = boxcount + 1
            b_mini(boxcount) = tmp_mini(ii)
            b_mini(boxcount) = tmp_mini(ii)
            b_maxi(boxcount) = tmp_maxi(ii)
            b_minj(boxcount) = tmp_minj(ii)
            b_maxj(boxcount) = tmp_maxj(ii)
            b_mink(boxcount) = tmp_mink(ii)
            b_maxk(boxcount) = tmp_maxk(ii)
         end if
      end do

      numbox = boxcount

      if (ltrace) then
         write(*,*) 'level_clusterDD:  numbox = ',numbox
         write(*,*) 'level_clusterDD: DONE.'
         write(*,*) '********************************'
      end if

      return
      end      ! END: level_clusterDD
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc                                                                            cc
cc  my_dims_create                                                            cc
cc                                                                            cc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine my_dims_create( nump, dims)
      implicit   none
      integer    nump
      integer    dims(3)

      if ( nump .eq. 1 ) then
        dims = (/1,1,1/)
      else if ( nump .eq. 2 ) then
        dims = (/2,1,1/)
      else if ( nump .eq. 3 ) then
        dims = (/3,1,1/)
      else if ( nump .eq. 4 ) then
        dims = (/2,2,1/)
      else if ( nump .eq. 5 ) then
        dims = (/5,1,1/)
      else if ( nump .eq. 6 ) then
        dims = (/3,2,1/)
      else if ( nump .eq. 7 ) then
        dims = (/7,1,1/)
      else if ( nump .eq. 8 ) then
        dims = (/2,2,2/)
      else if ( nump .eq. 9 ) then
        dims = (/3,3,1/)
      else if ( nump .eq. 10 ) then
        dims = (/5,2,1/)
      else if ( nump .eq. 11 ) then
        dims = (/11,1,1/)
      else if ( nump .eq. 12 ) then
        dims = (/3,2,2/)
      else if ( nump .eq. 13 ) then
        dims = (/13,1,1/)
      else if ( nump .eq. 14 ) then
        dims = (/7,2,1/)
      else if ( nump .eq. 15 ) then
        dims = (/5,3,1/)
      else if ( nump .eq. 16 ) then
        dims = (/4,4,1/)
      else if ( nump .eq. 17 ) then
        dims = (/17,1,1/)
      else if ( nump .eq. 18 ) then
        dims = (/3,3,2/)
      else if ( nump .eq. 19 ) then
        dims = (/19,1,1/)
      else if ( nump .eq. 20 ) then
        dims = (/5,2,2/)
      else if ( nump .eq. 21 ) then
        dims = (/7,3,1/)
      else if ( nump .eq. 22 ) then
        dims = (/11,2,1/)
      else if ( nump .eq. 23 ) then
        dims = (/23,1,1/)
      else if ( nump .eq. 24 ) then
        dims = (/4,3,2/)
      else if ( nump .eq. 25 ) then
        dims = (/5,5,1/)
      else if ( nump .eq. 26 ) then
        dims = (/13,2,1/)
      else if ( nump .eq. 27 ) then
        dims = (/3,3,3/)
      else if ( nump .eq. 28 ) then
        dims = (/7,2,2/)
      else if ( nump .eq. 29 ) then
        dims = (/29,1,1/)
      else if ( nump .eq. 30 ) then
        dims = (/5,3,2/)
      else if ( nump .eq. 31 ) then
        dims = (/31,1,1/)
      else if ( nump .eq. 32 ) then
        dims = (/4,4,2/)
      else if ( nump .eq. 33 ) then
        dims = (/11,3,1/)
      else if ( nump .eq. 34 ) then
        dims = (/17,2,1/)
      else if ( nump .eq. 35 ) then
        dims = (/7,5,1/)
      else if ( nump .eq. 36 ) then
        dims = (/4,3,3/)
      else if ( nump .eq. 37 ) then
        dims = (/37,1,1/)
      else if ( nump .eq. 38 ) then
        dims = (/39,2,1/)
      else if ( nump .eq. 39 ) then
        dims = (/13,3,1/)
      else if ( nump .eq. 40 ) then
        dims = (/5,4,2/)
      end if

      return
      end    ! END: my_dims_create
