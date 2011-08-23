c  Copyright (c) 2011 Steve Liebling
c  Copyright (c) 2011 Matt Anderson
c
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc                                                                            cc
cc  level_cluster:                                                            cc
cc                                                                            cc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine level_cluster(flag,sigi,sigj,sigk, lapi,lapj,lapk,
     *                         asigi,asigj,asigk,alapi,alapj,alapk,
     *                         time, b_minx,b_maxx, b_miny,b_maxy,
     *                         b_minz,b_maxz,
     *                         minx,maxx,
     *                         miny,maxy,
     *                         minz,maxz,
     *                         numbox, nx,ny,nz,
     *                         clusterstyle,minefficiency,mindim,
     *                         ghostwidth,refine_factor,
     *                         minx0,miny0,minz0,maxx0,maxy0,maxz0)
      implicit     none
      integer clusterstyle,mindim,ghostwidth,refine_factor
      real(kind=8) minefficiency
      integer      numbox, nx,ny,nz
      integer      b_minx(*), b_maxx(*),
     *             b_miny(*), b_maxy(*),
     *             b_minz(*), b_maxz(*)
      real(kind=8) flag(nx,ny,nz), sigi(nx),  sigj(ny),  sigk(nz),
     *                             lapi(nx),  lapj(ny),  lapk(nz),
     *                             asigi(nx), asigj(ny), asigk(nz),
     *                             alapi(nx), alapj(ny), alapk(nz),
     *                             minx, maxx, miny,maxy, minz,maxz,
     *                             time,
     *                             minx0,miny0,minz0,maxx0,maxy0,maxz0

      !
      ! Flagging constants:
      !
      !     ...points     to be refined
      real(kind=8)   FLAG_REFINE
      parameter    ( FLAG_REFINE   =  1.d0 )
      !     ...points NOT to be refined
      real(kind=8)   FLAG_NOREFINE
      parameter    ( FLAG_NOREFINE =  0.d0 )
      !     ...points which do not exist on the level
      real(kind=8)   FLAG_DISALLOW
      parameter    ( FLAG_DISALLOW = -1.d0 )
      !
      integer     maxnumchildren ! total # of children for any grid
      parameter ( maxnumchildren = 16384       )
      !
      !  sigi_min  ---- minimum value of Signature X within the window
      !  sigi_min_i---- index of "i" at which Signature X reaches minimum
      !  sigi_min_s---- span...
      !
      ! Could conserve memory here by using some temporary array for 
      ! the sig's and lap's (these are the so-called Laplacian's from
      ! the B&R article on their clustering).
      !
      real(kind=8)      sigi_min,       sigj_min,       sigk_min,
     *            lapi_max,       lapj_max,       lapk_max,
     *            alapi_max,      alapj_max,      alapk_max,
     *            lapmax,         efficiency,
     *            new1efficiency, new2efficiency, netefficiency
      real(kind=8)      tmpdp1, tmpdp2
      integer     i,              j,              k,   ii, l,
     *            sigi_min_i,     sigj_min_j,     sigk_min_k,
     *            sigi_min_s,     sigj_min_s,     sigk_min_s,
     *            lapi_max_i,     lapj_max_j,     lapk_max_k,
     *            alapi_max_i,    alapj_max_j,    alapk_max_k,
     *            lengthx,        lengthy,        lengthz,
     *            maxlength,      minlength,      numdisallowed,
     *            xcut_quality,   ycut_quality,   zcut_quality,
     *            tmpi
      integer     tmp1, tmp2, tmp3, tmp4, tmp5, tmp6
      logical     backcutout
      logical     double_equal
      external    double_equal
      real(kind=8)getabsmax
      external    getabsmax
      !
      ! To store the original bbox before a cut:
      !
      integer      o_minx, o_maxx,
     *             o_miny, o_maxy,
     *             o_minz, o_maxz
       integer    cuttype,        cutaxis,        cutpos, already

       !
       ! Constants for code to decide on type of cut:
       !
       integer     NOCUT,         LAPCUT,       ZEROCUT,     DISCUT
       parameter ( NOCUT  = 0,    LAPCUT  = 1,  ZEROCUT = 2, DISCUT = 3)
       integer     XAXIS,         YAXIS,        ZAXIS
       parameter ( XAXIS  = 0,    YAXIS   = 1,  ZAXIS   = 2)
       ! Constant to remember which axis we have tried to cut:
       integer     NILL
       parameter ( NILL= -1)

      !
      ! Threshold by which an inflection cut must
      ! increase the net efficiency (between 0 and 1)
      ! or else be backed out
      !
      real(kind=8) ETHRESH
      !parameter  ( ETHRESH = 0.1d0 )
      !parameter  ( ETHRESH = 0.008d0 )
      parameter  ( ETHRESH = 0.02d0 )
      !
      ! Use cuts based on inflection points  as well as zero signatures ala B&R '91:
      !
      logical     inflection
      !
      ! If a box is found to contain any disallowed points (points
      ! not "covered" by its parent level), then we need to cut things up:
      !
      logical     anydisallowed

      logical     ltrace
      parameter ( ltrace  = .false. )
      logical     ltrace2
      parameter ( ltrace2 = .false. )

      !
      ! Let user restrict clustering if needed:
      !
      if (clusterstyle .eq. 1) then
         if (ltrace) write(*,*) 'level_cluster: Turning off inflection'
         inflection = .false.
      else
         inflection = .true.
      end if

      !
      ! initialize first bounding box:
      !
      numbox    = 1
      i         = 1
      b_minx(i) = 1
      b_miny(i) = 1
      b_minz(i) = 1
      b_maxx(i) = nx
      b_maxy(i) = ny
      b_maxz(i) = nz
      ALREADY   = NILL

      if (ltrace) then
         write(*,*) '********************************'
         write(*,*) 'level_cluster: Clustering at time: ', time
         write(*,*) '                        nx,ny,nz:', nx,ny,nz
         write(*,*) '                      inflection:', inflection
         write(*,*) '                          mindim:', mindim
         write(*,*) '                   minefficiency:', minefficiency
         write(*,*) '                         ETHRESH:', ETHRESH
      end if

      !
      ! compute bounding box, and signatures:
      !
 15   continue
      if (ltrace) write(*,*) '   Shrinkwrapping box i = ',i
      call find_bbox( flag,
     *                sigi,           sigj,           sigk,
     *                b_minx(i),      b_miny(i),      b_minz(i),
     *                b_maxx(i),      b_maxy(i),      b_maxz(i),
     *                nx,             ny,             nz,
     *                efficiency )
      if ( efficiency .eq. 0 ) then
         if (ltrace) write(*,*) 'level_cluster: no flagged pts'
         numbox = 0
         return
      else if(double_equal(minx,minx0).and.double_equal(maxx,maxx0).and.
     *        double_equal(miny,miny0).and.double_equal(maxy,maxy0).and.
     *        double_equal(minz,minz0).and.double_equal(maxz,maxz0).and.
     *        (b_minx(i).eq.1).and. (b_maxx(i).eq.nx)  .and.
     *        (b_miny(i).eq.1).and. (b_maxy(i).eq.ny)  .and.
     *        (b_minz(i).eq.1).and. (b_maxz(i).eq.nz)  ) then
            if (ltrace) write(*,*) 'level_cluster:     Complete cover'
            return
      end if
      !
      lengthx = b_maxx(i) - b_minx(i) + 1
      lengthy = b_maxy(i) - b_miny(i) + 1
      lengthz = b_maxz(i) - b_minz(i) + 1
      !
      ! compute asig (signature of the disallowed points, if any)
      !
      call compute_disallowed( flag,
     *                asigi,          asigj,          asigk,
     *                alapi,          alapj,          alapk,
     *                b_minx(i),      b_miny(i),      b_minz(i),
     *                b_maxx(i),      b_maxy(i),      b_maxz(i),
     *                nx,             ny,             nz,
     *                anydisallowed )
      if (anydisallowed) then
         if (ltrace) write(*,*) 'level_cluster: Disallowed points ',i
       call find_inflect(alapi(b_minx(i)),alapi_max,alapi_max_i,lengthx)
       call find_inflect(alapj(b_miny(i)),alapj_max,alapj_max_j,lengthy)
       call find_inflect(alapk(b_minz(i)),alapk_max,alapk_max_k,lengthz)
      end if

      !
      ! Avoid trying to cut where it would produce small grids:
      !     (0.1d0 is just an arbitrary, nonzero number)
      !
      call load_scal1d(sigi(b_minx(i)),0.1d0, mindim)
      call load_scal1d(sigj(b_miny(i)),0.1d0, mindim)
      call load_scal1d(sigk(b_minz(i)),0.1d0, mindim)
      call load_scal1d(sigi(b_minx(i)+lengthx-mindim),0.1d0, mindim)
      call load_scal1d(sigj(b_miny(i)+lengthy-mindim),0.1d0, mindim)
      call load_scal1d(sigk(b_minz(i)+lengthz-mindim),0.1d0, mindim)

      !
      ! Compute signatures and find minimums:
      !
      call find_min1d( sigi(b_minx(i)),
     *                 sigi_min, sigi_min_i, sigi_min_s, lengthx)
      call find_min1d( sigj(b_miny(i)),
     *                 sigj_min, sigj_min_j, sigj_min_s, lengthy)
      call find_min1d( sigk(b_minz(i)),
     *                 sigk_min, sigk_min_k, sigk_min_s, lengthz)

      !
      ! Compute laplacians (as per B&R algorithm):
      !
      call  compute_lap(lapi(b_minx(i)), sigi(b_minx(i)), lengthx)
      call  compute_lap(lapj(b_miny(i)), sigj(b_miny(i)), lengthy)
      call  compute_lap(lapk(b_minz(i)), sigk(b_minz(i)), lengthz)

      ! Zero out regions near beginning and ends of this box
      ! since we do not want to split there else we would get
      ! too small clusters:
      !    (needs to go one more point than with the signatures)
      !
      call load_scal1d(lapi(b_minx(i)),0.d0, mindim+1)
      call load_scal1d(lapj(b_miny(i)),0.d0, mindim+1)
      call load_scal1d(lapk(b_minz(i)),0.d0, mindim+1)
      call load_scal1d(lapi(b_minx(i)+lengthx-mindim-1),0.d0, mindim+1)
      call load_scal1d(lapj(b_miny(i)+lengthy-mindim-1),0.d0, mindim+1)
      call load_scal1d(lapk(b_minz(i)+lengthz-mindim-1),0.d0, mindim+1)

      !
      ! Look for maximum inflection points in laplacians:
      !
      call find_inflect(lapi(b_minx(i)), lapi_max, lapi_max_i, lengthx)
      call find_inflect(lapj(b_miny(i)), lapj_max, lapj_max_j, lengthy)
      call find_inflect(lapk(b_minz(i)), lapk_max, lapk_max_k, lengthz)

      if ( ltrace ) then
          write(*,*) '   ************'
          write(*,*) '   Box     i = ', i
          write(*,*) '       b_minx(i), mxx(i) = ',b_minx(i),b_maxx(i)
          write(*,*) '       b_miny(i), mxy(i) = ',b_miny(i),b_maxy(i)
          write(*,*) '       b_minz(i), mxz(i) = ',b_minz(i),b_maxz(i)
          write(*,*) '   mindim: ',mindim
          write(*,*) '  lengths: ',lengthx,lengthy,lengthz
          write(*,96)'  sigi_min, sigj_min, sigk_min: ',
     *                               sigi_min,sigj_min,sigk_min
          write(*,*) '  efficiency: ',efficiency
          write(*,*) 'lapi_max, lapi_max_i = ',lapi_max, lapi_max_i
          write(*,*) 'lapj_max, lapj_max_j = ',lapj_max, lapj_max_j
          write(*,*) 'lapk_max, lapk_max_k = ',lapk_max, lapk_max_k
            if (ltrace2) then
              if (.not.anydisallowed) then
              write(*,*)' * * *   i:   sigi(i):                lapi(i):'
               do j = b_minx(i), b_minx(i)+lengthx-1
                  write(*,*) j, sigi(j),lapi(j)
               end do
              write(*,*)' * * *   j:   sigj(j):                lapj(j):'
               do j = b_miny(i), b_miny(i)+lengthy-1
                  write(*,*) j, sigj(j),lapj(j)
               end do
              write(*,*)' * * *   k:   sigk(k):                lapk(k):'
               do j = b_minz(i), b_minz(i)+lengthz-1
                  write(*,*) j, sigk(j),lapk(j)
               end do
              else
                write(*,*)' Cluster contains disallowed points:',i
                write(*,*)'alapi_max,alapi_max_i=',alapi_max,alapi_max_i
                write(*,*)'alapj_max,alapj_max_j=',alapj_max,alapj_max_j
                write(*,*)'alapk_max,alapk_max_k=',alapk_max,alapk_max_k
              write(*,*)' * * *   j:   asigi(j):              alapi(j):'
                  do j = b_minx(i), b_minx(i)+lengthx-1
                     write(*,*) j, asigi(j),alapi(j)
                  end do
              write(*,*)' * * *   j:   asigj(j):              alapj(j):'
                  do j = b_miny(i), b_miny(i)+lengthy-1
                     write(*,*) j, asigj(j),alapj(j)
                  end do
              write(*,*)' * * *   j:   asigk(j):              alapk(j):'
                  do j = b_minz(i), b_minz(i)+lengthz-1
                     write(*,*) j, asigk(j),alapk(j)
                  end do
              end if
            end if
      end if

      !
      ! Cut bbox into simply connected pieces:
      !
      if (efficiency .lt. minefficiency .or. anydisallowed) then
         if (ltrace)write(*,*)'  Efficiency too low:',efficiency
         !
         ! Decide best place to split (if possible to split):
         !   (1) If can do a split at a zero of sig, do the
         !       "best" zero split first
         !   (2) If no zero splits possible, look for best
         !       possible inflection point split
         !   (3) If no split possible, simply move on to next
         !       box, if there is one
         !
         cuttype = NOCUT
         if (ltrace) write(*,*) '   Looking for zero cuts first'
         !
         ! Find the best possible zero cut:
         !
         xcut_quality = min( sigi_min_i, lengthx - sigi_min_i)
         ycut_quality = min( sigj_min_j, lengthy - sigj_min_j)
         zcut_quality = min( sigk_min_k, lengthz - sigk_min_k)
         if (ltrace) then
            write(*,*) '   Looking for zero cuts:'
            write(*,*) '      xcut_quality = ',xcut_quality
            write(*,*) '      ycut_quality = ',ycut_quality
            write(*,*) '      zcut_quality = ',zcut_quality
         end if
         if ( sigi_min .eq. 0 .and. xcut_quality .ge. mindim) then
            !
            ! Zero split possible only if:
            !                disconnected flagged regions
            !                "left"  region would have minimum dimension
            !                "right" region would have minimum dimension
            !
            if (ltrace) write(*,*) '    X zero cut possible'
            cuttype = ZEROCUT
            cutaxis = XAXIS
            cutpos  = sigi_min_i
         end if
         if ( sigj_min .eq. 0 .and. ycut_quality .ge. mindim) then
            if (ltrace) write(*,*) '    Y zero cut possible'
            !
            ! Pick cut with highest quality
            !
            if (sigi_min.eq.0 .and. xcut_quality.gt.ycut_quality) then
               if (ltrace) write(*,*) '    Go with X cut instead'
            else
               cuttype = ZEROCUT
               cutaxis = YAXIS
               cutpos  = sigj_min_j
            endif
         end if
         if ( sigk_min .eq. 0 .and. zcut_quality .ge. mindim) then
            if (ltrace) write(*,*) '    Z zero cut possible'
            !
            ! Pick cut with highest quality
            !
            if ( (sigi_min.eq.0.and.xcut_quality.gt.zcut_quality) .or.
     *           (sigj_min.eq.0.and.ycut_quality.gt.zcut_quality) ) then
               if (ltrace) write(*,*) '    Go with other cut instead'
            else
               cuttype = ZEROCUT
               cutaxis = ZAXIS
               cutpos  = sigk_min_k
            end if
         end if
         !
         ! Cuts based on disallowed points:
         !
         if ( cuttype .eq. NOCUT .and. anydisallowed ) then
            xcut_quality = min(alapi_max_i, lengthx -alapi_max_i)
            ycut_quality = min(alapj_max_j, lengthy -alapj_max_j)
            zcut_quality = min(alapk_max_k, lengthz -alapk_max_k)
            if (ltrace) then
               write(*,*) '   Looking for DISallowed cuts:'
               write(*,*) '      xcut_quality = ',xcut_quality
               write(*,*) '      ycut_quality = ',ycut_quality
               write(*,*) '      zcut_quality = ',zcut_quality
               write(*,*) '      alapi_max_i  = ',alapi_max_i
               write(*,*) '      alapj_max_j  = ',alapj_max_j
               write(*,*) '      alapk_max_k  = ',alapk_max_k
               write(*,*) '      lengthx      = ',lengthx
               write(*,*) '      lengthy      = ',lengthy
               write(*,*) '      lengthz      = ',lengthz
            end if
            !
            lapmax = 0.d0
            if (alapk_max .gt. 0.d0) then
                if (ltrace) write(*,*) '    zcut possible'
                cuttype = DISCUT
                cutaxis = ZAXIS
                cutpos  = alapk_max_k
                lapmax  = alapk_max
            end if
            if (alapj_max .gt. lapmax) then
                if (ltrace) write(*,*) '    ycut preferable'
                cuttype = DISCUT
                cutaxis = YAXIS
                cutpos  = alapj_max_j
                lapmax  = alapj_max
            end if
            if (alapi_max .gt. lapmax ) then
                if (ltrace) write(*,*) '    xcut most preferred'
                cuttype = DISCUT
                cutaxis = XAXIS
                cutpos  = alapi_max_i
                lapmax  = alapi_max
            end if
         end if
         if ( cuttype .eq. NOCUT .and. inflection ) then
            !
            ! Decide based on:
            ! (1) New boxes must have minimum dimensions
            ! (2) lapX_max must be non zero, else not an inflection point
            ! (3) Pick axis with the maximum value of the inflection point
            !
            !
            xcut_quality = min( lapi_max_i, lengthx - lapi_max_i)
            ycut_quality = min( lapj_max_j, lengthy - lapj_max_j)
            zcut_quality = min( lapk_max_k, lengthz - lapk_max_k)
            if (ltrace) then
               write(*,*) '   Looking for inflection pts:'
               write(*,*) '      xcut_quality = ',xcut_quality
               write(*,*) '      ycut_quality = ',ycut_quality
               write(*,*) '      zcut_quality = ',zcut_quality
            end if
            !
  500       lapmax = 0.d0
            if ( lapk_max .gt. 0.d0
     *          .and. ALREADY .ne. ZAXIS
     *          .and. zcut_quality .gt. mindim       ) then
                if (ltrace) write(*,*) '    zcut possible'
                cuttype = LAPCUT
                cutaxis = ZAXIS
                cutpos  = lapk_max_k
                lapmax  = lapk_max
            end if
            if ( lapj_max .gt. lapmax
     *          .and. ALREADY .ne. YAXIS
     *                    .and. ycut_quality .gt. mindim       ) then
                if (ltrace) write(*,*) '    ycut preferable'
                cuttype = LAPCUT
                cutaxis = YAXIS
                cutpos  = lapj_max_j
                lapmax  = lapj_max
            end if
            if ( lapi_max .gt. lapmax
     *          .and. ALREADY .ne. XAXIS
     *                    .and. xcut_quality .gt. mindim       ) then
                if (ltrace) write(*,*) '    xcut most preferred'
                cuttype = LAPCUT
                cutaxis = XAXIS
                cutpos  = lapi_max_i
                lapmax  = lapi_max
            end if
         end if
         !
         ! Make the cut:
         !
         if (cuttype .ne. NOCUT) then
            if (ltrace) then
               write(*,*) ' A cut has been chosen for box:',i
               write(*,*) '      cuttype = ',cuttype
               write(*,*) '      cutaxis = ',cutaxis
               write(*,*) '      cutpos  = ',cutpos
            end if
            !
            ! Store original box:
            !
            o_minx         = b_minx(i)
            o_miny         = b_miny(i)
            o_minz         = b_minz(i)
            o_maxx         = b_maxx(i)
            o_maxy         = b_maxy(i)
            o_maxz         = b_maxz(i)
            !
            ! Add new box, and copy bounding box:
            !
            numbox         = numbox + 1
            b_minx(numbox) = b_minx(i)
            b_miny(numbox) = b_miny(i)
            b_minz(numbox) = b_minz(i)
            b_maxx(numbox) = b_maxx(i)
            b_maxy(numbox) = b_maxy(i)
            b_maxz(numbox) = b_maxz(i)
            if      (cutaxis .eq. XAXIS) then
               b_maxx(i)      = b_minx(i) + cutpos - 1
               b_minx(numbox) = b_maxx(i) + 1
            else if (cutaxis .eq. YAXIS) then
               b_maxy(i)      = b_miny(i) + cutpos - 1
               b_miny(numbox) = b_maxy(i) + 1
            else if (cutaxis .eq. ZAXIS) then
               b_maxz(i)      = b_minz(i) + cutpos - 1
               b_minz(numbox) = b_maxz(i) + 1
            end if
            call find_bbox( flag,
     *                sigi,           sigj,           sigk,
     *                b_minx(i),      b_miny(i),      b_minz(i),
     *                b_maxx(i),      b_maxy(i),      b_maxz(i),
     *                nx,             ny,             nz,
     *                new1efficiency )
            call find_bbox( flag,
     *                sigi,           sigj,           sigk,
     *                b_minx(numbox), b_miny(numbox), b_minz(numbox),
     *                b_maxx(numbox), b_maxy(numbox), b_maxz(numbox),
     *                nx,             ny,             nz,
     *                new2efficiency )
            netefficiency = (
     *                (new1efficiency*(b_maxx(i)     -b_minx(i)+1)
     *                               *(b_maxy(i)     -b_miny(i)+1)
     *                               *(b_maxz(i)     -b_minz(i)+1))
     *              + (new2efficiency*(b_maxx(numbox)-b_minx(numbox)+1)
     *                               *(b_maxy(numbox)-b_miny(numbox)+1)
     *                               *(b_maxz(numbox)-b_minz(numbox)+1))
     *                       )/(
     *                                (b_maxx(i)     -b_minx(i)+1)
     *                               *(b_maxy(i)     -b_miny(i)+1)
     *                               *(b_maxz(i)     -b_minz(i)+1)
     *              +                 (b_maxx(numbox)-b_minx(numbox)+1)
     *                               *(b_maxy(numbox)-b_miny(numbox)+1)
     *                               *(b_maxz(numbox)-b_minz(numbox)+1))
            if (ltrace) then
              write(*,96)
     *         'old,new1,new2:',efficiency,new1efficiency,new2efficiency
             write(*,*)'netefficiency=',netefficiency
             write(*,*)'After cut: Box     i = ', i
             write(*,*)'  b_min/maxx(i)= ',b_minx(i),b_maxx(i),
     *                                     b_maxx(i)-b_minx(i)+1
             write(*,*)'  b_min/maxy(i)= ',b_miny(i),b_maxy(i),
     *                                     b_maxy(i)-b_miny(i)+1
             write(*,*)'  b_min/maxz(i)= ',b_minz(i),b_maxz(i),
     *                                     b_maxz(i)-b_minz(i)+1
             write(*,*)'After cut: Box     i = ', numbox
            write(*,*)'  b_min/maxx(i)= ',b_minx(numbox),b_maxx(numbox),
     *                                   b_maxx(numbox)-b_minx(numbox)+1
            write(*,*)'  b_min/maxy(i)= ',b_miny(numbox),b_maxy(numbox),
     *                                   b_maxy(numbox)-b_miny(numbox)+1
            write(*,*)'  b_min/maxz(i)= ',b_minz(numbox),b_maxz(numbox),
     *                                   b_maxz(numbox)-b_minz(numbox)+1
            end if
 96         format(A,3F15.5)
            !
            ! Back the cut out?
            !
            if (cuttype .eq. LAPCUT) then
               if (      (b_maxx(i)-b_minx(i) .le.   mindim) .or.
     *                   (b_maxy(i)-b_miny(i) .le.   mindim) .or.
     *                   (b_maxz(i)-b_minz(i) .le.   mindim) ) then
                  !
                  ! If new box is too small....
                  !
                  if (ltrace) write(*,*) ' Too small box i=',i
                  backcutout = .true.
               else if ( 
     *             (b_maxx(numbox)-b_minx(numbox) .le.  mindim).or.
     *             (b_maxy(numbox)-b_miny(numbox) .le.  mindim).or.
     *             (b_maxz(numbox)-b_minz(numbox) .le.  mindim))then
                  !
                  ! If new box is too small....
                  !
                  if (ltrace) write(*,*) ' Too small box i=',numbox
                  backcutout = .true.
               else if ( (netefficiency-efficiency).le.ETHRESH) then
                  !
                  ! If efficiency did not increase above threshold...
                  !
                  if (ltrace) write(*,*) ' No more efficient'
                  backcutout = .true.
               else
                  backcutout = .false.
               end if
            else
               backcutout = .false.
            end if

            if ( backcutout ) then
               !
               ! Restore original box:
               !
               if (ltrace) write(*,*) ' Backing cut out!'
               b_minx(i) = o_minx
               b_miny(i) = o_miny
               b_minz(i) = o_minz
               b_maxx(i) = o_maxx
               b_maxy(i) = o_maxy
               b_maxz(i) = o_maxz
               ! Remove added box:
               numbox    = numbox - 1
               !
               ! Try another cut (if we have not already):
               !
               if (cuttype .eq. LAPCUT .and. ALREADY.eq.NILL) then
                  !
                  ! Remember which axis we tried already:
                  !
                  if (ltrace) write(*,*) ' Try a different axis'
                  ALREADY = cutaxis
                  cuttype = NOCUT
                  goto 500
               else
                  ! Move on to consider the next box:
                  if (ltrace) write(*,*) ' Moving on to next box'
                  i         = i + 1
                  ALREADY   = NILL
               end if
            else
               if (ltrace) write(*,*) ' Cut is good'
               ALREADY = NILL
            end if
         else
            if (ltrace) write(*,*) ' No cut possible'
            i = i + 1
         end if
      else
         !
         ! Box of sufficient efficiency, look at next one.
         !
         if (ltrace)write(*,*)'Box has sufficient efficiency',efficiency
         i = i + 1
      end if
      !
      if (ltrace) write(*,*) 'Moving on: i, numbox = ',i,numbox
      if (numbox .eq. maxnumchildren) then
         write(*,*) ' We have reached maxnumchildren, no more boxes'
      else if (i.le.numbox) then 
         if (ltrace) write(*,*) ' '
         goto 15         ! more boxes to process
      end if

      !
      ! Clustering is now done, just need to do some extra stuff
      !
      if (ltrace .and. .true.) then
      !if (ltrace .and. .false.) then
          !
          ! For debugging purposes we want to output
          ! the flag array such that we can examine
          ! the clusters it produced:
          !    (1) Recompute signatures
          !    (2) For every flagged point in each
          !        box, replace with the grid number
          !    (3) Along the edges, place the signatures
          !
          tmp1 = 1
          tmp2 = 1
          tmp3 = 1
          tmp4 = nx
          tmp5 = ny
          tmp6 = nz
          call find_bbox( flag,
     *                sigi,           sigj,           sigk,
     *                tmp1, tmp2, tmp3, tmp4, tmp5, tmp6,
     *                nx,             ny,             nz,
     *                efficiency )
         call  compute_lap(lapi, sigi, nx)
         call  compute_lap(lapj, sigj, ny)
         call  compute_lap(lapk, sigk, nz)
          write(*,*) 'level_cluster: Outputting flag array'
          !
          ! (2) Show into which cluster flagged points fall
          !
          do l = 1, numbox
             write(*,*) 'level_cluster:   Marking box l=',l
             do k = b_minz(l), b_maxz(l)
             do j = b_miny(l), b_maxy(l)
             do i = b_minx(l), b_maxx(l)
                if (flag(i,j,k) .eq. FLAG_REFINE) then
                   flag(i,j,k) = 1.d0*l
                end if
             end do
             end do
             end do
          end do
          !
          ! (3) Add normalized signature information:
          !      ( This can cause problems when clusters
          !        are checked for negative values. )
          !
          if (.false.) then
          tmpdp1 = getabsmax(sigi,nx)/numbox
          tmpdp2 = getabsmax(lapi,nx)/numbox
          do k = 1, nz
          do i = 1, nx
             flag(i,1, k) = sigi(i)/tmpdp1
             flag(i,ny,k) = lapi(i)/tmpdp2
          end do
          end do
          tmpdp1 = getabsmax(sigj,ny)/numbox
          tmpdp2 = getabsmax(lapj,ny)/numbox
          do k = 1, nz
          do j = 1, ny
             flag(1, j,k) = sigj(j)/tmpdp1
             flag(nx,j,k) = lapj(j)/tmpdp2
          end do
          end do
          end if
      end if

      !
      ! Add ghostwidth to clusters:
      !    (ghostwidth is defined on the fine grid, 
      !     so need to figure out how many coarse grid points)
      !
      tmpi = NINT( (1.d0*ghostwidth) / refine_factor )
      if (ltrace) write(*,*) '   Adding tmpi points: ',tmpi
      do i = 1, numbox
         if (ltrace) write(*,*) '   Growing box i=',i
         call grow_bbox(flag, FLAG_DISALLOW, tmpi,
     *               b_minx(i),b_maxx(i),
     *               b_miny(i),b_maxy(i),
     *               b_minz(i),b_maxz(i),
     *               nx, ny, nz )
      end do


      !
      ! Done making boxes, now check them:
      !
      if (ltrace) write(*,*) '   Check bounds of boxes'
      do i = 1, numbox
         !
         if (ltrace) then
            write(*,*) '   Box     i = ', i
            write(*,*) '      b_minx(i), mxx(i) = ',b_minx(i),b_maxx(i),
     *                                             b_maxx(i)-b_minx(i)+1
            write(*,*) '      b_miny(i), mxy(i) = ',b_miny(i),b_maxy(i),
     *                                             b_maxy(i)-b_miny(i)+1
            write(*,*) '      b_minz(i), mxz(i) = ',b_minz(i),b_maxz(i),
     *                                             b_maxz(i)-b_minz(i)+1
         end if


         if ( ( (b_maxx(i)-b_minx(i)+1) .lt. mindim ) .or.
     *        ( (b_maxy(i)-b_miny(i)+1) .lt. mindim ) .or.
     *        ( (b_maxz(i)-b_minz(i)+1) .lt. mindim ) ) then
            write(*,*) 'level_cluster: Small box created'
            write(*,*) 'level_cluster: Presuming cut was correct,'
            write(*,*) 'level_cluster: but box shrank afterward.'
            write(*,*) '   nx,ny,nz  = ',nx,ny,nz
            write(*,*) '   mindim    = ',mindim
            write(*,*) '   numbox    = ',numbox
            write(*,*) '   Box     i = ', i
            write(*,*) '       b_minx(i), mxx(i) = ',b_minx(i),b_maxx(i)
            write(*,*) '       b_miny(i), mxy(i) = ',b_miny(i),b_maxy(i)
            write(*,*) '       b_minz(i), mxz(i) = ',b_minz(i),b_maxz(i)
            if (numbox .eq. 1) then
               write(*,*) 'level_cluster: Simply removing bad box:'
               write(*,*) 'level_cluster: Continuing...'
               numbox = 0
               return
            else
               write(*,*) 'level_cluster: More than one box=',numbox
               write(*,*) 'level_cluster: So cannot remove'
               write(*,*) '**** There very likely is a problem'
               write(*,*) '**** with the clusters produced, but'
               write(*,*) '**** I will not quit so maybe you '
               write(*,*) '**** can see what is wrong'
               write(*,*) ''
            end if
            write(*,*)'level_cluster: Outputting flag array: flagDEBUG'
            write(*,*)'level_cluster: *******'
            write(*,*)'level_cluster: *******'
            write(*,*)'level_cluster: Please include this file',
     *                           ' for debugging'
            write(*,*)'level_cluster: *******'
            write(*,*)'level_cluster: *******'
         end if
         !
         ! Ensure sub grids aren't at interior grid boundaries:
         !
         if (  (b_minx(i).lt. 1) .or.  (b_maxx(i).gt. nx) .or.
     *         (b_miny(i).lt. 1) .or.  (b_maxy(i).gt. ny) .or.
     *         (b_minz(i).lt. 1) .or.  (b_maxz(i).gt. nz) ) then
            write(*,*) 'level_cluster: Problem:'
            write(*,*) '   nx,ny,nz  = ',nx,ny,nz
            write(*,*) '   mindim    = ',mindim
            write(*,*) '   Box     i = ', i
            write(*,*) '       b_minx(i), mxx(i) = ',b_minx(i),b_maxx(i)
            write(*,*) '       b_miny(i), mxy(i) = ',b_miny(i),b_maxy(i)
            write(*,*) '       b_minz(i), mxz(i) = ',b_minz(i),b_maxz(i)
            write(*,*) 'level_cluster: Bbox extends to/past parent bdy'
            write(*,*)'level_cluster: Outputting flag array: flagDEBUG'
            write(*,*)'level_cluster: *******'
            write(*,*)'level_cluster: *******'
            write(*,*)'level_cluster: Please include this file',
     *                           ' for debugging'
            write(*,*)'level_cluster: *******'
            write(*,*)'level_cluster: *******'
         end if
         !
         ! Ensure bboxes are properly formed:
         !
         if ( (b_minx(i) .ge. b_maxx(i) ) .or.
     *        (b_miny(i) .ge. b_maxy(i) ) .or.
     *        (b_minz(i) .ge. b_maxz(i) )      ) then
            if (numbox .eq. 1) then
               numbox = 0
               if (ltrace) write(*,*) 'level_cluster: '
            end if
            write(*,*) 'level_cluster: Box is malformed'
            write(*,*) 'level_cluster: Box i: ', i
            write(*,*) '       b_minx(i), mxx(i) = ',b_minx(i),b_maxx(i)
            write(*,*) '       b_miny(i), mxy(i) = ',b_miny(i),b_maxy(i)
            write(*,*) '       b_minz(i), mxz(i) = ',b_minz(i),b_maxz(i)
            write(*,*) 'level_cluster: Quitting...'
            write(*,*)'level_cluster: Outputting flag array: flagDEBUG'
            write(*,*)'level_cluster: *******'
            write(*,*)'level_cluster: *******'
            write(*,*)'level_cluster: Please include this file',
     *                           ' for debugging'
            write(*,*)'level_cluster: *******'
            write(*,*)'level_cluster: *******'
         end if
         !
         ! Check that no disallowed points (points that would NOT be
         ! fully nested in the parent level) are included:
         !
         numdisallowed = 0
         do k  = b_minz(i), b_maxz(i)
         do j  = b_miny(i), b_maxy(i)
         do ii = b_minx(i), b_maxx(i)
            if (double_equal(flag(ii,j,k),FLAG_DISALLOW)) then
               numdisallowed = numdisallowed + 1
               write(*,*) 'level_cluster: at point ii,j,k:',ii,j,k
            end if
         end do
         end do
         end do
         if (numdisallowed.gt.0) then
            write(*,*) 'level_cluster: Disallowed point found'
            write(*,*) 'level_cluster: Cluster: ',i
            write(*,*) '       b_minx(i), mxx(i) = ',b_minx(i),b_maxx(i)
            write(*,*) '       b_miny(i), mxy(i) = ',b_miny(i),b_maxy(i)
            write(*,*) '       b_minz(i), mxz(i) = ',b_minz(i),b_maxz(i)
            write(*,*) '   nx,ny,nz  = ',nx,ny,nz
            write(*,*) '   numbox    = ',numbox
            write(*,*) 'level_cluster: numdisallowed: ',numdisallowed
            write(*,*) 'level_cluster: outputting flagdis'
             !do l = numbox, 1, -1
             do l = 1, numbox
             write(*,*) 'level_cluster:   Marking box l=',l
            write(*,*) '       b_minx(i), mxx(i) = ',b_minx(l),b_maxx(l)
            write(*,*) '       b_miny(i), mxy(i) = ',b_miny(l),b_maxy(l)
            write(*,*) '       b_minz(i), mxz(i) = ',b_minz(l),b_maxz(l)
             do k  = b_minz(l), b_maxz(l)
             do j  = b_miny(l), b_maxy(l)
             do ii = b_minx(l), b_maxx(l)
                if (flag(ii,j,k) .eq. FLAG_REFINE) then
                   flag(ii,j,k) = 1.d0*l
                end if
             end do
             end do
             end do
            end do
            write(*,*)'level_cluster: Outputting flag array: flagdis'
            write(*,*)'level_cluster: *******'
            write(*,*)'level_cluster: *******'
            write(*,*)'level_cluster: Please include this file',
     *                           ' for debugging'
            write(*,*)'level_cluster: *******'
            write(*,*)'level_cluster: *******'
c        call level_output_flag(   flag, 0.d0,
c    *                    b_minx,b_maxx,
c    *                    b_miny,b_maxy,
c    *                    b_minz,b_maxz,
c    *                    'flag',nx,ny,nz,numbox)
         end if
      end do

 800  if (ltrace) then
         write(*,*) 'level_cluster: DONE Clustering:'
         do i = 1, numbox
           write(*,*) '   Box     i = ', i
           write(*,*) '       b_minx(i), mxx(i) = ',b_minx(i),b_maxx(i),
     *                                             b_maxx(i)-b_minx(i)+1
           write(*,*) '       b_miny(i), mxy(i) = ',b_miny(i),b_maxy(i),
     *                                             b_maxy(i)-b_miny(i)+1
           write(*,*) '       b_minz(i), mxz(i) = ',b_minz(i),b_maxz(i),
     *                                             b_maxz(i)-b_minz(i)+1
         end do
         write(*,*) '********************************'
      end if

      !write(*,*) 'h'
      return
      end      ! END: level_cluster
