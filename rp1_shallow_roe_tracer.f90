! =========================================================
subroutine rp1(maxmx,num_eqn,num_waves,num_aux,num_ghost,num_cells, &
    ql,qr,auxl,auxr,wave,s,amdq,apdq)
! =========================================================

! Solve Riemann problems for the 1D shallow water equations
! with an additional passively advected tracer:
!    (h)_t + (u h)_x = 0
!    (uh)_t + ( uuh + .5*gh^2 )_x = 0
!    c_t + uc_x = 0
! using Roe's approximate Riemann solver with entropy fix for
! transonic rarefractions.

! waves: 3
! equations: 3

! Conserved quantities:
!       1 depth
!       2 momentum
!       3 tracer

! This function solves the Riemann problem at all interfaces in one call

! On input, ql contains the state vector at the left edge of each cell
!           qr contains the state vector at the right edge of each cell
! On output, wave contains the waves,
!            s the speeds,
!            amdq the  left-going flux difference  A^- \Delta q
!            apdq the right-going flux difference  A^+ \Delta q

! Note that the i'th Riemann problem has left state qr(:,i-1)
!                                    and right state ql(:,i)
! From the basic clawpack routine step1, rp is called with ql = qr = q.


    implicit none

    integer, intent(in) :: maxmx, num_eqn, num_waves, num_aux, num_ghost, &
                            num_cells
    real(kind=8), intent(in), dimension(num_eqn,1-num_ghost:maxmx+num_ghost) :: ql, qr
    real(kind=8), intent(in), dimension(num_aux,1-num_ghost:maxmx+num_ghost) :: auxl, auxr
    real(kind=8), intent(out) :: s(num_waves, 1-num_ghost:maxmx+num_ghost)
    real(kind=8), intent(out) :: wave(num_eqn, num_waves, 1-num_ghost:maxmx+num_ghost)
    real(kind=8), intent(out), dimension(num_eqn,1-num_ghost:maxmx+num_ghost) :: amdq,apdq

    ! local variables:
    real(kind=8) :: a1,a2,ubar,cbar,s0,s1,s2,s3,hr1,uhr1,hl2,uhl2,sfract,df
    real(kind=8) :: delta(2)
    integer :: i,m,mw


    logical :: efix

    data efix /.false./    !# Use entropy fix for transonic rarefactions

    ! Gravity constant set in setprob.f or the shallow1D.py file
    real(kind=8) :: grav
    common /cparam/ grav

    ! Main loop of the Riemann solver.
    do 30 i=2-num_ghost,num_cells+num_ghost
    
    
        ! compute  Roe-averaged quantities:
        ubar = (qr(2,i-1)/dsqrt(qr(1,i-1)) + ql(2,i)/dsqrt(ql(1,i)))/ &
               ( dsqrt(qr(1,i-1)) + dsqrt(ql(1,i)) )
        cbar=dsqrt(0.5d0*grav*(qr(1,i-1) + ql(1,i)))
                 
        ! delta(1)=h(i)-h(i-1) and  delta(2)=hu(i)-hu(i-1)
        delta(1) = ql(1,i) - qr(1,i-1)
        delta(2) = ql(2,i) - qr(2,i-1)

        ! Compute coeffs in the evector expansion of delta(1),delta(2)
        a1 = 0.5d0*(-delta(2) + (ubar + cbar) * delta(1))/cbar
        a2 = 0.5d0*( delta(2) - (ubar - cbar) * delta(1))/cbar

        ! Finally, compute the waves.
        wave(1,1,i) = a1
        wave(2,1,i) = a1*(ubar - cbar)
        wave(3,1,i) = 0.d0
        s(1,i) = ubar - cbar
                 
        wave(1,2,i) = a2
        wave(2,2,i) = a2*(ubar + cbar)
        wave(3,2,i) = 0.d0
        s(2,i) = ubar + cbar

        wave(1,3,i) = 0.d0
        wave(2,3,i) = 0.d0
        wave(3,3,i) = ql(3,i) - qr(3,i-1)
        s(3,i) = ubar
                 
    30 enddo

    ! Compute fluctuations amdq and apdq
    ! ------------------------------------

    if (efix) go to 110

    ! No entropy fix
    ! ----------------------------------------------
    ! amdq = SUM s*wave   over left-going waves
    ! apdq = SUM s*wave   over right-going waves

    do m=1,num_waves
        do i=2-num_ghost, num_cells+num_ghost
            amdq(m,i) = 0.d0
            apdq(m,i) = 0.d0
            do mw=1,num_waves
                if (s(mw,i) < 0.d0) then
                    amdq(m,i) = amdq(m,i) + s(mw,i)*wave(m,mw,i)
                else
                    apdq(m,i) = apdq(m,i) + s(mw,i)*wave(m,mw,i)
                endif
            enddo
        enddo
    enddo
    
    ! with no entropy fix we are done...
    return


    ! -----------------------------------------------

    110 continue

    ! With entropy fix
    ! ------------------

    ! compute flux differences amdq and apdq.
    ! First compute amdq as sum of s*wave for left going waves.
    ! Incorporate entropy fix by adding a modified fraction of wave
    ! if s should change sign.

    do 200 i=2-num_ghost,num_cells+num_ghost
              
        ! ------------------------------------------------------
        ! check 1-wave:
        ! ---------------
    
        ! u-c in left state (cell i-1)
        s0 = qr(2,i-1)/qr(1,i-1) - dsqrt(grav*qr(1,i-1))
                 
        ! check for fully supersonic case:
        if (s0 >= 0.d0 .and. s(1,i) > 0.d0)  then
            ! everything is right-going
            do m=1,num_eqn
                amdq(m,i) = 0.d0
                enddo
            go to 200
        endif
    
        ! u-c to right of 1-wave
        hr1  = qr(1,i-1) + wave(1,1,i)
        uhr1 = qr(2,i-1) + wave(2,1,i)
        s1 =  uhr1/hr1 - dsqrt(grav*hr1)
                         
        if (s0 < 0.d0 .and. s1 > 0.d0) then
            ! transonic rarefaction in the 1-wave
            sfract = s0 * (s1-s(1,i)) / (s1-s0)
        else if (s(1,i) < 0.d0) then
            ! 1-wave is leftgoing
            sfract = s(1,i)
        else
            ! 1-wave is rightgoing
            sfract = 0.d0   !# this shouldn't happen since s0 < 0
        endif

        do m=1,num_eqn
            amdq(m,i) = sfract*wave(m,1,i)
            enddo
          
        ! -------------------------------------------------------
        ! check 2-wave (second nonlinear wave, note tracer contact is 3-wave):
        ! ---------------
        ! u+c in right state  (cell i)
        s3 = ql(2,i)/ql(1,i) + dsqrt(grav*ql(1,i))
                      
        ! u+c to left of 2-wave
        hl2  = ql(1,i) - wave(1,2,i)
        uhl2 = ql(2,i) - wave(2,2,i)
        s2 = uhl2/hl2 + dsqrt(grav*hl2)
                          
        if (s2 < 0.d0 .and. s3 > 0.d0) then
            ! transonic rarefaction in the 2-wave
            sfract = s2 * (s3-s(2,i)) / (s3-s2)
        else if (s(2,i) < 0.d0) then
            ! 2-wave is leftgoing
            sfract = s(2,i)
        else
            ! 2-wave is rightgoing
            go to 200
        endif
    
        do m=1,num_eqn
            amdq(m,i) = amdq(m,i) + sfract*wave(m,2,i)
            enddo

    200 enddo


    ! compute the rightgoing flux differences:
    ! df = SUM s*wave   is the total flux difference and apdq = df - amdq

    do i = 2-num_ghost, num_cells+num_ghost
        do m=1,2
            df = 0.d0
            do mw=1,2
                df = df + s(mw,i)*wave(m,mw,i)
                enddo
            apdq(m,i) = df - amdq(m,i)
            enddo
            
        ! tracer (which is in non-conservation form)
        if (s(3,i) < 0) then
            amdq(m,i) = amdq(m,i) + s(3,i)*wave(m,3,i)
          else
            apdq(m,i) = apdq(m,i) + s(3,i)*wave(m,3,i)
          endif
          
        enddo

    return

    end subroutine rp1



