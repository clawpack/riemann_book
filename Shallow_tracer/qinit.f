c
c
c =========================================================
       subroutine qinit(meqn,mbc,mx,xlower,dx,q,maux,aux)
c =========================================================
c
c     # Set initial conditions for q.
c
c
      implicit double precision (a-h,o-z)
      dimension q(meqn,1-mbc:mx+mbc)
      dimension aux(1,1-mbc:mx+mbc)
      common /comic/ sloc,hl,ul,hr,ur
c
c
      do i=1,mx
        xcenter = xlower + (i - 0.5d0) * dx
        if (xcenter < sloc) then
            q(1,i) = hl
            q(2,i) = ul
        else
            q(1,i) = hr
            q(2,i) = ur
        endif
      q(3,i) = xcenter
      enddo
c
      return
      end

