      subroutine setprob()
      implicit double precision (a-h,o-z)
      character*25 fname
      common /cparam/ grav
      common /comic/ sloc,hl,ul,hr,ur
c
c     # begin_html
c     # Read various parameters from [code: setprob.data].
c
c     # The other parameters are passed to [code: qinit.f] 
c     # in common block comic.
c     # end_html
c
c
      iunit = 7
      fname = 'setprob.data'
c     # open the unit with new routine from Clawpack 4.4 to skip over
c     # comment lines starting with #:
      call opendatafile(iunit, fname)


      read(7,*) g
      grav = g
      drytol = 1.d-3

      read(7,*) sloc
      read(7,*) hl
      read(7,*) ul
      read(7,*) hr
      read(7,*) ur

      return
      end

