def burgers_viscous_time( ):
  import numpy as np
  import platform

  vtn = 11
  vxn = 11
  nu = 0.01 / np.pi

  print ( '' )
  print ( 'burgers_viscous_time_exact1_test01():' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  burgers_viscous_time_exact1() evaluates solution #1' )
  print ( '  to the Burgers equation.' )
  print ( '' )
  print ( '  Viscosity NU = %g' % ( nu ) )
  print ( '  NX = %d' % ( vxn ) )
  print ( '  NT = %d' % ( vtn ) )

  xlo = -1.0
  xhi = +1.0
  vx = np.linspace ( xlo, xhi, vxn )
  r8vec_print ( vxn, vx, '  X grid points:' )

  tlo = 0.0
  thi = 3.0 / np.pi
  vt = np.linspace ( tlo, thi, vtn )
  r8vec_print ( vtn, vt, '  T grid points:' )

  vu = burgers_viscous_time_exact1 ( nu, vxn, vx, vtn, vt )

  r8mat_print ( vxn, vtn, vu, '  U(X,T) at grid points:' )

  filename = 'burgers_solution_test01.txt'

  r8mat_write ( filename, vxn, vtn, vu )

  print ( '' )
  print ( '  Data written to file "%s"' % ( filename ) )
#
#  Terminate
#
  print ( '' )
  print ( 'burgers_viscous_time_exact1_test01():' )
  print ( '  Normal end of execution.' )
  return