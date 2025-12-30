module surface_energy_mod
  use, intrinsic :: iso_c_binding, only: c_int, c_double
  implicit none
contains

  pure function cross3(a, b) result(c)
    real(c_double), intent(in) :: a(3), b(3)
    real(c_double) :: c(3)
    c(1) = a(2)*b(3) - a(3)*b(2)
    c(2) = a(3)*b(1) - a(1)*b(3)
    c(3) = a(1)*b(2) - a(2)*b(1)
  end function cross3

  pure function norm2_3(x) result(r)
    real(c_double), intent(in) :: x(3)
    real(c_double) :: r
    r = sqrt(x(1)*x(1) + x(2)*x(2) + x(3)*x(3))
  end function norm2_3

! Compute total surface energy and gradient for triangle facets.
!
! Conventions:
!   pos is (nv,3) (matches engine's positions_view layout)
!   tri is (nf,3) with 0-based indices if zero_based=1, else 1-based
!
subroutine surface_energy_and_gradient(nv, nf, pos, tri, gamma, grad, E, zero_based)
  integer(c_int), intent(in) :: nv, nf
  real(c_double), intent(in) :: pos(nv, 3)
  integer(c_int), intent(in) :: tri(nf, 3)
  real(c_double), intent(in) :: gamma(nf)
  real(c_double), intent(inout) :: grad(nv, 3)
  real(c_double), intent(out) :: E
  integer(c_int), intent(in) :: zero_based

  integer :: f
  integer :: i0, i1, i2
  real(c_double) :: v0(3), v1(3), v2(3)
  real(c_double) :: e1(3), e2(3), nvec(3), nhat(3)
  real(c_double) :: A2, area, g0(3), g1(3), g2(3)
  real(c_double), parameter :: eps = 1.0d-12
  integer :: shift

  E = 0.0d0
  if (zero_based /= 0_c_int) then
    shift = 1
  else
    shift = 0
  end if

  do f = 1, nf
    i0 = tri(f, 1) + shift
    i1 = tri(f, 2) + shift
    i2 = tri(f, 3) + shift

    ! load vertices
    if (i0 < 1 .or. i0 > nv) cycle
    if (i1 < 1 .or. i1 > nv) cycle
    if (i2 < 1 .or. i2 > nv) cycle
    v0(1) = pos(i0, 1)
    v0(2) = pos(i0, 2)
    v0(3) = pos(i0, 3)
    v1(1) = pos(i1, 1)
    v1(2) = pos(i1, 2)
    v1(3) = pos(i1, 3)
    v2(1) = pos(i2, 1)
    v2(2) = pos(i2, 2)
    v2(3) = pos(i2, 3)
    e1 = v1 - v0
    e2 = v2 - v0
    nvec = cross3(e1, e2)
    A2 = norm2_3(nvec)  ! this is ||cross|| = 2*Area

    if (A2 < eps) cycle

    nhat = nvec / A2
    area = 0.5d0 * A2
    E = E + gamma(f) * area

    ! Area gradients (match Python):
    ! g0 = 0.5 * (v1 - v2) x nhat
    ! g1 = 0.5 * (v2 - v0) x nhat
    ! g2 = 0.5 * (v0 - v1) x nhat
    g0 = 0.5d0 * cross3(v1 - v2, nhat)
    g1 = 0.5d0 * cross3(v2 - v0, nhat)
    g2 = 0.5d0 * cross3(v0 - v1, nhat)

    g0 = gamma(f) * g0
    g1 = gamma(f) * g1
    g2 = gamma(f) * g2

    grad(i0, 1) = grad(i0, 1) + g0(1)
    grad(i0, 2) = grad(i0, 2) + g0(2)
    grad(i0, 3) = grad(i0, 3) + g0(3)
    grad(i1, 1) = grad(i1, 1) + g1(1)
    grad(i1, 2) = grad(i1, 2) + g1(2)
    grad(i1, 3) = grad(i1, 3) + g1(3)
    grad(i2, 1) = grad(i2, 1) + g2(1)
    grad(i2, 2) = grad(i2, 2) + g2(2)
    grad(i2, 3) = grad(i2, 3) + g2(3)
  end do
end subroutine surface_energy_and_gradient
end module surface_energy_mod
