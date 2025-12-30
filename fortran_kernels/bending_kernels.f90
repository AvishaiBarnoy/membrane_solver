module bending_kernels_mod
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

  pure function dot3(a, b) result(r)
    real(c_double), intent(in) :: a(3), b(3)
    real(c_double) :: r
    r = a(1)*b(1) + a(2)*b(2) + a(3)*b(3)
  end function dot3

  pure function norm3(a) result(r)
    real(c_double), intent(in) :: a(3)
    real(c_double) :: r
    r = sqrt(dot3(a,a))
  end function norm3

  ! ----------------------------
  ! grad_cotan for batched inputs
  ! u, v: (n, 3)  (native engine layout)
  ! grad_u, grad_v: (n, 3)
  ! ----------------------------

  subroutine grad_cotan_batch(n, u, v, grad_u, grad_v)
    integer(c_int), intent(in) :: n
    real(c_double), intent(in) :: u(n, 3), v(n, 3)
    ! f2py: require caller to supply output buffers to avoid per-call allocations.
    real(c_double), intent(inout) :: grad_u(n, 3), grad_v(n, 3)

  integer :: i
  real(c_double) :: ui(3), vi(3), w(3)
  real(c_double) :: C, S, invS, invS3
  real(c_double) :: v_cross_w(3), w_cross_u(3)
  real(c_double), parameter :: eps = 1.0d-15

  grad_u = 0.0d0
  grad_v = 0.0d0

  do i=1, n
    ui(1) = u(i, 1)
    ui(2) = u(i, 2)
    ui(3) = u(i, 3)
    vi(1) = v(i, 1)
    vi(2) = v(i, 2)
    vi(3) = v(i, 3)

    C = dot3(ui, vi)
    w = cross3(ui, vi)
    S = norm3(w)

    if (S <= eps) cycle

    invS = 1.0d0 / S
    invS3 = 1.0d0 / (S*S*S)

    v_cross_w = cross3(vi, w) ! v x (u x v)
    w_cross_u = cross3(w, ui) ! (u x v) x u

    grad_u(i, 1) = vi(1) * invS - (C * invS3) * v_cross_w(1)
    grad_u(i, 2) = vi(2) * invS - (C * invS3) * v_cross_w(2)
    grad_u(i, 3) = vi(3) * invS - (C * invS3) * v_cross_w(3)

    grad_v(i, 1) = ui(1) * invS - (C * invS3) * w_cross_u(1)
    grad_v(i, 2) = ui(2) * invS - (C * invS3) * w_cross_u(2)
    grad_v(i, 3) = ui(3) * invS - (C * invS3) * w_cross_u(3)
  end do
end subroutine grad_cotan_batch

! -------------------------------------------------------
! Apply discrete Laplace-Beltrami with cotan weight
!
! weights:  (nf, 3) -> [c0, c1, c2] per face
! tri:      (nf, 3) -> [v0, v1, v2] per face
! field:    (nv, dim)
! out:      (nv, dim) (output)
!
! zero_based=1 if tri indices are 0-based (Python); else 0.
! -------------------------------------------------------

subroutine apply_beltrami_laplacian(dim, nv, nf, weights, tri, field, out, zero_based)
    ! f2py-friendly signature (no bind(C)); keep arguments explicit.
    ! dim is usually 3 for vector fields.
  integer(c_int), intent(in) :: dim, nv, nf
  real(c_double), intent(in) :: weights(nf, 3)
  integer(c_int), intent(in) :: tri(nf, 3)
  real(c_double), intent(in) :: field(nv, dim)
  ! f2py: require caller to supply output buffer to avoid per-call allocations.
  real(c_double), intent(inout) :: out(nv, dim)
  integer(c_int), intent(in) :: zero_based

  integer :: f, d
  integer :: v0, v1, v2, shift
  real(c_double) :: c0, c1, c2
  real(c_double) :: f0, f1, f2

  out = 0.0d0

  if (zero_based /= 0_c_int) then
    shift = 1
  else
    shift = 0
  end if

  do f = 1, nf
    c0 = weights(f, 1)
    c1 = weights(f, 2)
    c2 = weights(f, 3)

    v0 = tri(f, 1) + shift
    v1 = tri(f, 2) + shift
    v2 = tri(f, 3) + shift

    if (v0 < 1 .or. v0 > nv) cycle
    if (v1 < 1 .or. v1 > nv) cycle
    if (v2 < 1 .or. v2 > nv) cycle

    do d = 1, dim
      f0 = field(v0, d)
      f1 = field(v1, d)
      f2 = field(v2, d)

      out(v0, d) = out(v0, d) + 0.5d0 * (c1*(f0 - f2) + c2*(f0 - f1))
      out(v1, d) = out(v1, d) + 0.5d0 * (c2*(f1 - f0) + c0*(f1 - f2))
      out(v2, d) = out(v2, d) + 0.5d0 * (c0*(f2 - f1) + c1*(f2 - f0))
    end do
  end do
end subroutine apply_beltrami_laplacian
end module bending_kernels_mod
