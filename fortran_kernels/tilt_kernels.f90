module tilt_kernels_mod
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
    r = sqrt(dot3(a, a))
  end function norm3

  subroutine p1_triangle_divergence(nv, nf, positions, tilts, tri, div_tri, area, g0, g1, g2, zero_based)
    integer(c_int), intent(in) :: nv, nf
    real(c_double), intent(in) :: positions(3, nv), tilts(3, nv)
    integer(c_int), intent(in) :: tri(3, nf)
    real(c_double), intent(inout) :: div_tri(nf), area(nf)
    real(c_double), intent(inout) :: g0(3, nf), g1(3, nf), g2(3, nf)
    integer(c_int), intent(in) :: zero_based

    integer :: f
    integer :: v0, v1, v2, shift
    real(c_double) :: v0p(3), v1p(3), v2p(3)
    real(c_double) :: n(3), n2, denom
    real(c_double) :: e0(3), e1(3), e2(3)
    real(c_double) :: t0(3), t1(3), t2(3)
    real(c_double), parameter :: eps = 1.0d-20

    div_tri = 0.0d0
    area = 0.0d0
    g0 = 0.0d0
    g1 = 0.0d0
    g2 = 0.0d0

    if (zero_based /= 0_c_int) then
      shift = 1
    else
      shift = 0
    end if

    do f = 1, nf
      v0 = tri(1, f) + shift
      v1 = tri(2, f) + shift
      v2 = tri(3, f) + shift

      if (v0 < 1 .or. v0 > nv) cycle
      if (v1 < 1 .or. v1 > nv) cycle
      if (v2 < 1 .or. v2 > nv) cycle

      v0p = positions(:, v0)
      v1p = positions(:, v1)
      v2p = positions(:, v2)

      n = cross3(v1p - v0p, v2p - v0p)
      n2 = dot3(n, n)
      denom = max(n2, eps)

      e0 = v2p - v1p
      e1 = v0p - v2p
      e2 = v1p - v0p

      g0(:, f) = cross3(n, e0) / denom
      g1(:, f) = cross3(n, e1) / denom
      g2(:, f) = cross3(n, e2) / denom

      t0 = tilts(:, v0)
      t1 = tilts(:, v1)
      t2 = tilts(:, v2)

      div_tri(f) = dot3(t0, g0(:, f)) + dot3(t1, g1(:, f)) + dot3(t2, g2(:, f))
      area(f) = 0.5d0 * sqrt(max(n2, 0.0d0))
    end do
  end subroutine p1_triangle_divergence

  subroutine compute_curvature_data(nv, nf, positions, tri, k_vecs, vertex_areas, weights, zero_based)
    integer(c_int), intent(in) :: nv, nf
    real(c_double), intent(in) :: positions(3, nv)
    integer(c_int), intent(in) :: tri(3, nf)
    real(c_double), intent(inout) :: k_vecs(3, nv)
    real(c_double), intent(inout) :: vertex_areas(nv)
    real(c_double), intent(inout) :: weights(3, nf)
    integer(c_int), intent(in) :: zero_based

    integer :: f
    integer :: v0, v1, v2, shift
    real(c_double) :: v0p(3), v1p(3), v2p(3)
    real(c_double) :: e0(3), e1(3), e2(3)
    real(c_double) :: l0_sq, l1_sq, l2_sq
    real(c_double) :: cross(3), area_doubled, tri_area
    real(c_double) :: c0, c1, c2
    real(c_double) :: va0, va1, va2
    logical :: is_obtuse_v0, is_obtuse_v1, is_obtuse_v2, any_obtuse
    real(c_double), parameter :: area_eps = 1.0d-12

    k_vecs = 0.0d0
    vertex_areas = 0.0d0
    weights = 0.0d0

    if (zero_based /= 0_c_int) then
      shift = 1
    else
      shift = 0
    end if

    do f = 1, nf
      v0 = tri(1, f) + shift
      v1 = tri(2, f) + shift
      v2 = tri(3, f) + shift

      if (v0 < 1 .or. v0 > nv) cycle
      if (v1 < 1 .or. v1 > nv) cycle
      if (v2 < 1 .or. v2 > nv) cycle

      v0p = positions(:, v0)
      v1p = positions(:, v1)
      v2p = positions(:, v2)

      e0 = v2p - v1p
      e1 = v0p - v2p
      e2 = v1p - v0p

      l0_sq = dot3(e0, e0)
      l1_sq = dot3(e1, e1)
      l2_sq = dot3(e2, e2)

      cross = cross3(e1, e2)
      area_doubled = norm3(cross)
      if (area_doubled < area_eps) area_doubled = area_eps
      tri_area = 0.5d0 * area_doubled

      c0 = dot3(-e1, e2) / area_doubled
      c1 = dot3(-e2, e0) / area_doubled
      c2 = dot3(-e0, e1) / area_doubled

      weights(1, f) = c0
      weights(2, f) = c1
      weights(3, f) = c2

      k_vecs(:, v0) = k_vecs(:, v0) + 0.5d0 * (c1 * (-e1) + c2 * e2)
      k_vecs(:, v1) = k_vecs(:, v1) + 0.5d0 * (c2 * (-e2) + c0 * e0)
      k_vecs(:, v2) = k_vecs(:, v2) + 0.5d0 * (c0 * (-e0) + c1 * e1)

      is_obtuse_v0 = (c0 < 0.0d0)
      is_obtuse_v1 = (c1 < 0.0d0)
      is_obtuse_v2 = (c2 < 0.0d0)
      any_obtuse = is_obtuse_v0 .or. is_obtuse_v1 .or. is_obtuse_v2

      if (.not. any_obtuse) then
        va0 = (l1_sq * c1 + l2_sq * c2) / 8.0d0
        va1 = (l2_sq * c2 + l0_sq * c0) / 8.0d0
        va2 = (l0_sq * c0 + l1_sq * c1) / 8.0d0
      else
        va0 = 0.0d0
        va1 = 0.0d0
        va2 = 0.0d0
        if (is_obtuse_v0) va0 = tri_area / 2.0d0
        if (is_obtuse_v1 .or. is_obtuse_v2) va0 = tri_area / 4.0d0

        if (is_obtuse_v1) va1 = tri_area / 2.0d0
        if (is_obtuse_v0 .or. is_obtuse_v2) va1 = tri_area / 4.0d0

        if (is_obtuse_v2) va2 = tri_area / 2.0d0
        if (is_obtuse_v0 .or. is_obtuse_v1) va2 = tri_area / 4.0d0
      end if

      vertex_areas(v0) = vertex_areas(v0) + va0
      vertex_areas(v1) = vertex_areas(v1) + va1
      vertex_areas(v2) = vertex_areas(v2) + va2
    end do
  end subroutine compute_curvature_data

end module tilt_kernels_mod
