import os.path
import numpy as np
import sys
from . import objUtil

class SlicePerimeter:

    def __init__(self, verts, faces):
        self.verts = verts
        self.faces = faces

    def calc_perimeter(self):
        bmin, bmax = self.calculate_bounding_box(self.verts)
        size = bmax - bmin
        mindim = size.argmin()

        plane_v = 0.5 * (bmin + bmax)
        plane_n = np.array([0, 0, 0])
        plane_n[mindim] = 1.0

        # eps = np.finfo(np.float32).eps
        eps = 0.0000001
        segs = self.isect_mesh_plane(self.verts, self.faces, plane_v, plane_n, eps)

        peri = 0.0
        for seg in segs:
            s = seg[0] - seg[1]
            peri += np.linalg.norm(s)

        return peri


    def isect_segment_plane(self, p1, p0, plane_v, plane_n, eps):
        u = p1 - p0
        w = p0 - plane_v
        D = np.dot(plane_n, u)
        N = -np.dot(plane_n, w)
        if abs(D) < eps:
            if N == 0:
                return 2, None
            else:
                return 0, None
        # they are not parallel
        # compute intersect param
        sI = N / D
        if sI < -eps or sI > 1 + eps:
            return 0, None

        return 1, p0 + sI * u

    def isect_triangle_plane(self, t0, t1, t2, plane_v, plane_n, eps):
        t0v = t0 - plane_v
        t1v = t1 - plane_v
        t2v = t2 - plane_v
        t = [t0, t1, t2]
        d = np.array([0.0, 0.0, 0.0])
        d[0] = np.dot(t0v, plane_n)
        d[1] = np.dot(t1v, plane_n)
        d[2] = np.dot(t2v, plane_n)
        # all triangle vertices lie on the same side of the plane
        if d[0] * d[1] > eps and d[0] * d[2] > eps and d[1] * d[2] > eps:
            return []

        isec_pnts = []
        for i in range(3):
            if abs(d[i]) <= eps:
                isec_pnts.append(t[i])

        for i in range(3):
            i0 = i
            i1 = (i + 1) % 3
            if abs(d[i0]) > eps and abs(d[i1]) > eps and d[i0] * d[i1] < 0:
                ret, p = self.isect_segment_plane(t[i0], t[i1], plane_v, plane_n, eps)
                assert ret == 1
                isec_pnts.append(p)

        return isec_pnts

    def isect_mesh_plane(self, vertices, faces, plane_v, plane_n, eps):
        isct_segments = []
        nverts = len(vertices)
        for f in faces:
            if len(f) == 3:
                v0 = f[0]
                v1 = f[1]
                v2 = f[2]
                assert (v0 < nverts and v1 < nverts and v2 < nverts)
                pnts = self.isect_triangle_plane(
                    np.asarray(vertices[v0]), np.asarray(vertices[v1]), np.asarray(vertices[v2]),
                    plane_v, plane_n, eps)
                if len(pnts) == 2:
                    isct_segments.append(pnts)

        return isct_segments

    def calculate_bounding_box(self, vertices):
        maxflt = np.finfo('float32').max
        bmin = np.array([maxflt, maxflt, maxflt])
        bmax = np.array([-maxflt, -maxflt, -maxflt])
        for v in vertices:
            vco = np.asarray(v)
            bmin = np.minimum(bmin, vco)
            bmax = np.maximum(bmax, vco)

        return bmin, bmax