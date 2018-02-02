import vtk
import numpy as np

class VtkPointsData:
    def __init__(self, points, color = [1.0, 0, 0], psize = 1):
        self.points = points
        self.color = color
        self.psize = psize

    def build_actor(self):
        npoints = len(self.points)

        vtk_pcoords = vtk.vtkFloatArray()
        vtk_pcoords.SetNumberOfComponents(3)
        vtk_pcoords.SetNumberOfTuples(npoints)
        for i in range(0, npoints):
            p = self.points[i]
            vtk_pcoords.SetTuple3(i, p[0], p[1], p[2])

        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(vtk_pcoords)

        vertices = vtk.vtkCellArray()
        vertices.Allocate(2 * npoints, 1000)

        for i in range(0, npoints):
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(i)

        assert vertices.GetNumberOfCells() == npoints

        vtk_poly = vtk.vtkPolyData()
        vtk_poly.SetPoints(vtk_points)
        vtk_poly.SetVerts(vertices)

        # Visualize
        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(vtk_poly)
        else:
            mapper.SetInputData(vtk_poly)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(self.color[0], self.color[1], self.color[2])
        actor.GetProperty().SetPointSize(self.psize)
        return actor


class VtkTrianglesData:
    def __init__(self, verts, faces, color=[1.0, 0, 0], ecolor=[1.0, 1.0, 0]):
        self.verts = verts
        self.faces = faces
        self.color = color
        self.ecolor = ecolor

        #self.width = width

    def build_edge_actor(self):
        npoints = len(self.verts)

        vtk_pcoords = vtk.vtkFloatArray()
        vtk_pcoords.SetNumberOfComponents(3)
        vtk_pcoords.SetNumberOfTuples(npoints)
        for i in range(0, npoints):
            p = self.verts[i]
            vtk_pcoords.SetTuple3(i, p[0], p[1], p[2])

        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(vtk_pcoords)

        nedges = len(self.faces) * 3
        vtk_edges = vtk.vtkCellArray()
        vtk_edges.Allocate(nedges * 2, 1000)

        for f in self.faces:
            assert len(f) == 3
            for i in range(3):
                vtk_edges.InsertNextCell(2)
                vtk_edges.InsertCellPoint(f[i])
                vtk_edges.InsertCellPoint(f[(i+1)%3])

        vtk_poly = vtk.vtkPolyData()
        vtk_poly.SetPoints(vtk_points)
        vtk_poly.SetLines(vtk_edges)

        # Visualize
        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(vtk_poly)
        else:
            mapper.SetInputData(vtk_poly)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(self.ecolor[0], self.ecolor[1], self.ecolor[2])

        return actor

    def build_actor(self):
        npoints = len(self.verts)

        vtk_pcoords = vtk.vtkFloatArray()
        vtk_pcoords.SetNumberOfComponents(3)
        vtk_pcoords.SetNumberOfTuples(npoints)
        for i in range(0, npoints):
            p = self.verts[i]
            vtk_pcoords.SetTuple3(i, p[0], p[1], p[2])

        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(vtk_pcoords)

        nfaces = len(self.faces)
        vtk_faces = vtk.vtkCellArray()
        vtk_faces.Allocate(nfaces * 3, 1000)

        for f in self.faces:
            vtk_faces.InsertNextCell(3)
            for fv in f:
                vtk_faces.InsertCellPoint(fv)

        vtk_poly = vtk.vtkPolyData()
        vtk_poly.SetPoints(vtk_points)
        vtk_poly.SetPolys(vtk_faces)


        # Visualize
        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(vtk_poly)
        else:
            mapper.SetInputData(vtk_poly)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(self.color[0], self.color[1], self.color[2])

        return actor
    
class VtkSegmentsData:
    def __init__(self, segments, color=[1.0, 0, 0], width=1):
        self.segments = segments
        self.color = color
        self.width = width

    def build_actor(self):
        npoints = len(self.segments)

        vtk_pcoords = vtk.vtkFloatArray()
        vtk_pcoords.SetNumberOfComponents(3)
        vtk_pcoords.SetNumberOfTuples(npoints)
        for i in range(0, npoints):
            p = self.segments[i]
            vtk_pcoords.SetTuple3(i, p[0], p[1], p[2])

        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(vtk_pcoords)

        lines = vtk.vtkCellArray()
        lines.Allocate(npoints, 1000)

        for i in range(0, npoints, 2):
            lines.InsertNextCell(2)
            lines.InsertCellPoint(i)
            lines.InsertCellPoint(i+1)

        assert lines.GetNumberOfCells() == npoints/2

        vtk_poly = vtk.vtkPolyData()
        vtk_poly.SetPoints(vtk_points)
        vtk_poly.SetLines(lines)

        # Visualize
        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(vtk_poly)
        else:
            mapper.SetInputData(vtk_poly)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(self.color[0], self.color[1], self.color[2])
        actor.GetProperty().SetLineWidth(self.width)

        return actor


class VtkDrawUtil:
    def __init__(self):
        self.points = []
        self.segments = []
        self.triangles = []

    def add_segments(self, segs, color, width):
        self.segments.append(VtkSegmentsData(segs, color, width))

    def add_points(self, points, color, psize):
        self.points.append(VtkPointsData(points, color, psize))

    def add_triangles(self, verts, faces, color, ecolor):
        self.triangles.append(VtkTrianglesData(verts, faces, color, ecolor))

    def draw(self):
        camera = vtk.vtkCamera()
        camera.SetPosition(1, 1, 1)
        camera.SetFocalPoint(0, 0, 0)

        vtk_renderer = vtk.vtkRenderer()

        for pdata in self.points:
            ator = pdata.build_actor()
            vtk_renderer.AddActor(ator)

        for segdata in self.segments:
            ator = segdata.build_actor()
            vtk_renderer.AddActor(ator)

        for tdata in self.triangles:
            ator = tdata.build_actor()
            vtk_renderer.AddActor(ator)
            etor = tdata.build_edge_actor()
            vtk_renderer.AddActor(etor)

        vtk_renderer.SetActiveCamera(camera)
        vtk_renderer.ResetCamera()
        vtk_renderer.SetBackground(0, 0, 0)

        vtk_renWin = vtk.vtkRenderWindow()
        vtk_renWin.SetSize(800, 600)
        vtk_renWin.AddRenderer(vtk_renderer)

        style = vtk.vtkInteractorStyleTrackballCamera()
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(vtk_renWin)
        iren.SetInteractorStyle(style)
        # interact with data
        vtk_renWin.Render()
        iren.Start()