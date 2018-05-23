#include<vcg/complex/complex.h>
#include<wrap/io_trimesh/import.h>
#include<wrap/io_trimesh/export.h>
#include <vcg/space/index/kdtree/kdtree.h>
#include <vcg/space/index/aabb_binary_tree/aabb_binary_tree.h>
#include<vcg/complex/algorithms/update/normal.h>
#include<vcg/complex/algorithms/update/color.h>
#include<vcg/complex/complex.h>
#include<vcg/complex/algorithms/create/platonic.h>
#include <vcg/space/point3.h>
#include <vcg/space/fitting3.h>
#include <vcg/space/plane3.h>
#include <vcg/space/index/octree.h>

#include <igl/boundary_loop.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/list_to_matrix.h>


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Dense>

using namespace vcg;
using namespace std;
using namespace cv;

typedef double ScalarType;
typedef vcg::Point3<ScalarType> vcgPoint3;
typedef vcg::Point2<ScalarType> vcgPoint2;
typedef cv::Vec2d				cvVec2;
typedef cv::Point2d				cvPoint2;
typedef std::array<vcgPoint3, 4> vcgRect3;
typedef Eigen::MatrixX3d		EMatrix3xScalar;
typedef Eigen::MatrixX2d		EMatrix2xScalar;
typedef Eigen::MatrixXd			EMatrixXScalar;
typedef Eigen::MatrixXi			EMatrixX;
typedef Eigen::MatrixX3i		EMatrix3x;
typedef Eigen::MatrixX2i		EMatrix2x;
typedef Eigen::Vector3i			EVector3X;
typedef Eigen::VectorXi			EVectorX;
typedef Eigen::Vector2d			EVector2Scalar;
typedef Eigen::Vector3d			EVector3Scalar;

class MyEdge;
class MyFace;
class MyVertex;
struct MyUsedTypes : public UsedTypes<	Use<MyVertex>   ::AsVertexType,
	Use<MyEdge>     ::AsEdgeType,
	Use<MyFace>     ::AsFaceType> {};

class MyVertex : public Vertex<MyUsedTypes, vertex::VEAdj, vertex::Coord3d, vertex::Normal3d, vertex::BitFlags  > {};
class MyFace : public Face< MyUsedTypes, face::VertexRef, face::FEAdj, face::FFAdj, face::WedgeTexCoord2d, face::BitFlags, face::Mark > {};
class MyEdge : public Edge<MyUsedTypes, edge::VertexRef, edge::VEAdj, edge::EFAdj, edge::BitFlags> {};

typedef vector<MyVertex> VertexContainer;
typedef vector<MyFace> FaceContainer;
typedef vector<MyEdge> EdgeContainer;
class MyMesh : public tri::TriMesh< VertexContainer, FaceContainer , EdgeContainer> {};

typedef MyMesh::VertexPointer	VPointer;
typedef MyMesh::FacePointer		FPointer;
typedef MyMesh::EdgePointer		EPointer;
typedef vcg::Octree<MyMesh::VertexType, double> OctreeType;

struct LocalDecalTriangle
{
	cvVec2		coords[3];
	double		depths[3];
	cvVec2		tex_coords[3];
}; 

template <class S >
void compute_covariance_matrix(const std::vector<vcg::Point3<S> > &pointVec, vcg::Point3<S> &barycenter, Eigen::Matrix<S, 3, 3> &m)
{
	// first cycle: compute the barycenter
	barycenter.SetZero();
	typename  std::vector<vcg::Point3<S> >::const_iterator pit;
	for (pit = pointVec.begin(); pit != pointVec.end(); ++pit) barycenter += (*pit);
		barycenter /= pointVec.size();

	// second cycle: compute the covariance matrix
	m.setZero();
	Eigen::Matrix<S, 3, 1> p;
	for (pit = pointVec.begin(); pit != pointVec.end(); ++pit) {
		((*pit) - barycenter).ToEigenVector(p);
		m += p*p.transpose(); // outer product
	}
}

void calc_decal_space(const vcgRect3 &rect, vcgPoint3 &center, vcgPoint3 axis[3])
{
	center = (rect[0] + rect[1] + rect[2] + rect[3]) * 0.25;
	axis[0] = (rect[1] - rect[0]).normalized();
	axis[1] = (rect[2] - rect[1]).normalized();
	axis[2] = (axis[0] ^ axis[1]).normalized();
}

void calc_decal_space(const MyMesh &mesh, vcgPoint3 &center, vcgPoint3 axis[3])
{
	std::vector<vcgPoint3> points(mesh.VN());
	size_t i = 0;
	vcg::Box3d bb;
	for (MyMesh::ConstVertexIterator vi = mesh.vert.cbegin(); vi != mesh.vert.end(); ++vi, ++i)
	{
		points[i] = (*vi).P();
		bb.Add(points[i]);
	}

	Eigen::Matrix<double, 3, 3> covMat = Eigen::Matrix<double, 3, 3>::Zero();
	vcgPoint3 mass_center;
	compute_covariance_matrix(points, mass_center, covMat);

	Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3> > eig(covMat);
	Eigen::Matrix<double, 3, 1> eval = eig.eigenvalues();
	Eigen::Matrix<double, 3, 3> evec = eig.eigenvectors();
	eval = eval.cwiseAbs();
	int minInd;
	eval.minCoeff(&minInd);
	//normal of the approximated plane
	axis[2][0] = evec(0, minInd);
	axis[2][1] = evec(1, minInd);
	axis[2][2] = evec(2, minInd);
	
	//horizontal
	axis[0][0] = evec(0, (minInd + 1) % 3);
	axis[0][1] = evec(1, (minInd + 1) % 3);
	axis[0][2] = evec(2, (minInd + 1) % 3);

	//vertical
	axis[1][0] = evec(0, (minInd + 2) % 3);
	axis[1][1] = evec(1, (minInd + 2) % 3);
	axis[1][2] = evec(2, (minInd + 2) % 3);

#if 0
	double displacment = 0.5f * bb.Dim()[bb.MaxDim()];
	center = mass_center +  axis[2] * displacment;
#else
	center = mass_center;
#endif
}

vcgPoint3 transform_point_to_decal_space(const vcgPoint3 &center, const vcgPoint3 axis[3], const vcgPoint3 &p)
{
	vcgPoint3 relative_pos = p - center;
	vcgPoint3 local_p;
	for (auto i = 0; i < 3; ++i)
		local_p[i] = relative_pos.dot(axis[i]);
	return local_p;
}

void transform_mesh_to_decal_space(const vcgPoint3 &center, const vcgPoint3 axis[3], MyMesh &mesh, std::vector<LocalDecalTriangle> &local_trigs)
{
	local_trigs.resize(mesh.FN());
	vcg::Box2d bb;
	size_t idx = 0;

	for (MyMesh::ConstFaceIterator fit = mesh.face.begin(); fit != mesh.face.end(); ++fit)
	{
		const MyMesh::FaceType &f  = *fit;
		if (f.VN() == 3)
		{
			for (auto iv = 0; iv < 3; ++iv)
			{
				vcgPoint3 local_p = transform_point_to_decal_space(center, axis, f.cP(iv));
				local_trigs[idx].coords[iv] = cvPoint2(local_p[0], local_p[1]);
				local_trigs[idx].depths[iv] = std::abs(local_p[2]);
				local_trigs[idx].tex_coords[iv] = cvVec2(f.cWT(iv).U(), f.cWT(iv).V());

				bb.Add(vcg::Abs(vcgPoint2(local_p[0], local_p[1])));
			}
			idx++;
		}
	}

	//normalize
	//transform to the range [0,1]*[0,1]
	vcgPoint2 extent = bb.max;
	for (auto i = 0; i < local_trigs.size(); ++i)
	{
		for (auto d = 0; d < 3; ++d)
		{
			local_trigs[i].coords[d][0] /= extent[0]; 
			local_trigs[i].coords[d][1] /= extent[1];
			local_trigs[i].coords[d] = 0.5f * (local_trigs[i].coords[d] + cvVec2(1.0f, 1.0f));
		}
	}
}

void mesh_matrix(MyMesh &mesh, EMatrixXScalar &V, EMatrixX &F)
{
	//collect all boundary vertices
	V.resize(mesh.VN(),3);
	F.resize(mesh.FN(),3);
	for (auto vit = mesh.vert.begin(); vit != mesh.vert.end(); ++vit)
	{
		auto idx = vcg::tri::Index(mesh, *vit);
		const vcgPoint3 &co = vit->cP();
		V(idx, 0) = co[0];
		V(idx, 1) = co[1];
		V(idx, 2) = co[2];
	}

	int face_v_idx[3];
	for (auto fit = mesh.face.begin(); fit != mesh.face.end(); ++fit)
	{
		auto idx = vcg::tri::Index(mesh, *fit);
		for (int i = 0; i < 3; ++i)
			F(idx, i) = vcg::tri::Index(mesh, fit->V(i));
	}
}

void mesh_matrix(MyMesh &mesh, std::vector<FPointer> &trigs, EMatrixXScalar &V, EMatrixX &F, EVectorX &vmap)
{
	vmap.resize(mesh.VN());
	vmap.array().fill(-1);
	std::vector<EVector3Scalar> vcoords;
	F.resize(trigs.size(), 3);
	int cnt = 0;
	for (int ti = 0; ti < trigs.size(); ++ti)
	{
		FPointer t = trigs[ti];
		for (int i = 0; i < 3; ++i)
		{
			int idx = vcg::tri::Index(mesh, t->V(i));
			if (vmap[idx] == -1)
			{
				vmap[idx] = cnt++;
				const vcgPoint3 &co = t->V(i)->cP();
				vcoords.push_back(EVector3Scalar(co[0], co[1], co[2]));
			}

			F(ti, i) = vmap[idx];
		}
	}

	V.resize(vcoords.size(), 3);
	for (size_t i = 0; i < vcoords.size(); ++i)
		V.row(i) = vcoords[i];
}

void distort_circle_to_square(EMatrixXScalar &circle_points)
{
	size_t n = circle_points.rows();
	for (size_t i = 0; i < n; ++i)
	{
		EVector2Scalar p = circle_points.row(i);
		EVector2Scalar dif = EVector2Scalar(1.0, 1.0) - p.cwiseAbs();
		size_t closest_idx = dif[0] < dif[1] ? 0 : 1;
		p[closest_idx] = p[closest_idx] < 0 ? -1 : 1;
		circle_points.row(i) = p;
	}
}

double calc_path_len(const std::vector<VPointer> &path)
{
	double len = 0;
	for (int i = 1; i < path.size(); ++i)
		len += (path[i]->cP() - path[i - 1]->cP()).Norm();
	return len;
}

void construct_uv_rect_boundary(
	const std::vector<EVector2Scalar> &corners,
	const std::vector<std::vector<VPointer>> &paths, 
	std::vector<EMatrixXScalar> &uvpaths)
{
	assert(corners.size() == 4);
	assert(paths.size() == 4);
	assert(uvpaths.size() == 4);
	for (int i = 0; i <4; ++i){
		assert(!paths[i].empty());
	}

	for (int i = 0; i < 4; ++i)
	{
		const std::vector<VPointer> &path = paths[i];
		size_t  n_path_verts = path.size();
		double	path_len = calc_path_len(path);
		
		EVector2Scalar uv_start = corners[i], uv_end = corners[(i + 1)%4];
		EVector2Scalar	edge = uv_end - uv_start;
		
		EMatrixXScalar &uvpath = uvpaths[i];
		uvpath.resize(n_path_verts,2);
		uvpath.row(0) = uv_start;

		double went_so_far = 0.;
		for (int j = 1; j < n_path_verts; ++j)
		{
			went_so_far += (path[j]->cP() - path[j - 1]->cP()).Norm();
			uvpath.row(j) = uv_start + (went_so_far / path_len) * edge;
		}
	}
}

void parameterize_mesh_to_rectangular_domain(const EMatrixXScalar &V, const EMatrixX &F, EMatrixXScalar &V_uv)
{
	EVectorX bnd;
	igl::boundary_loop(F, bnd);

	// Map the boundary to a circle, preserving edge proportions
	EMatrixXScalar bnd_uv;
	igl::map_vertices_to_circle(V, bnd, bnd_uv);

	//distort_circle_to_square(bnd_uv);

	// Harmonic parametrization for the internal vertices
	igl::harmonic(V, F, bnd, bnd_uv, 1, V_uv);

	V_uv = 0.5 * (1. + V_uv.array());
}

CvRect triangle_rect(const cvPoint2 coords[3])
{
	double xmin, xmax, ymin, ymax;
	xmin = xmax = coords[0].x;
	ymin = ymax = coords[0].y;
	for (auto i = 1; i < 3; i++)
	{
		const cvPoint2 &pt = coords[i];
		if (xmin > pt.x)
			xmin = pt.x;
		if (xmax < pt.x)
			xmax = pt.x;
		if (ymin > pt.y)
			ymin = pt.y;
		if (ymax < pt.y)
			ymax = pt.y;
	}

	return CvRect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
}

bool inside_triangle(const cvPoint2 trig_coords[3], const cvPoint2 &p, double &L1, double &L2, double &L3, const double EPSILON)
{
	//const double EPSILON = double(0.1);
	double x1 = trig_coords[0].x;
	double x2 = trig_coords[1].x;
	double x3 = trig_coords[2].x;

	double y1 = trig_coords[0].y;
	double y2 = trig_coords[1].y;
	double y3 = trig_coords[2].y;

	double x = p.x;
	double y = p.y;

	L1 = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3));
	L2 = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / ((y3 - y1)*(x2 - x3) + (x1 - x3)*(y2 - y3));
	L3 = 1 - L1 - L2;
	if (math::IsNAN(L1) || math::IsNAN(L2) || math::IsNAN(L3)) L1 = L2 = L3 = (double)(1.0 / 3.0);
	bool inside = true;
	inside &= (L1 >= 0 - EPSILON) && (L1 <= 1 + EPSILON);
	inside &= (L2 >= 0 - EPSILON) && (L2 <= 1 + EPSILON);
	inside &= (L3 >= 0 - EPSILON) && (L3 <= 1 + EPSILON);
	return inside;
}

bool import_decal_mesh_plane(std::string file_path, vcgRect3 &decal_rect)
{
	MyMesh mesh;
	tri::io::Importer<MyMesh>::Open(mesh, file_path.c_str());
	if (mesh.VertexNumber() == 4)
	{
		for (auto vit = mesh.vert.begin(); vit != mesh.vert.end(); ++vit)
			decal_rect[std::distance(mesh.vert.begin(), vit)] = vit->cP();
		return true;
	}
	return false;
}

bool import_decal_rectangle(std::string file_path, vcgRect3 &decal_rect)
{
	std::ifstream ff(file_path);
	if (ff.good())
	{
		for (int iv = 0; iv < 4; ++iv)
		{
			for (int d = 0; d < 3; ++d)
			{
				double co; ff >> co;
				decal_rect[iv][d] = co;
			}
		}
		return true;
	}
	else
		return false;
}

VPointer edge_other_vert(MyMesh::EdgePointer e, VPointer v)
{
	return (e->V(0) == v) ? e->V(1) : e->V(0);
}

EPointer edge_from_verts(VPointer v0, VPointer v1)
{
	edge::VEIterator<MyMesh::EdgeType> ve_iter(v0);
	while (!ve_iter.End())
	{
		VPointer other_v = edge_other_vert(ve_iter.E(), v0);
		if (other_v == v1)
			return ve_iter.E();
		++ve_iter;
	}
	return nullptr;
}

bool find_geodesic_path(MyMesh &mesh, VPointer vstart, VPointer vend, std::vector<VPointer> &path, int max_path_len = 10000)
{
	std::vector<int> path_trace(mesh.VN(),-1);
	std::vector<double> short_distances(mesh.VN(),-1.);

	bool reach_end = false;

	for (auto vit = mesh.vert.begin(); vit != mesh.vert.end(); ++vit)
		vit->Flags() = 0;

	int cur_path_len = 0;

	std::deque<VPointer> vqueue;
	vstart->Flags() |= MyMesh::VertexType::VISITED;
	vqueue.push_front(vstart);
	short_distances[vcg::tri::Index(mesh, vstart)] = 0.0;
	while (!vqueue.empty())
	{
		VPointer v = vqueue.front();
		vqueue.pop_front();

		int v_idx = vcg::tri::Index(mesh, v);
		
		bool got_one = false;

		double cur_dst = short_distances[v_idx];
		edge::VEIterator<MyMesh::EdgeType> ve_iter(v);
		while (!ve_iter.End())
		{
			VPointer other_v = edge_other_vert(ve_iter.E(), v);
			size_t other_v_idx = vcg::tri::Index(mesh, other_v);
			double e_len = (other_v->cP() - v->cP()).Norm();
			bool is_visited = other_v->cFlags() & MyMesh::VertexType::VISITED;
			if (is_visited)
			{
				if (short_distances[other_v_idx] > cur_dst + e_len)
				{
					short_distances[other_v_idx] = cur_dst + e_len;
					path_trace[other_v_idx] = v_idx;
				}
			}
			else
			{
				other_v->Flags() |= MyMesh::VertexType::VISITED;

				path_trace[other_v_idx] = v_idx;
				short_distances[other_v_idx] = cur_dst + e_len;

				vqueue.push_back(other_v);
			}

			if (other_v == vend)
			{
				reach_end = true;
				break;
			}

			++ve_iter;
		}

		if (reach_end)
		{
			break;
		}

		cur_path_len++;

		if (cur_path_len > max_path_len)
			break;
	}

	if (reach_end)
	{
		int trace_idx = vcg::tri::Index(mesh, vend);

		while (path_trace[trace_idx] != -1)
		{
			path.push_back(&mesh.vert[path_trace[trace_idx]]);
			trace_idx = path_trace[trace_idx];
		}
		std::reverse(path.begin(), path.end());




		return true;
	}
	else 
	{
		return false;
	}
}

bool find_decal_boundary(MyMesh &mesh, OctreeType &octree, vcgRect3 &decal_rect, std::vector<VPointer> &decal_verts, std::vector<std::vector<VPointer>> &paths)
{
	const double max_dst = (decal_rect[0] - decal_rect[1]).Norm();
	double min_dst;
	decal_verts.resize(4);
	for (int i = 0; i < 4; ++i) 
	{
		decal_verts[i] = vcg::tri::GetClosestVertex(mesh, octree, decal_rect[i], max_dst, min_dst);
		if (decal_verts[i] == nullptr)
			return false;
	}

	paths.resize(4);
	for (auto &p : paths) p.clear();
	
	for (int i = 0; i < 4; ++i)
	{
		VPointer vstart = decal_verts[i];
		VPointer vend   = decal_verts[(i + 1)%4];
		find_geodesic_path(mesh, vstart, vend, paths[i], 100000);
	}

	for (int i = 0; i < 4; ++i)
		if (paths[i].empty())
			return false;
	return true;
}


void merge_path(std::vector<std::vector<VPointer>> &paths, std::vector<VPointer> &path)
{
	for (auto &p : paths)
		path.insert(path.end(), p.begin(), p.end());
}

bool extract_decal_triangles(MyMesh &mesh, std::vector<std::vector<VPointer>> &boundaries, FPointer seed_trig, std::vector<FPointer> &decal_trigs)
{
	auto is_boundary_edge = [](EPointer e) -> bool
	{
		return e->Flags() & MyMesh::EdgeType::VISITED;
	};

	//propagate from the center triangle
	std::deque<FPointer> queue;
	seed_trig->Flags() |= MyMesh::FaceType::VISITED;
	queue.push_back(seed_trig);
	decal_trigs.push_back(seed_trig);

	for (auto fit = mesh.face.begin(); fit != mesh.face.end(); ++fit)
		fit->Flags() = (fit->Flags() & ~MyMesh::FaceType::VISITED);
	for (auto it = mesh.edge.begin(); it != mesh.edge.end(); ++it)
		it->Flags() = (it->Flags() & ~MyMesh::EdgeType::VISITED);

	//close the boundary to let no triangle escape.
	std::vector<VPointer> closed_bdr;
	merge_path(boundaries, closed_bdr);
	size_t npoints = closed_bdr.size();
	for (int i = 0; i < closed_bdr.size(); ++i)
	{
		VPointer v0 = closed_bdr[i];
		VPointer v1 = closed_bdr[(i + 1)%npoints];
		EPointer e = edge_from_verts(v0, v1);
		assert(e != nullptr);
		e->Flags() |= MyMesh::EdgeType::VISITED;
	}
	
	face::Pos<MyMesh::FaceType> pos;
	while (!queue.empty())
	{
		FPointer face = queue.front();
		queue.pop_front();
		for (int i = 0; i < 3; ++i)
		{
			FPointer fadj = face->FFp(i);
			if (!fadj) continue;
			EPointer e	= face->FEp(i);
			bool is_visited = fadj->cFlags() & MyMesh::FaceType::VISITED;
			if (!is_visited)
			{
				fadj->Flags() |= MyMesh::FaceType::VISITED;
			
				if (!is_boundary_edge(e)) 
				{
					decal_trigs.push_back(fadj);
					queue.push_back(fadj);
				}
			}
		}
	}

	for (auto fit = mesh.face.begin(); fit != mesh.face.end(); ++fit)
		fit->Flags() = (fit->Flags() & ~MyMesh::FaceType::VISITED);

	for (auto it = mesh.edge.begin(); it != mesh.edge.end(); ++it)
		it->Flags() = (it->Flags() & ~MyMesh::EdgeType::VISITED);

	return true;
}

bool is_tri_candiate(FPointer tri, const vcg::Plane3d &plane, const vcg::Box3d &box, const double &furthest_dist)
{
	double dst = std::abs(vcg::SignedDistancePlanePoint(plane, vcg::Barycenter(*tri)));
	if (dst > furthest_dist)
		return false;
	
	for (int i = 0; i < 3; ++i)
	{
		vcgPoint3 proj_point = plane.Projection(tri->V(i)->cP());
		if (box.IsIn(proj_point))
			return true;
	}
	return false;
}

void construct_a_mesh(MyMesh &mesh, const std::vector<FPointer> &tris, MyMesh &new_mesh)
{
	for (auto &f : mesh.face)
		f.ClearS();
	for (auto &v : mesh.vert)
		v.ClearS();

	for (const FPointer &tri : tris)
	{
		for (auto i = 0; i < 3; ++i)
		{
			tri->V(i)->SetS();
			tri->SetS();
		}
	}

	vcg::tri::Append<MyMesh,MyMesh>::Selected(new_mesh, mesh);

	for (const FPointer &tri : tris)
	{
		for (auto i = 0; i < 3; ++i)
		{
			tri->V(i)->ClearS();
			tri->ClearS();
		}
	}
}

double estimate_distance(MyMesh &mesh, const vcg::Plane3d &plane, const vcgPoint3 &closest_p_on_mesh)
{
	vcg::tri::UpdateBounding<MyMesh>::Box(mesh);
	double body_width = mesh.bbox.Dim()[mesh.bbox.MinDim()];
	double distance = std::abs(vcg::SignedDistancePlanePoint(plane, closest_p_on_mesh));
	return 0.2 * body_width + distance;
}

FPointer find_seed_triangle(MyMesh &mesh, OctreeType &octree, vcgRect3 &decal_rect)
{
	vcgPoint3 center = (decal_rect[0] + decal_rect[1] + decal_rect[2] + decal_rect[3])*0.25;
	double min_dst;
	const double max_dst = octree.BoundingBox().Dim()[octree.BoundingBox().MinDim()];
	VPointer closest = vcg::tri::GetClosestVertex(mesh, octree, center, max_dst, min_dst);
	
	edge::VEIterator<MyMesh::EdgeType> ve_iter(closest);
	while (!ve_iter.End())
	{
		EPointer e = ve_iter.E();
		if (e && e->EFp())
			return e->EFp();
		++ve_iter;
	}

	return nullptr;
}

FPointer find_seed_triangle(MyMesh &mesh, const vcgRect3 &decal_rect)
{
	vcgPoint3 center = (decal_rect[0] + decal_rect[1] + decal_rect[2] + decal_rect[3])*0.25;
	vcgPoint3 normal = ((decal_rect[1] - decal_rect[0]) ^ (decal_rect[2] - decal_rect[1])).normalized();

	typedef vcg::AABBBinaryTreeIndex<MyMesh::FaceType, MyMesh::ScalarType> Spacial;
	Spacial spacial;
	spacial.Set(mesh.face.begin(), mesh.face.end());
	double min_dst = std::numeric_limits<double>::max();
	const  double max_dst_limit = min_dst;
	vcg::Ray3d ray(center, normal);
	FPointer closest_face = vcg::tri::DoRay(mesh, spacial, ray, max_dst_limit, min_dst);
	if (!closest_face)
	{
		normal = -normal;
		vcg::Ray3d ray(center, normal);
		closest_face = vcg::tri::DoRay(mesh, spacial, ray, max_dst_limit, min_dst);
		if (!closest_face)
			return nullptr;
		return closest_face;
	}
	return nullptr;
}

vcg::Plane3d which_plane(const vcgRect3 &decal_rect, const vcgPoint3 &point)
{
	vcgPoint3 normal = ((decal_rect[1] - decal_rect[0]) ^ (decal_rect[2] - decal_rect[1])).normalized();
	vcgPoint3 center = (decal_rect[0] + decal_rect[1] + decal_rect[2] + decal_rect[3])*0.25;
	if ((point - center).dot(normal) < 0)
		normal = -normal;

	vcg::Plane3d rect_plane;
	rect_plane.Init(center, normal);
	return rect_plane;
}

bool find_candidate_triangles(MyMesh &mesh, const vcgRect3 &decal_rect, std::vector<FPointer> &r_tri_candidates)
{
	vcg::Box3d rect_box;
	for (int i = 0; i < 4; ++i)
		rect_box.Add(decal_rect[i]);

	FPointer seed_face = find_seed_triangle(mesh, decal_rect);
	if (seed_face)
	{
		vcgPoint3 seed_point = vcg::Barycenter(*seed_face);
		vcg::Plane3d rect_plane = which_plane(decal_rect, seed_point);

		double furthest_dst = estimate_distance(mesh, rect_plane, seed_point);

		//propagate from the center triangle
		std::deque<FPointer> queue;
		seed_face->Flags() |= MyMesh::FaceType::VISITED;
		queue.push_back(seed_face);
		r_tri_candidates.push_back(seed_face);
		while (!queue.empty())
		{
			FPointer face = queue.front();
			queue.pop_front();
			for (int i = 0; i < 3; ++i)
			{
				FPointer fadj = face->FFp(i);
				
				if (fadj && !(fadj->cFlags() & MyMesh::FaceType::VISITED) &&  is_tri_candiate(fadj, rect_plane, rect_box, furthest_dst))
				{
					fadj->Flags() |= MyMesh::FaceType::VISITED;
					queue.push_back(fadj);
					r_tri_candidates.push_back(fadj);
				}
			}
		}
		return true;
	}

	return false;
}

void rasterize_triangles(const std::vector<LocalDecalTriangle> &trigs, const cv::Size2i &img_size, cv::Mat1f &depth, cv::Mat2f &tex_coords)
{
	depth = cv::Mat1f(img_size, std::numeric_limits<double>::max());
	tex_coords = cv::Mat2f(img_size);
	cvPoint2 img_trig_verts[3];

	for(auto i = 0; i < trigs.size(); ++i)
	{
		const auto &trig = trigs[i];

		//to image space
		for (auto vi = 0; vi < 3; ++vi)
		{
			img_trig_verts[vi].x = trig.coords[vi][0] * (double)img_size.width;
			img_trig_verts[vi].y = trig.coords[vi][1] * (double)img_size.height;
		}

		CvRect rect = triangle_rect(img_trig_verts);
		double L1, L2, L3;
		for (int ix = 0; ix < rect.width; ix++)
		{
			int px = rect.x + ix;
			for (int iy = 0; iy < rect.height; iy++)
			{
				int py = rect.y + iy;
				//note: our unit is one pixel, therefore, choosing a big epsilon to make sure that each pixel belongs to a triangle
				if (inside_triangle(img_trig_verts, cvPoint2((double)px, (double)py), L1, L2, L3, 0.9))
				{
					double d = L1*trig.depths[0]  + L2*trig.depths[1] * L3*trig.depths[2];
					if (d < depth(px, py))
					{
						depth(px, py) = d;
						//cvVec2 tex((double)rand()/RAND_MAX, (double)rand()/RAND_MAX);
						//cvVec2 tex = 1.0f/3.0f*(trig.tex_coords[0] + trig.tex_coords[1] + trig.tex_coords[2]);
						cvVec2 tex = L1*trig.tex_coords[0] + L2*trig.tex_coords[1] + L3*trig.tex_coords[2];
						//cvVec2 tex = trig.tex_coords[0];
						tex_coords(px, py)[0] = tex[0];
						tex_coords(px, py)[1] = tex[1];
					}
				}
			}
		}

		//if (i > 1000)
		//	break;
	}
}

void test_draw_triangles_over_texture(cv::Mat1b &tex, const EMatrixX &F, const EMatrixXScalar &V_uv)
{
	cv::Point trig_tex_coords[3];
	int width = tex.size[0];
	int height = tex.size[1];

	for (auto i = 0; i < F.rows(); ++i)
	{
		const EVectorX &tri = F.row(i);
		for (auto i = 0; i < 3; ++i)
		{
			trig_tex_coords[i].x = int(V_uv(tri[i],0)* (double)width);
			trig_tex_coords[i].y = int(V_uv(tri[i],1) * (double)height);
		}

		//cv::fillConvexPoly(tex, trig_tex_coords, 3, Scalar(255));

		for (auto i = 0; i < 3; ++i)
			cv::line(tex, trig_tex_coords[i], trig_tex_coords[(i + 1) % 3], Scalar(255, 0, 0), 1);
	}
}

void test_draw_triangles_over_texture(cv::Mat1b &tex, const std::vector<LocalDecalTriangle> &trigs)
{
	cv::Point trig_tex_coords[3];
	int width = tex.size[0];
	int height = tex.size[1];

	for (const LocalDecalTriangle &trig : trigs)
	{
		for (auto i = 0; i < 3; ++i)
		{
			trig_tex_coords[i].x = int(trig.tex_coords[i][0] * (double)width);
			trig_tex_coords[i].y = int((1.0 - trig.tex_coords[i][1]) * (double)height);
		}

		cv::fillConvexPoly(tex, trig_tex_coords, 3, Scalar(255));

		//for (auto i = 0; i < 3; ++i)
		//	cv::line(tex, trig_tex_coords[i], trig_tex_coords[(i + 1) % 3], Scalar(255, 0, 0), 1);
	}
}

cv::Mat3b test_draw_triangles_over_image(const cv::Mat3b &img, const std::vector<LocalDecalTriangle> &trigs)
{
	cv::Point tri_coords[3];
	int width = img.size[0];
	int height = img.size[1];

	cv::Mat3b img_out = img.clone();

	for (const LocalDecalTriangle &trig : trigs)
	{
		for (auto i = 0; i < 3; ++i)
		{
			tri_coords[i].y = int(trig.coords[i][0] * (double)width);
			tri_coords[i].x = int(trig.coords[i][1] * (double)height);
		}

		for (auto i = 0; i < 3; ++i)
			cv::line(img_out, tri_coords[i], tri_coords[(i + 1) % 3], Scalar(255, 0, 0), 1, cv::LINE_4);
	}

	return img_out;
}

cv::Mat3b test_draw_triangles(const cv::Size &size, const std::vector<LocalDecalTriangle> &trigs)
{
	cv::Point trig_coords[3];
	cv::Mat3b img_out(size, cv::Vec3b(0,0,0));

	for (const LocalDecalTriangle &trig : trigs)
	{
		for (auto i = 0; i < 3; ++i)
		{
			trig_coords[i].x = int(trig.coords[i][0] * (double)size.width);
			trig_coords[i].y = int(trig.coords[i][1] * (double)size.height);
		}

		for (auto i = 0; i < 3; ++i)
			cv::line(img_out, trig_coords[i], trig_coords[(i + 1) % 3], Scalar(255, 255, 255), 1);
	}

	return img_out;
}

void test_draw_segments(cv::Mat3b &mat, const std::vector<EMatrixXScalar> &paths)
{
	int width = mat.cols;
	int height = mat.rows;

	for (const EMatrixXScalar &path : paths)
	{
		size_t npoints = path.rows();
		for (int i = 0; i < npoints - 1; ++i)
		{
			cv::Point p0(path(i,0) * width, path(i,1) * height);
			cv::Point p1(path(i+1,0) * width, path(i+1,1) * height);
			cv::line(mat, p0, p1, Scalar(255, 255, 255), 1);
		}
	}
}

cv::Mat3b generate_background_image(cv::Size size, cv::Vec3b mean_color, cv::Vec3b variance)
{
	cv::RNG rng(12345);
	cv::Mat3b img(size.width, size.height);
	rng.fill(img, cv::RNG::NORMAL, mean_color, variance);
	return img;
}

cv::Vec3b find_background_color(const cv::Mat3b &img)
{
	cv::Mat hist[3];
	cv::Mat bgr_planes[3];
	cv::split(img, bgr_planes);
	int histSize = 256;

	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	
	/// Compute the histograms:
	cv::calcHist(&bgr_planes[0], 1, 0, Mat(), hist[0], 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&bgr_planes[1], 1, 0, Mat(), hist[1], 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&bgr_planes[2], 1, 0, Mat(), hist[2], 1, &histSize, &histRange, uniform, accumulate);

	int max_val[3] = { -INT_MAX,-INT_MAX,-INT_MAX };
	int max_idx[3];
	for (int h = 0; h < 3; ++h)
	{
		const cv::Mat &cur_hist = hist[h];
		for (int i = 1; i < 256; ++i)
		{
			if (cur_hist.at<int>(i, 0) > max_val[h])
			{
				max_val[h] = cur_hist.at<int>(i, 0);
				max_idx[h] = i;
			}
		}
	}

	return cv::Vec3b(max_idx[0], max_idx[1], max_idx[2]);
}

cv::Mat3b build_textured_rasterization(const cv::Size &size, const cv::Mat3b &tex_img, const cv::Mat3b &decal_img, cv::Rect2i decal_rect, const cv::Mat2f  &tex_coords)
{
	const cv::MatSize &tex_size = tex_img.size;
	cv::Mat3b textured_raserization_img = cv::Mat3b(size, cv::Vec3b(0, 0, 0));
	for (auto i = 0; i < size.width; ++i)
	{
		for (auto j = 0; j < size.height; ++j)
		{
			cvVec2 co = tex_coords(i, j);
			co[1] = 1.0 - co[1];
			int ix = int((double)tex_size[0] * co[1]);
			int iy = int((double)tex_size[1] * co[0]);
			if (ix >= 0 && ix < tex_size[0] && iy >= 0 && iy < tex_size[1])
			{
				cv::Vec3b pix_val = tex_img(ix, iy);
				textured_raserization_img(i, j) = pix_val;
			}
		}
	}
	return textured_raserization_img;
}

cv::Mat3b output_textured_rasterization(const cv::Size &size, const cv::Mat3b &tex_img, const cv::Mat3b &decal_img, cv::Rect2i decal_rect, const cv::Mat2f  &tex_coords)
{
	const cv::MatSize &tex_size = tex_img.size;
	cv::Mat3b textured_raserization_img = cv::Mat3b(size, cv::Vec3b(0, 0, 0));
	for (auto i = 0; i < size.width; ++i)
	{
		for (auto j = 0; j < size.height; ++j)
		{
			cvVec2 co = tex_coords(i, j);
			co[1] = 1.0 - co[1];
			int ix = int((double)tex_size[0] * co[1]);
			int iy = int((double)tex_size[1] * co[0]);
			if (ix >= 0 && ix < tex_size[0] && iy >= 0 && iy < tex_size[1])
			{
				cv::Vec3b pix_val = tex_img(ix, iy);
				textured_raserization_img(i, j) = pix_val;
			}
		}
	}

	return textured_raserization_img;
}

size_t total_vertices(const std::vector<std::vector<VPointer>> &boundary)
{
	size_t cnt = 0;
	for (int i = 0; i < boundary.size(); ++i)
		cnt += boundary[i].size();
	return cnt;
}

void debug_mesh_points(MyMesh &mesh, std::vector<VPointer> &points)
{
	vcg::Box3d bbox;
	for (auto p : mesh.vert)
		bbox.Add(p.cP());

	vcgPoint3 dim = bbox.Dim();
	cv::Mat3b mat(512, 512, cv::Vec3b(0, 0, 0));
	auto transform = [&](vcgPoint3 &p)
	{
		vcgPoint3 p0 = (p - bbox.min);
		p0.X() = p0.X() / dim.X();
		p0.Y() = p0.Y() / dim.Y();
		p0.Z() = p0.Z() / dim.Z();
		p0 *= 512;
		return cv::Point(p0.X(), p0.Z());
	};

	for (size_t i = 0; i < mesh.FN(); ++i)
	{
		auto color = cv::Vec3b(255, 0, 0);
		FPointer t = &mesh.face[i];
		cv::line(mat, transform(t->cP(0)), transform(t->cP(1)), color);
		cv::line(mat, transform(t->cP(1)), transform(t->cP(2)), color);
		cv::line(mat, transform(t->cP(2)), transform(t->cP(0)), color);
	}

	for (size_t i = 0; i < points.size(); ++i)
	{
		VPointer v0 = points[i];
		auto color = cv::Vec3b(0, 255, 255);
		cv::circle(mat, transform(v0->cP()), 5, color);
	}

	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\test_1.png", mat);
}

void debug_triangle_boundary(MyMesh &mesh, std::vector<FPointer> &decal_trigs, std::vector<std::vector<VPointer>> &boundary, EVectorX &vmap)
{
	std::vector<VPointer> bdr;
	merge_path(boundary, bdr);
	vcg::Box3d bbox;
	for (auto p : mesh.vert)
		bbox.Add(p.cP());

	vcgPoint3 dim = bbox.Dim();
	cv::Mat3b mat(2048, 2048, cv::Vec3b(0, 0, 0));
	auto transform = [&](vcgPoint3 &p)
	{
		vcgPoint3 p0 = (p - bbox.min);
		p0.X() = p0.X() / dim.X();
		p0.Y() = p0.Y() / dim.Y();
		p0.Z() = p0.Z() / dim.Z();
		p0 *= 2048;
		return cv::Point(p0.X(), p0.Y());
	};
	
	for (size_t i =0 ; i < decal_trigs.size(); ++i)
	{
		auto color = cv::Vec3b(0, 255, 0);
		FPointer t = decal_trigs[i];
		cv::line(mat, transform(t->cP(0)), transform(t->cP(1)), color);
		cv::line(mat, transform(t->cP(1)), transform(t->cP(2)), color);
		cv::line(mat, transform(t->cP(2)), transform(t->cP(0)), color);
	}

	for (size_t i = 0; i < bdr.size(); ++i)
	{
		VPointer v0 = bdr[i];
		VPointer v1 = bdr[(i + 1) % bdr.size()];
		cv::Point p0 = transform(v0->cP());
		cv::Point p1 = transform(v1->cP());
		cv::Vec3b color(255, 0, 0);
		if (vmap[vcg::tri::Index(mesh, v0)] == -1 || vmap[vcg::tri::Index(mesh, v1)] == -1)
			color = cv::Vec3b(0, 0, 255);
		cv::line(mat, p0, p1, color);
	}

	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\test.png", mat);
}

void parameterizre_decal(MyMesh &mesh, std::vector<FPointer> &decal_trigs, std::vector<std::vector<VPointer>> &boundary)
{
	std::vector<EVector2Scalar> corners(4);
	corners[0] = EVector2Scalar(1.0, 1.0);
	corners[1] = EVector2Scalar(-1.0, 1.0);
	corners[2] = EVector2Scalar(-1.0, -1.0);
	corners[3] = EVector2Scalar(1.0, -1.0);

	std::vector<EMatrixXScalar> uvs(4);
	construct_uv_rect_boundary(corners, boundary, uvs);

	EMatrixXScalar V;
	EMatrixX F;	
	EVectorX v_map;
	mesh_matrix(mesh, decal_trigs, V, F, v_map);

	debug_triangle_boundary(mesh, decal_trigs, boundary, v_map);

	return;

	size_t n_bdr_points = total_vertices(boundary);
	EVectorX bnd(n_bdr_points);
	EMatrixXScalar bnd_uv(n_bdr_points, 2);
	size_t cnt = 0;
	for (size_t path_idx = 0; path_idx < 4; ++path_idx)
	{
		for (size_t v_idx = 0; v_idx < boundary[path_idx].size(); ++v_idx)
		{
			int mapped_idx = v_map[vcg::tri::Index(mesh, boundary[path_idx][v_idx])];
			assert(mapped_idx >= 0);
			bnd[cnt] = mapped_idx;
			bnd_uv.row(cnt) = uvs[path_idx].row(v_idx);
			cnt++;
		}
	}

	// Harmonic parametrization for the internal vertices
	EMatrixXScalar V_uv;
	igl::harmonic(V, F, bnd, bnd_uv, 1, V_uv);
	V_uv = 0.5 * (1. + V_uv.array());

	cv::Mat1b img(1024, 1024, uchar(0));
	test_draw_triangles_over_texture(img, F, V_uv);
	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\harmonic_raster.png", img);
}

void test_find_decal_region(MyMesh &mesh, string decal_rect_path)
{
	vcgRect3 decal_rect;
 	import_decal_rectangle(decal_rect_path, decal_rect);

	typedef vcg::Octree<MyMesh::VertexType, double> OctreeType;
	OctreeType octree;
	octree.Set(mesh.vert.begin(), mesh.vert.end());

	vcg::tri::RequireVEAdjacency(mesh);
	vcg::tri::UpdateTopology<MyMesh>::FaceFace(mesh);
	vcg::tri::UpdateTopology<MyMesh>::AllocateEdge(mesh);
	vcg::tri::UpdateTopology<MyMesh>::VertexEdge(mesh);

	std::vector<VPointer> decal_verts(4);
	std::vector<std::vector<VPointer>> paths(4);
	find_decal_boundary(mesh, octree, decal_rect, decal_verts, paths);

	//find a seed triangle
	FPointer seed_tri = find_seed_triangle(mesh, octree, decal_rect);
	
	std::vector<FPointer> decal_trigs;
	extract_decal_triangles(mesh, paths, seed_tri, decal_trigs);
	
#if 0
	MyMesh decal_mesh;
	construct_a_mesh(mesh, decal_trigs, decal_mesh);
	string export_file_path = "D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\test_decal_mesh.obj";
	tri::io::Exporter<MyMesh>::Save(decal_mesh, export_file_path.c_str());
#endif
	parameterizre_decal(mesh, decal_trigs, paths);
}

int main(int argc, char **argv)
{
	char c;
	string  decal_img_path = "D:\\Projects\\Oh\\data\\3D\\Texture_retargeting\\decal_images\\decal.png";
	//string decal_img_path = "D:\\Projects\\Oh\\data\\3D\\Texture_retargeting\\decal_images\\front_12x16_thick_flame_top_modified.png";
	string decal_plane_file_path	= "D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\laxsquadT_mBBB_xLA4_0430_front_decal_plane.obj";
	string decal_rect_file_path = "D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\laxsquadT_mBBB_xLA4_0430_decal_rectangle.txt";
	string mesh_path				= "D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\laxsquadT_mBBB_xLA4_0430_front.obj";
	string texture_img_path			= "D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\laxsquadT_mBBB_xLA4_0430.1001_backup.jpg";
	//string texture_img_path = "D:\\Projects\\Oh\\data\\3D\\Texture_retargeting\\result\\original_texture_with_triangles.png";

	//read texture image
	cv::Size2i	size(1024 * 2, 1024 * 2);
	cv::Mat3b	tex_img = cv::imread(texture_img_path);
	const cv::MatSize &tex_size = tex_img.size;
	cv::Size2i decal_size(static_cast<int>(0.6*size.width), static_cast<int>(0.6*size.height));
	cv::Point2i  pos(static_cast<int>(0.5*(size.width - decal_size.width)), static_cast<int>(0.5*(size.height - decal_size.height)));
	cv::Rect2i decal_rect(pos.x, pos.y, decal_size.width, decal_size.height);

	cv::Mat3b  decal_img = cv::imread(decal_img_path, cv::IMREAD_COLOR);
	cv::flip(decal_img, decal_img, 1); 
	cv::rotate(decal_img, decal_img, cv::ROTATE_90_CLOCKWISE);
	cv::resize(decal_img, decal_img, decal_size, 0, 0, INTER_AREA);

	vcgRect3 decal_plane;
	if (!import_decal_mesh_plane(decal_plane_file_path, decal_plane))
	{
		printf("Error loading decal file  %s\n", decal_plane_file_path);
		cin >> c;
	}

	MyMesh decal_mesh;
	tri::io::Importer<MyMesh>::Open(decal_mesh, decal_plane_file_path.c_str());

	MyMesh mesh;
#ifdef EXTRACT_DECAL_MESH
	MyMesh org_mesh;
	auto ret = tri::io::Importer<MyMesh>::Open(org_mesh, mesh_path.c_str());
	if (ret != 0 && ret != 3 && ret != 5)
		printf("Error reading file  %s\n", argv[1]);
	vcg::tri::UpdateTopology<MyMesh>::FaceFace(org_mesh);

	std::vector<FPointer> tri_candidates;
	find_candidate_triangles(org_mesh, decal_plane, tri_candidates);

	construct_a_mesh(org_mesh, tri_candidates, mesh);

	//string export_file_path = "D:\\Projects\\Oh\\data\\3D\\Texture_retargeting\\test_export_decal_tris.obj";
	//tri::io::Exporter<MyMesh>::Save(mesh, export_file_path.c_str());
#else
	tri::io::Importer<MyMesh>::Open(mesh, mesh_path.c_str());
#endif
	test_find_decal_region(mesh, decal_rect_file_path);
	//test_harmonic_parameterize(mesh);
	exit(1);

	vcgPoint3 center, axis[3];
	calc_decal_space(decal_mesh, center, axis);

	std::vector<LocalDecalTriangle> local_decal_trigs;
	transform_mesh_to_decal_space(center, axis, mesh, local_decal_trigs);

#define DRAW_MAPPING 1
#ifdef DRAW_MAPPING
	cv::Mat1b tex_img_triangle = cv::Mat1b(tex_img.size[0], tex_img.size[1]);
	test_draw_triangles_over_texture(tex_img_triangle, local_decal_trigs);
	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\triangle_texture_space.png", tex_img_triangle);

	cv::Mat3b our_masterpiece_1 = test_draw_triangles(size, local_decal_trigs);
	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\rasterization_triangles.png", our_masterpiece_1);

	exit(1);
#endif

	cv::Mat1f  depth;
	cv::Mat2f  tex_coords;
	rasterize_triangles(local_decal_trigs, size, depth, tex_coords);

//#define DEBUG_TEXTURE_COORDS 1
#ifdef DEBUG_TEXTURE_COORDS 
	cv::Mat1b debug_tex_coords(tex_coords.size[0], tex_coords.size[1], uchar(0));
	for (int i = 0; i < tex_coords.size[0]; ++i)
	{
		for (int j = 0; j < tex_coords.size[1]; ++j)
		{
			double scale = (tex_coords(i, j)[0] + tex_coords(i, j)[1]) / 2.0f;
			debug_tex_coords(i, j) = (uchar)(scale * 255);
		}
	}
	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\texture_coords.png", debug_tex_coords);

	cv::Mat3b textured_ras_img;
	textured_ras_img = output_textured_rasterization(size, tex_img, decal_img, decal_rect, tex_coords);

	cv::Vec3b bg_color = find_background_color(textured_ras_img);
	int variance = 3;
	cv::Mat3b bg_img = generate_background_image(size, bg_color, cv::Vec3b(variance, variance, variance));

	//textured_ras_img = test_draw_triangles_over_image(textured_ras_img, local_decal_trigs);
	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\rasterize.png", textured_ras_img);
	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\make_up_bacground_image.png", bg_img);
#endif

	cv::Mat3b mod_tex_img = tex_img.clone();
	cv::Mat1b mask_tex_img = cv::Mat1b(mod_tex_img.size[0], mod_tex_img.size[1]);
	for (int i = 0; i < decal_rect.width; ++i)
	{
		int px = i + decal_rect.x;
		for (int j = 0; j < decal_rect.height; ++j)
		{
			int py = j + decal_rect.y;
			cvVec2 co = tex_coords(px, py);
			co[1] = 1.0 - co[1];
			int tex_ix = int((double)tex_size[0] * co[1]);
			int tex_iy = int((double)tex_size[1] * co[0]);
			if (tex_ix >= 0 && tex_ix < tex_size[0] && tex_iy >= 0 && tex_iy < tex_size[1])
			{
				cv::Vec3f decal_pix = decal_img(i,j);
				const double threshold = 0.0;
				if (true/*decal_pix[0] > threshold && decal_pix[1] > threshold && decal_pix[2] > threshold*/)
				{
					cv::Vec3f tex_pix = tex_img(tex_ix, tex_iy);
					//tex_pix = 0.5f * tex_pix + 0.5f*decal_pix;
					//mod_tex_img(tex_ix, tex_iy) = decal_pix;
					mod_tex_img(tex_ix, tex_iy) = cv::Vec3b(0, 0, 255);
					mask_tex_img(tex_ix, tex_iy) = 255;
				}
			}
		}
	}
	//for printing
	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\laxsquadT_mBBB_xLA4_0430.1001.jpg", mod_tex_img);
	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\texture_mask.png", mask_tex_img);

	return 0;
}
