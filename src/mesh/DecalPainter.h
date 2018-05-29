#ifndef DECAL_PAINTER_H
#define DECAL_PAINTER_H
#include<vcg/complex/complex.h>
#include<wrap/io_trimesh/import.h>
#include<wrap/io_trimesh/export.h>
#include<vcg/space/index/kdtree/kdtree.h>
#include<vcg/space/index/aabb_binary_tree/aabb_binary_tree.h>
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
#include <opencv2/photo.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Dense>

using namespace vcg;
using namespace std;
using namespace cv;

class DecalPainter
{
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
	class MyMesh : public tri::TriMesh< VertexContainer, FaceContainer, EdgeContainer> {};

	typedef MyMesh::VertexPointer	VPointer;
	typedef MyMesh::FacePointer		FPointer;
	typedef MyMesh::EdgePointer		EPointer;
	typedef vcg::Octree<MyMesh::VertexType, double> OctreeType;
private:
	vcgRect3	m_decal_anchor_corners;
	MyMesh		m_mesh;
	cv::Mat3b	m_decal_img;
	cv::Mat3b	m_tex_img;
public:
	void paint_decal();
	bool set_mesh(std::string path);
	bool set_mesh_texture(std::string path);
	void set_decal_anchor_points(std::array<vcgPoint3, 4> points);
	bool set_decal_anchor_points(std::string path);
	bool set_decal_image(std::string path);
	bool is_file_exist(string path);
private:
	void mesh_matrix(MyMesh &mesh, EMatrixXScalar &V, EMatrixX &F);
	void mesh_matrix(MyMesh &mesh, const std::vector<FPointer> &trigs, EMatrixXScalar &V, EMatrixX &F, EVectorX &vmap);
	double calc_path_len(const std::vector<VPointer> &path);
	void construct_uv_rect_boundary(
		const std::vector<EVector2Scalar> &corners,
		const std::vector<std::vector<VPointer>> &paths,
		std::vector<EMatrixXScalar> &uvpaths);

	void parameterize_mesh_to_rectangular_domain(const EMatrixXScalar &V, const EMatrixX &F, EMatrixXScalar &V_uv);
	CvRect triangle_rect(const cvPoint2 coords[3]);
	bool inside_triangle(const cvPoint2 trig_coords[3], const cvPoint2 &p, double &L1, double &L2, double &L3, const double EPSILON);
	bool import_decal_rectangle(std::string file_path, vcgRect3 &decal_rect);
	VPointer edge_other_vert(MyMesh::EdgePointer e, VPointer v);
	EPointer edge_from_verts(VPointer v0, VPointer v1);
	bool find_geodesic_path(MyMesh &mesh, VPointer vstart, VPointer vend, std::vector<VPointer> &path, int max_path_len = 10000);
	bool find_decal_boundary(MyMesh &mesh, OctreeType &octree, vcgRect3 &decal_rect, std::vector<VPointer> &decal_verts, std::vector<std::vector<VPointer>> &paths);
	void merge_path(std::vector<std::vector<VPointer>> &paths, std::vector<VPointer> &path);
	bool extract_decal_triangles(MyMesh &mesh, std::vector<std::vector<VPointer>> &boundaries, FPointer seed_trig, std::vector<FPointer> &decal_trigs);
	bool is_tri_candiate(FPointer tri, const vcg::Plane3d &plane, const vcg::Box3d &box, const double &furthest_dist);
	void construct_a_mesh(MyMesh &mesh, const std::vector<FPointer> &tris, MyMesh &new_mesh);
	FPointer find_seed_triangle(MyMesh &mesh, OctreeType &octree, vcgRect3 &decal_rect);
	void triangle_texture_coords(FPointer trig, cvVec2 tex_cos[3]);
	void generate_texture_coordinates(const std::vector<FPointer> &trigs, const EMatrixX &F, const EMatrixXScalar &V_uv, const cv::Size2i &img_size, cv::Mat2f &tex_coords);

	void draw_texture_triangle_over_img(cv::Mat3b &tex, const EMatrixX &trigs, const EMatrixXScalar &V_uv, Scalar color);
	void draw_texture_triangle_over_img(cv::Mat &tex, const std::vector<FPointer> &trigs, Scalar color);
	void test_draw_segments(cv::Mat3b &mat, const std::vector<EMatrixXScalar> &paths);
	cv::Mat3b generate_background_image(cv::Size size, cv::Vec3b mean_color, cv::Vec3b variance);
	cv::Vec3b find_background_color(const cv::Mat3b &img);
	cv::Mat3b build_textured_rasterization(const cv::Size &size, const cv::Mat3b &tex_img, const cv::Mat3b &decal_img, cv::Rect2i decal_rect, const cv::Mat2f  &tex_coords);
	cv::Mat3b output_textured_rasterization(const cv::Size &size, const cv::Mat3b &tex_img, const cv::Mat3b &decal_img, cv::Rect2i decal_rect, const cv::Mat2f  &tex_coords);
	size_t total_vertices(const std::vector<std::vector<VPointer>> &boundary);
	void debug_mesh_points(MyMesh &mesh, std::vector<VPointer> &points);
	void debug_triangle_boundary(MyMesh &mesh, std::vector<FPointer> &decal_trigs, std::vector<std::vector<VPointer>> &boundary, EVectorX &vmap);
	void parameterizre_decal_rect(MyMesh &mesh,
		const std::vector<FPointer> &decal_trigs,
		const std::vector<std::vector<VPointer>> &boundary,
		EMatrixX &F, EMatrixXScalar &V_uv);
	bool find_decal_area(MyMesh &mesh, vcgRect3 decal_rect,
		std::vector<FPointer> &decal_trigs, std::vector<std::vector<VPointer>> &boundary);
	cv::Point select_seed_point_texture_space(const cv::Mat1b &mapping);
};
#endif