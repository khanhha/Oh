#ifndef DECAL_PAINTER_H
#define DECAL_PAINTER_H
#include<vcg/complex/complex.h>
#include<wrap/io_trimesh/import.h>
#include<wrap/io_trimesh/export.h>
#include<vcg/complex/complex.h>
#include <vcg/space/point3.h>
#include <vcg/space/index/octree.h>

#include <igl/harmonic.h>
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
	cv::Mat1f	m_decal_img_alpha;
	cv::Mat3b	m_tex_img;
	cv::Size	m_mapping_size; //the resolution of 2D mapping of decal area. The higher the value is, the less tiny gaps in the mapping there are
	double		m_paint_percent; //how many percent of m_mapping_size that the decal image takes up
public:
	enum ERROR
	{
		NO_ERROR = 0,
		CANNOT_FIND_A_CLOSED_BOUNDARY = 1 << 1,
		CANNOT_FIND_A_SEED_TRIANGLE_INSIDE_CLOSED_BDR = 1 << 2,
		INVALID_DATA = 1 << 3
	};
public:
	DecalPainter();
	std::string error_string(int error);

	bool check_valid_data();
	int  paint_decal(cv::Mat3b &painted_texture, float brightness_mult = -1.0, float decal_smooth_sigma = 0.0);
	int  erase_decal(cv::Mat3b &erased_texture, cv::Rect2d bgr_rect = cv::Rect2d(0.1, 0.1, 0.1, 0.1), float brightness_mult = -1.0);
	int  erase_paint_decal(cv::Mat3b &modified_texture, cv::Rect2d bgr_rect = cv::Rect2d(0.1, 0.1, 0.1, 0.1), float brightness_mult = -1.0, float decal_smooth_sigma = 0.0);
	bool set_mesh(std::string path);
	bool set_mesh_texture(std::string path);
	void set_decal_anchor_corners(std::array<vcgPoint3, 4> points);
	bool set_decal_anchor_corners(std::string path);
	bool set_decal_image(std::string path);
	bool is_file_exist(string path);
	void set_mapping_size(cv::Size size, double paint_size = 1.0);
	bool import_decal_rectangle(std::string file_path, vcgRect3 &decal_rect);
private:
	void preprocess_decal_img(cv::Size2i size, float decal_smooth_sigma = 0.0);
	void mesh_matrix(MyMesh &mesh, EMatrixXScalar &V, EMatrixX &F);
	void mesh_matrix(MyMesh &mesh, const std::vector<FPointer> &trigs, EMatrixXScalar &V, EMatrixX &F, EVectorX &vmap);
	CvRect triangle_bounding_rect(const cvPoint2 coords[3]);
	bool is_inside_triangle(const cvPoint2 trig_coords[3], const cvPoint2 &p, double &L1, double &L2, double &L3, const double EPSILON);

	double calc_path_len(const std::vector<VPointer> &path);
	VPointer edge_other_vert(MyMesh::EdgePointer e, VPointer v);
	EPointer edge_from_verts(VPointer v0, VPointer v1);
	bool find_geodesic_path(MyMesh &mesh, VPointer vstart, VPointer vend, std::vector<VPointer> &path, int max_path_len = 10000);
	bool find_decal_3D_boundary(MyMesh &mesh, OctreeType &octree, vcgRect3 &decal_rect, std::vector<VPointer> &decal_verts, std::vector<std::vector<VPointer>> &paths);
	void collect_decal_triangles_inside_boundary(MyMesh &mesh, std::vector<std::vector<VPointer>> &boundaries, FPointer seed_trig, std::vector<FPointer> &decal_trigs);
	size_t total_vertices(const std::vector<std::vector<VPointer>> &boundary);
	void merge_path(std::vector<std::vector<VPointer>> &paths, std::vector<VPointer> &path);
	FPointer find_seed_triangle(MyMesh &mesh, OctreeType &octree, vcgRect3 &decal_rect);
	int find_3D_mesh_decal_area(MyMesh &mesh, vcgRect3 decal_rect,
		std::vector<FPointer> &decal_trigs, std::vector<std::vector<VPointer>> &boundary);

	void construct_uv_rect_boundary(
		const std::vector<EVector2Scalar> &corners,
		const std::vector<std::vector<VPointer>> &paths,
		std::vector<EMatrixXScalar> &uvpaths);

	int generate_mapping_texture_space(cv::Size tex_size, std::vector<FPointer> &decal_trigs, cv::Mat2f &tex_coords_map, cv::Mat1b &map_mask);
	void triangle_texture_coords(FPointer trig, cvVec2 tex_cos[3]);
	void generate_texture_coordinates_texture_space(const std::vector<FPointer> &trigs, const EMatrixX &F, const EMatrixXScalar &V_uv, const cv::Size2i &img_size, 
		cv::Mat2f &tex_coords, cv::Mat1b &tex_coords_mask);
	void fix_island_boundary_gaps(cv::Mat2f &tex_coords_mapping, cv::Mat1b &tex_coords_mask);
	
	cv::Mat3b blend_decal_with_texture(const cv::Mat3b &tex_img, const cv::Mat2f &mapping_tex_coords, const cv::Mat1b &mapping_mask,
		const cv::Mat3b &decal_img, const cv::Mat1f &decal_img_alpha);

	void generate_decal_in_tex_space_mask(cv::Mat &tex, const std::vector<FPointer> &trigs, Scalar color);
	void draw_decal_trigs_in_tex_space_mask(cv::Mat &tex, const std::vector<FPointer> &trigs, Scalar color);
	cv::Mat3b fill_decal_image_with_texture_colors(const cv::Size &decal_size, const cv::Mat3b &tex_img, const cv::Mat2f  &tex_coords, const cv::Mat1b tex_coords_mask);
	float estimate_brightness_multiplifer(cv::Mat3b textured_decal_mapping) const;
#if _DEBUG
	void construct_a_mesh(MyMesh &mesh, const std::vector<FPointer> &tris, MyMesh &new_mesh);
	void test_draw_segments(cv::Mat3b &mat, const std::vector<EMatrixXScalar> &paths);
	void debug_mesh_points(MyMesh &mesh, std::vector<VPointer> &points);
	void debug_triangle_boundary(MyMesh &mesh, std::vector<FPointer> &decal_trigs, std::vector<std::vector<VPointer>> &boundary, EVectorX &vmap);
#endif
	void parameterizre_decal_rect(MyMesh &mesh,
		const std::vector<FPointer> &decal_trigs,
		const std::vector<std::vector<VPointer>> &boundary,
		EMatrixX &F, EMatrixXScalar &V_uv);

};
#endif