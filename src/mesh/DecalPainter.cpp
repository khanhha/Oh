
#include "DecalPainter.h"
#include <algorithm>
DecalPainter::DecalPainter()
	: m_mapping_size(1024, 1024)
	, m_paint_percent(1.0)
{}

std::string DecalPainter::error_string(int error)
{
	string error_str;
	if (error & CANNOT_FIND_A_CLOSED_BOUNDARY)
		error_str += "CANNOT_FIND_A_CLOSED_BOUNDARY.";

	if (error & CANNOT_FIND_A_SEED_TRIANGLE_INSIDE_CLOSED_BDR)
		error_str += "CANNOT_FIND_A_SEED_TRIANGLE_INSIDE_CLOSED_BDR";

	if (error & INVALID_DATA)
		error_str += "INVALID_DATA";
	
	return  error_str;
}

bool DecalPainter::check_valid_data()
{
	if (m_decal_img.empty())
	{
		std::cerr << "error empty decal image" << std::endl;
		return false;
	}

	if (m_tex_img.empty())
	{
		std::cerr << "error empty texture image" << std::endl;
		return false;
	}

	if (m_mesh.IsEmpty())
	{
		std::cerr << "error empty  mesh" << std::endl;
		return false;
	}

	return true;
}

CvRect DecalPainter::triangle_bounding_rect(const cvPoint2 coords[3])
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

	return CvRect(int(xmin), int(ymin), int(xmax - xmin + 1), int(ymax - ymin + 1));
}

bool DecalPainter::is_inside_triangle(const cvPoint2 trig_coords[3], const cvPoint2 &p, double &L1, double &L2, double &L3, const double EPSILON)
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

void DecalPainter::mesh_matrix(MyMesh &mesh, EMatrixXScalar &V, EMatrixX &F)
{
	//collect all boundary vertices
	V.resize(mesh.VN(), 3);
	F.resize(mesh.FN(), 3);
	for (auto vit = mesh.vert.begin(); vit != mesh.vert.end(); ++vit)
	{
		auto idx = vcg::tri::Index(mesh, *vit);
		const vcgPoint3 &co = vit->cP();
		V(idx, 0) = co[0];
		V(idx, 1) = co[1];
		V(idx, 2) = co[2];
	}

	for (auto fit = mesh.face.begin(); fit != mesh.face.end(); ++fit)
	{
		auto idx = vcg::tri::Index(mesh, *fit);
		for (int i = 0; i < 3; ++i)
			F(idx, i) = (int)vcg::tri::Index(mesh, fit->V(i));
	}
}

void DecalPainter::mesh_matrix(MyMesh &mesh, const std::vector<FPointer> &trigs, EMatrixXScalar &V, EMatrixX &F, EVectorX &vmap)
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
			size_t idx = vcg::tri::Index(mesh, t->V(i));
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

double DecalPainter::calc_path_len(const std::vector<VPointer> &path)
{
	double len = 0;
	for (int i = 1; i < path.size(); ++i)
		len += (path[i]->cP() - path[i - 1]->cP()).Norm();
	return len;
}

DecalPainter::VPointer DecalPainter::edge_other_vert(MyMesh::EdgePointer e, VPointer v)
{
	return (e->V(0) == v) ? e->V(1) : e->V(0);
}

/*find the edge that connect two input vertices*/
DecalPainter::EPointer DecalPainter::edge_from_verts(VPointer v0, VPointer v1)
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

/*
	use breath first search to find the closet path from vstart to vend
	@path: the vertices along path, not includes vend.
	@max_path_len: stop searching if the number of path is larger than max_path_len
*/
bool DecalPainter::find_geodesic_path(MyMesh &mesh, VPointer vstart, VPointer vend, std::vector<VPointer> &path, int max_path_len /*= 10000*/)
{
	std::vector<int> path_trace(mesh.VN(), -1);
	std::vector<double> short_distances(mesh.VN(), -1.);

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

		size_t v_idx = vcg::tri::Index(mesh, v);

		double cur_dst = short_distances[v_idx];
		edge::VEIterator<MyMesh::EdgeType> ve_iter(v);
		while (!ve_iter.End())
		{
			VPointer other_v = edge_other_vert(ve_iter.E(), v);
			size_t other_v_idx = vcg::tri::Index(mesh, other_v);
			double e_len = (other_v->cP() - v->cP()).Norm();
			bool is_visited = (other_v->cFlags() & MyMesh::VertexType::VISITED) != 0;
			if (is_visited)
			{
				if (short_distances[other_v_idx] > cur_dst + e_len)
				{
					short_distances[other_v_idx] = cur_dst + e_len;
					path_trace[other_v_idx] = (int)v_idx;
				}
			}
			else
			{
				other_v->Flags() |= MyMesh::VertexType::VISITED;

				path_trace[other_v_idx] = (int)v_idx;
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
			break;

		cur_path_len++;

		if (cur_path_len > max_path_len)
			break;
	}

	if (reach_end)
	{
		int trace_idx = (int)vcg::tri::Index(mesh, vend);

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
		path.clear();
		return false;
	}
}

/*find a closed mesh boundary of decal area
@decal_rect: four reference corners of decal on mesh. Each corner should be as close as possible to a vertex on mesh
@decal_verts: four mesh vertices correspond to four corners of decal_rect.
@paths: four vertex paths along each edge of decal rectangle.
*/
bool DecalPainter::find_decal_3D_boundary(MyMesh &mesh, OctreeType &octree, vcgRect3 &decal_rect, std::vector<VPointer> &decal_verts, std::vector<std::vector<VPointer>> &paths)
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

	bool ret[4];
	for (size_t i = 0; i < 4; ++i)
	{
		VPointer vstart = decal_verts[i];
		VPointer vend = decal_verts[(i + 1) % 4];
		ret[i] = find_geodesic_path(mesh, vstart, vend, paths[i], 100000);
	}

	return (ret[0] && ret[1] && ret[2] && ret[3]);
}

//************************************
// Method:    collect_decal_triangles_inside_boundary: find all decal triangle inside a closed boundary
// FullName:  DecalPainter::collect_decal_triangles_inside_boundary
// Access:    private 
// Returns:   bool
// Qualifier:
// Parameter: MyMesh & mesh
// Parameter: std::vector<std::vector<VPointer>> & boundaries: four segments that define a closed boundary
// Parameter: FPointer seed_trig: a seed triangle inside the closed boundary to start from
// Parameter: std::vector<FPointer> & decal_trigs: list of all triangles inside the closed boundary
//************************************
void DecalPainter::collect_decal_triangles_inside_boundary(MyMesh &mesh, std::vector<std::vector<VPointer>> &boundaries, FPointer seed_trig, std::vector<FPointer> &decal_trigs)
{
	auto is_boundary_edge = [](EPointer e) -> bool
	{
		return (e->Flags() & MyMesh::EdgeType::VISITED) != 0;
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
		VPointer v1 = closed_bdr[(i + 1) % npoints];
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
			EPointer e = face->FEp(i);
			bool is_visited = (fadj->cFlags() & MyMesh::FaceType::VISITED) != 0;
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
}

size_t DecalPainter::total_vertices(const std::vector<std::vector<VPointer>> &boundary)
{
	size_t cnt = 0;
	for (int i = 0; i < boundary.size(); ++i)
		cnt += boundary[i].size();
	return cnt;
}

void DecalPainter::merge_path(std::vector<std::vector<VPointer>> &paths, std::vector<VPointer> &path)
{
	for (auto &p : paths)
		path.insert(path.end(), p.begin(), p.end());
}

/*
find the closest triangle to the center of decal_rect on the mesh. 
when thing could go wrong:  the closest vertex we found are isolated. it has no adjacent triangles
*/
DecalPainter::FPointer DecalPainter::find_seed_triangle(MyMesh &mesh, OctreeType &octree, vcgRect3 &decal_rect)
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

	/*TODO: go through all triangles of the mesh to find the closest one*/
	
	return nullptr;
}

/*find a closed boundary of decal area and all the triangles inside the boundary
@decal_rect: 4 reference corners of decal area on mesh
*/
int DecalPainter::find_3D_mesh_decal_area(MyMesh &mesh, vcgRect3 decal_rect, std::vector<FPointer> &decal_trigs, std::vector<std::vector<VPointer>> &boundary)
{
	int err = DecalPainter::NO_ERROR;

	typedef vcg::Octree<MyMesh::VertexType, double> OctreeType;
	OctreeType octree;
	octree.Set(mesh.vert.begin(), mesh.vert.end());

	vcg::tri::RequireVEAdjacency(mesh);
	vcg::tri::UpdateTopology<MyMesh>::FaceFace(mesh);
	vcg::tri::UpdateTopology<MyMesh>::AllocateEdge(mesh);
	vcg::tri::UpdateTopology<MyMesh>::VertexEdge(mesh);

	std::vector<VPointer> decal_verts(4);
	boundary.resize(4);
	if (!find_decal_3D_boundary(mesh, octree, decal_rect, decal_verts, boundary))
	{
		err |= CANNOT_FIND_A_CLOSED_BOUNDARY;
		return err;
	}

	//find a seed triangle
	FPointer seed_tri = find_seed_triangle(mesh, octree, decal_rect);
	if (!seed_tri)
	{
		err |= CANNOT_FIND_A_SEED_TRIANGLE_INSIDE_CLOSED_BDR;
		return err;
	}

	collect_decal_triangles_inside_boundary(mesh, boundary, seed_tri, decal_trigs);

#if 0
	MyMesh decal_mesh;
	construct_a_mesh(mesh, decal_trigs, decal_mesh);
	string export_file_path = "D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\test_decal_mesh.obj";
	tri::io::Exporter<MyMesh>::Save(decal_mesh, export_file_path.c_str());
#endif
	return err;
}

/*generate UV rectangular coordinates for the closed boundary
@corners: 4 vertex corners of the rectangular boundary
@paths: 4 segments of the closed boundary. 
@uvpaths: UV coordinates correspond to each vertex on the boundary
*/
void DecalPainter::construct_uv_rect_boundary(const std::vector<EVector2Scalar> &corners, const std::vector<std::vector<VPointer>> &paths, std::vector<EMatrixXScalar> &uvpaths)
{
	assert(corners.size() == 4);
	assert(paths.size() == 4);
	assert(uvpaths.size() == 4);
	for (int i = 0; i < 4; ++i) {
		assert(!paths[i].empty());
	}

	for (int i = 0; i < 4; ++i)
	{
		const std::vector<VPointer> &path = paths[i];
		size_t  n_path_verts = path.size();
		double	path_len = calc_path_len(path);

		EVector2Scalar uv_start = corners[i], uv_end = corners[(i + 1) % 4];
		EVector2Scalar	edge = uv_end - uv_start;

		EMatrixXScalar &uvpath = uvpaths[i];
		uvpath.resize(n_path_verts, 2);
		uvpath.row(0) = uv_start;

		double went_so_far = 0.;
		for (int j = 1; j < n_path_verts; ++j)
		{
			went_so_far += (path[j]->cP() - path[j - 1]->cP()).Norm();
			uvpath.row(j) = uv_start + (went_so_far / path_len) * edge;
		}
	}
}

/*collect texture coordinate of the triagnle*/
void DecalPainter::triangle_texture_coords(FPointer trig, cvVec2 tex_cos[3])
{
	for (int i = 0; i < 3; ++i) {
		const auto &co = trig->WT(i);
		tex_cos[i][0] = co.U();
		tex_cos[i][1] = co.V();
	}
}

void DecalPainter::generate_texture_coordinates_texture_space(const std::vector<FPointer> &mesh_trigs,
	const EMatrixX &F, const EMatrixXScalar &V_uv, 
	const cv::Size2i &tex_size, cv::Mat2f &tex_coords, cv::Mat1b &tex_coords_mask)
{
	tex_coords = cv::Mat2f(tex_size, cv::Vec2f(0,0));
	tex_coords_mask = cv::Mat1b(tex_size, uchar(0));
	cvPoint2 tex_trig_verts[3];
	cvVec2   decal_tri_tex_coords[3];

	for (auto i = 0; i < F.rows(); ++i)
	{
		for (int vi = 0; vi < 3; ++vi)
			decal_tri_tex_coords[vi] = cvVec2(V_uv(F(i, vi), 0), V_uv(F(i, vi), 1));

		for (int vi = 0; vi < 3; ++vi) {
			const auto &co = mesh_trigs[i]->WT(vi);
			tex_trig_verts[vi].x = int(co.U() * (double)tex_size.width);
			tex_trig_verts[vi].y = int((1.0 - co.V())* (double)tex_size.height);
		}

		CvRect tex_rect = triangle_bounding_rect(tex_trig_verts);
		double L1, L2, L3;
		for (int ix = 0; ix < tex_rect.width; ix++)
		{
			int px = tex_rect.x + ix;
			for (int iy = 0; iy < tex_rect.height; iy++)
			{
				int py = tex_rect.y + iy;
				if (is_inside_triangle(tex_trig_verts, cvPoint2((double)px, (double)py), L1, L2, L3, 0.01))
				{
					cvVec2 tex = L1*decal_tri_tex_coords[0] + L2*decal_tri_tex_coords[1] + L3*decal_tri_tex_coords[2];
					tex[0] = tex[0] > 1.0 ? 1.0 : (tex[0] < 0.0 ? 0.0 : tex[0]);
					tex[1] = tex[1] > 1.0 ? 1.0 : (tex[1] < 0.0 ? 0.0 : tex[1]);

					tex_coords(py, px) = cv::Vec2f((float)tex[0], (float)tex[1]) ;
					tex_coords_mask(py, px) = 255;
				}
			}
		}
	}
}

void DecalPainter::draw_decal_trigs_in_tex_space_mask(cv::Mat &tex, const std::vector<FPointer> &trigs, Scalar color)
{
	cvVec2	tri_coords[3];
	cv::Point tri_points[3];
	int width = tex.size[0];
	int height = tex.size[1];

	for (auto t : trigs)
	{
		triangle_texture_coords(t, tri_coords);
		for (auto i = 0; i < 3; ++i)
		{
			tri_points[i].x = int(tri_coords[i][0] * width);
			tri_points[i].y = int((1.0 - tri_coords[i][1])* height);
		}

		for (auto i = 0; i <3; ++i)
			cv::line(tex, tri_points[i], tri_points[(i + 1) % 3], color);
	}
}

void DecalPainter::generate_decal_in_tex_space_mask(cv::Mat &tex, const std::vector<FPointer> &trigs, Scalar color)
{
	cvVec2	tri_coords[3];
	cv::Point tri_points[3];
	int width = tex.size[0];
	int height = tex.size[1];

	for (auto t : trigs)
	{
		triangle_texture_coords(t, tri_coords);
		for (auto i = 0; i < 3; ++i)
		{
			tri_points[i].x = int(tri_coords[i][0] * width);
			tri_points[i].y = int((1.0 - tri_coords[i][1])* height);
		}

		cv::fillConvexPoly(tex, tri_points, 3, color, cv::LINE_8);
	}
}

void DecalPainter::fix_island_boundary_gaps(cv::Mat2f &tex_coords_mapping, cv::Mat1b &tex_coords_mask)
{
	cv::Mat strel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4));
	cv::dilate(tex_coords_mapping, tex_coords_mapping, strel);
	cv::dilate(tex_coords_mask, tex_coords_mask, strel);
}

cv::Mat3b DecalPainter::blend_decal_with_texture(const cv::Mat3b &tex_img, const cv::Mat2f &mapping_tex_coords, const cv::Mat1b &mapping_mask,
	const cv::Mat3b &decal_img, const cv::Mat1f &decal_img_alpha)
{
	cv::Mat3b blended_tex_img = tex_img.clone();

	int decal_rows_max = decal_img.rows-1;
	int decal_cols_max = decal_img.cols-1;
	for (int ir = 0; ir < tex_img.rows; ir++)
	{
		for (int ic = 0; ic < tex_img.cols; ic++)
		{
			if (mapping_mask(ir, ic) > 0)
			{
				const cv::Vec2f &norm_tex_co = mapping_tex_coords(ir, ic);
				const cv::Point tex_co(decal_rows_max*norm_tex_co[0], decal_cols_max*norm_tex_co[1]);
				const float decal_alpha = decal_img_alpha(tex_co);
				if (decal_alpha > 0.0f)
				{
					const cv::Vec3b &decal_pix = decal_img(tex_co);
					blended_tex_img(ir, ic) = (decal_alpha * decal_pix + (1.0f - decal_alpha)*tex_img(ir,ic));
				}
			}
		}
	}

	return blended_tex_img;
}

/*decal image is often too bright, so we want to estimate a brightness scale to make it more natural*/
/*brightness scale is calculated as the average intensity of pixel colors inside decal texture area*/
float DecalPainter::estimate_brightness_multiplifer(cv::Mat3b textured_decal_mapping) const
{
	cv::Mat1b gray_img;
	cv::cvtColor(textured_decal_mapping, gray_img, cv::COLOR_BGR2GRAY);
	Scalar avg_bright = cv::mean(gray_img);
	float avg_bright_val = 2.0*float(avg_bright[0] / 255.0);
	return avg_bright_val;
}

cv::Mat3b DecalPainter::fill_decal_image_with_texture_colors(const cv::Size &decal_size, 
	const cv::Mat3b &tex_img, 
	const cv::Mat2f  &tex_coords, const cv::Mat1b tex_coords_mask)
{
	cv::Mat3b textured_decal_img = cv::Mat3b(decal_size);
	int rows_max = decal_size.height - 1;
	int cols_max = decal_size.width - 1;
	for (int ir = 0; ir < tex_img.rows; ++ir)
	{
		for (int ic = 0; ic < tex_img.cols; ++ic)
		{
			if (tex_coords_mask(ir, ic) > 0)
			{
				const cv::Vec2f &norm_tex_co = tex_coords.at<cv::Vec2f>(ir, ic);
				const cv::Point tex_co(norm_tex_co[0] * rows_max, norm_tex_co[1] * cols_max);
				textured_decal_img(tex_co) = tex_img.at<cv::Vec3b>(ir, ic);
			}
		}
	}
	return textured_decal_img;
}

#ifdef _DEBUG

void DecalPainter::construct_a_mesh(MyMesh &mesh, const std::vector<FPointer> &tris, MyMesh &new_mesh)
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

	vcg::tri::Append<MyMesh, MyMesh>::Selected(new_mesh, mesh);

	for (const FPointer &tri : tris)
	{
		for (auto i = 0; i < 3; ++i)
		{
			tri->V(i)->ClearS();
			tri->ClearS();
		}
	}
}

void DecalPainter::test_draw_segments(cv::Mat3b &mat, const std::vector<EMatrixXScalar> &paths)
{
	int width = mat.cols;
	int height = mat.rows;

	for (const EMatrixXScalar &path : paths)
	{
		size_t npoints = path.rows();
		for (int i = 0; i < npoints - 1; ++i)
		{
			cv::Point p0(path(i, 0) * width, path(i, 1) * height);
			cv::Point p1(path(i + 1, 0) * width, path(i + 1, 1) * height);
			cv::line(mat, p0, p1, Scalar(255, 255, 255), 1);
		}
	}
}




void DecalPainter::debug_mesh_points(MyMesh &mesh, std::vector<VPointer> &points)
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

void DecalPainter::debug_triangle_boundary(MyMesh &mesh, std::vector<FPointer> &decal_trigs, std::vector<std::vector<VPointer>> &boundary, EVectorX &vmap)
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
#endif

void DecalPainter::parameterizre_decal_rect(MyMesh &mesh, const std::vector<FPointer> &decal_trigs, const std::vector<std::vector<VPointer>> &boundary, EMatrixX &F, EMatrixXScalar &V_uv)
{
	std::vector<EVector2Scalar> corners(4);
	corners[0] = EVector2Scalar(1.0, 1.0);
	corners[1] = EVector2Scalar(-1.0, 1.0);
	corners[2] = EVector2Scalar(-1.0, -1.0);
	corners[3] = EVector2Scalar(1.0, -1.0);

	std::vector<EMatrixXScalar> uvs(4);
	construct_uv_rect_boundary(corners, boundary, uvs);

	EMatrixXScalar V;
	EVectorX v_map;
	mesh_matrix(mesh, decal_trigs, V, F, v_map);

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
	igl::harmonic(V, F, bnd, bnd_uv, 1, V_uv);
	V_uv = 0.5 * (1. + V_uv.array());
}

void DecalPainter::preprocess_decal_img(cv::Size2i size, float smooth_sigma/*= 0.0*/)
{
	if (smooth_sigma > 0.0)
	{
		cv::GaussianBlur(m_decal_img, m_decal_img, cv::Size(0, 0), smooth_sigma, smooth_sigma);
		cv::GaussianBlur(m_decal_img_alpha, m_decal_img_alpha, cv::Size(0, 0), smooth_sigma, smooth_sigma);
	}
	cv::resize(m_decal_img, m_decal_img, size, 0, 0, INTER_AREA);
	cv::flip(m_decal_img, m_decal_img, 0);

	cv::resize(m_decal_img_alpha, m_decal_img_alpha, size, 0, 0, cv::INTER_NEAREST);
	cv::flip(m_decal_img_alpha, m_decal_img_alpha, 0);
}

int DecalPainter::generate_mapping_texture_space(cv::Size tex_size, std::vector<FPointer> &decal_trigs, cv::Mat2f &tex_coords_map, cv::Mat1b &map_mask)
{
	std::vector<std::vector<VPointer>> rect_boundary;
	int err = find_3D_mesh_decal_area(m_mesh, m_decal_anchor_corners, decal_trigs, rect_boundary);
	if (err != NO_ERROR)
		return err;

	EMatrixX F;
	EMatrixXScalar V_uv;
	parameterizre_decal_rect(m_mesh, decal_trigs, rect_boundary, F, V_uv);

	generate_texture_coordinates_texture_space(decal_trigs, F, V_uv, tex_size, tex_coords_map, map_mask);

	return NO_ERROR;
}

/*
replace all the decal area by an upscaled version of bgr_rect in the decal image
@bgr_rect: the rectangle where pixels are of an empty t-shirt*/
int DecalPainter::erase_decal(cv::Mat3b &erased_texture, cv::Rect2d bgr_rect, float brightness_mult)
{
	if (!check_valid_data())
		return INVALID_DATA;

	cv::Size m_tex_size = cv::Size(m_tex_img.cols, m_tex_img.rows);
	std::vector<FPointer> decal_trigs;
	cv::Mat2f  mapping_tex_coords;
	cv::Mat1b  mapping_mask;
	generate_mapping_texture_space(m_tex_size, decal_trigs, mapping_tex_coords, mapping_mask);
	fix_island_boundary_gaps(mapping_tex_coords, mapping_mask);

	cv::Mat3b textured_decal_img = fill_decal_image_with_texture_colors(m_mapping_size, m_tex_img, mapping_tex_coords, mapping_mask);

	cv::Rect bgr_roi(int(bgr_rect.x*m_mapping_size.width), int(bgr_rect.y*m_mapping_size.height), int(bgr_rect.width*m_mapping_size.width), int(bgr_rect.height*m_mapping_size.height));
	cv::Mat3b background_img = textured_decal_img(bgr_roi);
	cv::resize(background_img, background_img, m_mapping_size, 0, 0, cv::INTER_CUBIC);
	if (brightness_mult < 0)
		brightness_mult = estimate_brightness_multiplifer(background_img);

	preprocess_decal_img(m_mapping_size);
	background_img = brightness_mult * background_img;

	cv::Mat1f alpha_blend_img(m_decal_img.rows, m_decal_img.cols, 1.0);
	erased_texture = blend_decal_with_texture(m_tex_img, mapping_tex_coords, mapping_mask, background_img, alpha_blend_img);
	
	return NO_ERROR;
}

int DecalPainter::paint_decal(cv::Mat3b &painted_texture, float brightness_mult, float decal_smooth_sigma/*= 0.0*/)
{
	if (!check_valid_data())
		return INVALID_DATA;

	cv::Size m_tex_size = cv::Size(m_tex_img.cols, m_tex_img.rows);
	std::vector<FPointer> decal_trigs;
	cv::Mat2f  mapping_tex_coords;
	cv::Mat1b  mapping_mask;
	generate_mapping_texture_space(m_tex_size, decal_trigs, mapping_tex_coords, mapping_mask);
	fix_island_boundary_gaps(mapping_tex_coords, mapping_mask);

	preprocess_decal_img(m_mapping_size, decal_smooth_sigma);

	if (brightness_mult <= 0.0)
	{
		cv::Mat3b textured_decal_img = fill_decal_image_with_texture_colors(m_mapping_size, m_tex_img, mapping_tex_coords, mapping_mask);
		brightness_mult = estimate_brightness_multiplifer(textured_decal_img);
	}

	m_decal_img = brightness_mult * m_decal_img;
	painted_texture = blend_decal_with_texture(m_tex_img, mapping_tex_coords, mapping_mask, m_decal_img, m_decal_img_alpha);
	
	return NO_ERROR;
}

int  DecalPainter::erase_paint_decal(cv::Mat3b &modified_texture, cv::Rect2d bgr_rect, float brightness_mult, float decal_smooth_sigma/*= 0.0*/)
{
	if (!check_valid_data())
		return INVALID_DATA;

	cv::Size m_tex_size = cv::Size(m_tex_img.cols, m_tex_img.rows);
	std::vector<FPointer> decal_trigs;
	cv::Mat2f  mapping_tex_coords;
	cv::Mat1b  mapping_mask;
	generate_mapping_texture_space(m_tex_size, decal_trigs, mapping_tex_coords, mapping_mask);

	fix_island_boundary_gaps(mapping_tex_coords, mapping_mask);

	cv::Mat3b textured_decal_img = fill_decal_image_with_texture_colors(m_mapping_size, m_tex_img, mapping_tex_coords, mapping_mask);

	cv::Rect bgr_roi(int(bgr_rect.x*m_mapping_size.width), int(bgr_rect.y*m_mapping_size.height), int(bgr_rect.width*m_mapping_size.width), int(bgr_rect.height*m_mapping_size.height));
	cv::Mat3b background_img = textured_decal_img(bgr_roi);
	cv::resize(background_img, background_img, m_mapping_size, 0, 0, cv::INTER_CUBIC);
	if (brightness_mult < 0)
		brightness_mult = estimate_brightness_multiplifer(background_img);

	preprocess_decal_img(m_mapping_size, decal_smooth_sigma);

	cv::Mat3b painted_decal_over_bgr_img;
	cv::blendLinear(background_img, m_decal_img, 1.0 - m_decal_img_alpha, m_decal_img_alpha, painted_decal_over_bgr_img);

	painted_decal_over_bgr_img = brightness_mult * painted_decal_over_bgr_img;
	cv::Mat1f alpha_blend_img(m_decal_img.rows, m_decal_img.cols, 1.0);
	modified_texture = blend_decal_with_texture(m_tex_img, mapping_tex_coords, mapping_mask, painted_decal_over_bgr_img, alpha_blend_img);
	return NO_ERROR;
}

bool DecalPainter::set_mesh(std::string path)
{
	if (!is_file_exist(path))
	{
		std::cerr << "missing file: " << path << std::endl;
		return false;
	}

	tri::io::Importer<MyMesh>::Open(m_mesh, path.c_str());
	return true;
}

bool DecalPainter::set_mesh_texture(std::string path)
{
	if (!is_file_exist(path))
	{
		std::cerr << "missing file: " << path << std::endl;
		return false;
	}
	m_tex_img = cv::imread(path);
	return true;
}

void DecalPainter::set_decal_anchor_corners(std::array<vcgPoint3, 4> points)
{
	m_decal_anchor_corners = points;
}

bool DecalPainter::set_decal_anchor_corners(std::string path)
{
	if (!is_file_exist(path))
	{
		std::cerr << "missing file: " << path << std::endl;
		return false;
	}

	return import_decal_rectangle(path, m_decal_anchor_corners);
}

bool DecalPainter::set_decal_image(std::string path)
{
	if (!is_file_exist(path))
	{
		std::cerr << "missing file: " << path << std::endl;
		return false;
	}

	cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
	if (img.channels() != 4)
	{
		std::cerr << "decal image has not enough 4 channel" << std::endl;
		return false;
	}

	cv::Mat rgba[4];
	cv::split(img, rgba);
	cv::Mat rgb[3] = { rgba[0], rgba[1], rgba[2] };
	cv::merge(rgb, 3, m_decal_img);

	m_decal_img_alpha = (1.0 / 255.0) * rgba[3];
	return true;
}

bool DecalPainter::is_file_exist(string path)
{
	std::ifstream ff(path);
	if (!ff.good())
		return false;
	ff.close();
	return true;
}

void DecalPainter::set_mapping_size(cv::Size size, double paint_percent /*= 1.0*/)
{
	assert(paint_percent > 0 && paint_percent <= 1.0);
	m_mapping_size = size;
	m_paint_percent = paint_percent;
}

bool DecalPainter::import_decal_rectangle(std::string file_path, vcgRect3 &decal_rect)
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

