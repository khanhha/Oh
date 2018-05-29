
#include "DecalPainter.h"

DecalPainter::DecalPainter()
	: m_mapping_size(2*1024, 2*2014)
	, m_paint_percent(1.0)
{}

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

	int face_v_idx[3];
	for (auto fit = mesh.face.begin(); fit != mesh.face.end(); ++fit)
	{
		auto idx = vcg::tri::Index(mesh, *fit);
		for (int i = 0; i < 3; ++i)
			F(idx, i) = vcg::tri::Index(mesh, fit->V(i));
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

double DecalPainter::calc_path_len(const std::vector<VPointer> &path)
{
	double len = 0;
	for (int i = 1; i < path.size(); ++i)
		len += (path[i]->cP() - path[i - 1]->cP()).Norm();
	return len;
}

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

	return CvRect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
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

DecalPainter::VPointer DecalPainter::edge_other_vert(MyMesh::EdgePointer e, VPointer v)
{
	return (e->V(0) == v) ? e->V(1) : e->V(0);
}

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

bool DecalPainter::find_geodesic_path(MyMesh &mesh, VPointer vstart, VPointer vend, std::vector<VPointer> &path, int max_path_len /*= 10000*/)
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
			break;

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
		path.clear();
		return false;
	}
}

bool DecalPainter::find_decal_boundary(MyMesh &mesh, OctreeType &octree, vcgRect3 &decal_rect, std::vector<VPointer> &decal_verts, std::vector<std::vector<VPointer>> &paths)
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
		VPointer vend   = decal_verts[(i + 1)%4];
		ret[i] = find_geodesic_path(mesh, vstart, vend, paths[i], 100000);
	}

	return (ret[0] && ret[1] && ret[2] && ret[3]);
}

void DecalPainter::merge_path(std::vector<std::vector<VPointer>> &paths, std::vector<VPointer> &path)
{
	for (auto &p : paths)
		path.insert(path.end(), p.begin(), p.end());
}

bool DecalPainter::extract_decal_triangles(MyMesh &mesh, std::vector<std::vector<VPointer>> &boundaries, FPointer seed_trig, std::vector<FPointer> &decal_trigs)
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

	return nullptr;
}

void DecalPainter::triangle_texture_coords(FPointer trig, cvVec2 tex_cos[3])
{
	for (int i = 0; i < 3; ++i) {
		const auto &co = trig->WT(i);
		tex_cos[i][0] = co.U();
		tex_cos[i][1] = co.V();
	}
}

void DecalPainter::generate_texture_coordinates(const std::vector<FPointer> &trigs, const EMatrixX &F, const EMatrixXScalar &V_uv, const cv::Size2i &img_size, cv::Mat2f &tex_coords)
{
	tex_coords = cv::Mat2f(img_size);
	cvPoint2 img_trig_verts[3];
	cvVec2 tri_tex_coords[3];

	for(auto i = 0; i < F.rows(); ++i)
	{
		EVector3X trig = F.row(i);

		triangle_texture_coords(trigs[i], tri_tex_coords);

		//to image space
		for (auto vi = 0; vi < 3; ++vi)
		{
			img_trig_verts[vi].x = V_uv(trig[vi],0) * (double)(img_size.width-1);
			img_trig_verts[vi].y = V_uv(trig[vi],1) * (double)(img_size.height-1);
		}

		CvRect rect = triangle_bounding_rect(img_trig_verts);
		double L1, L2, L3;
		for (int ix = 0; ix < rect.width; ix++)
		{
			int px = rect.x + ix;
			for (int iy = 0; iy < rect.height; iy++)
			{
				int py = rect.y + iy;
				//note: our unit is one pixel, therefore, choosing a big epsilon to make sure that each pixel belongs to a triangle
				if (is_inside_triangle(img_trig_verts, cvPoint2((double)px, (double)py), L1, L2, L3, 1))
				{
					//cvVec2 tex((double)rand()/RAND_MAX, (double)rand()/RAND_MAX);
					//cvVec2 tex = 1.0f/3.0f*(trig.tex_coords[0] + trig.tex_coords[1] + trig.tex_coords[2]);
					cvVec2 tex = L1*tri_tex_coords[0] + L2*tri_tex_coords[1] + L3*tri_tex_coords[2];
					//cvVec2 tex = trig.tex_coords[0];
					tex_coords(px, py)[0] = tex[0];
					tex_coords(px, py)[1] = tex[1];
				}
			}
		}
	}
}

cv::Point DecalPainter::select_seed_point_texture_space(const cv::Mat1b &mapping)
{
	//to do
	return cv::Point(0, 0);
}

cv::Mat1b DecalPainter::generate_decal_area_in_texture_space(const std::vector<FPointer> &decal_trigs, cv::MatSize texture_size)
{
	cv::Mat1b mask_triangle_tex_img = cv::Mat1b(texture_size[0], texture_size[1], uchar(0));
	cv::Scalar color(255, 255, 255);
	draw_texture_triangle_over_img(mask_triangle_tex_img, decal_trigs, color);
	cv::Point seed_pnt = select_seed_point_texture_space(mask_triangle_tex_img);
	cv::Mat1b mask_inv = mask_triangle_tex_img.clone();
	cv::floodFill(mask_inv, seed_pnt, 255);
	mask_inv = 255 - mask_inv;
	cv::bitwise_or(mask_triangle_tex_img, mask_inv, mask_triangle_tex_img);

	cv::Mat strel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20));
	cv::dilate(mask_triangle_tex_img, mask_triangle_tex_img, strel);
}

void DecalPainter::fix_tiny_gaps(const std::vector<FPointer> &decal_trigs, cv::Mat1b &blended_mask, cv::Mat3b &tex_img)
{
	cv::Mat1b mask_triangle_tex_img = generate_decal_area_in_texture_space(decal_trigs, m_tex_img.size);
	mask_triangle_tex_img = mask_triangle_tex_img - blended_mask;

	cv::inpaint(tex_img, mask_triangle_tex_img, tex_img, 5, cv::INPAINT_NS);
}

void DecalPainter::blend_decal_with_texture(
	const cv::Mat3b &tex_img, const cv::Mat2f &tex_coords, const cv::Rect2i &blend_rect, const cv::Mat &decal_img, 
	cv::Mat3b &blended_tex_img, cv::Mat1b &blend_mask)
{
	blended_tex_img = m_tex_img.clone();
	cv::MatSize tex_size	= m_tex_img.size;
	blend_mask = cv::Mat1b(blended_tex_img.size[0], blended_tex_img.size[1], uchar(0));
	cv::Mat3b	textured_decal_img = cv::Mat3b(blend_rect.width, blend_rect.height);

	for (int i = 0; i < blend_rect.width; ++i)
	{
		int px = i + blend_rect.x;
		for (int j = 0; j < blend_rect.height; ++j)
		{
			int py = j + blend_rect.y;
			cvVec2 co = tex_coords(px, py);
			co[1] = 1.0 - co[1];
			int tex_ix = int((double)tex_size[0] * co[1]);
			int tex_iy = int((double)tex_size[1] * co[0]);
			if (tex_ix >= 0 && tex_ix < tex_size[0] && tex_iy >= 0 && tex_iy < tex_size[1])
			{
				cv::Vec3f decal_pix = decal_img.at<cv::Vec3f>(blend_rect.width - i - 1, j);
				if (true/*decal_pix[0] > threshold && decal_pix[1] > threshold && decal_pix[2] > threshold*/)
				{
					cv::Vec3f tex_pix = m_tex_img(tex_ix, tex_iy);
					textured_decal_img(px, py) = tex_pix;
					blended_tex_img(tex_ix, tex_iy) = decal_pix;
					blend_mask(tex_ix, tex_iy) = uchar(255);
				}
			}
		}
	}
#if 0
	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\textured_decal.png", textured_decal_img);
	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\laxsquadT_mBBB_xLA4_0430.1001.jpg", mod_tex_img);
	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\texture_decal_blend_mask.png", mask_tex_img);
	//cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\texture_decal_blend_mask_1.png", mask_tex_img_1);
	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\texture_decal_blend_mask_triangle.png", mask_triangle_tex_img);
#endif
}

void DecalPainter::draw_texture_triangle_over_img(cv::Mat &tex, const std::vector<FPointer> &trigs, Scalar color)
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
			tri_points[i].x = tri_coords[i][0] * width;
			tri_points[i].y = (1.0 - tri_coords[i][1])* height;
		}
		for (auto i = 0; i < 3; ++i)
			cv::line(tex, tri_points[i], tri_points[(i + 1) % 3], color, 1, cv::LINE_4);
	}
}

cv::Mat3b DecalPainter::generate_background_image(cv::Size size, cv::Vec3b mean_color, cv::Vec3b variance)
{
	cv::RNG rng(12345);
	cv::Mat3b img(size.width, size.height);
	rng.fill(img, cv::RNG::NORMAL, mean_color, variance);
	return img;
}

cv::Vec3b DecalPainter::find_background_color(const cv::Mat3b &img)
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

cv::Mat3b DecalPainter::build_textured_rasterization(const cv::Size &size, const cv::Mat3b &tex_img, const cv::Mat3b &decal_img, cv::Rect2i decal_rect, const cv::Mat2f &tex_coords)
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


size_t DecalPainter::total_vertices(const std::vector<std::vector<VPointer>> &boundary)
{
	size_t cnt = 0;
	for (int i = 0; i < boundary.size(); ++i)
		cnt += boundary[i].size();
	return cnt;
}

#ifdef _DEBUG

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

cv::Mat3b DecalPainter::output_textured_rasterization(const cv::Size &size, const cv::Mat3b &tex_img, const cv::Mat3b &decal_img, cv::Rect2i decal_rect, const cv::Mat2f &tex_coords)
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

	//cv::Mat1b img(1024, 1024, uchar(0));
	//test_draw_triangles_over_texture(img, F, V_uv);
	//cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\harmonic_parameterization.png", img);
}

bool DecalPainter::find_3D_mesh_decal_area(MyMesh &mesh, vcgRect3 decal_rect, std::vector<FPointer> &decal_trigs, std::vector<std::vector<VPointer>> &boundary)
{
	typedef vcg::Octree<MyMesh::VertexType, double> OctreeType;
	OctreeType octree;
	octree.Set(mesh.vert.begin(), mesh.vert.end());

	vcg::tri::RequireVEAdjacency(mesh);
	vcg::tri::UpdateTopology<MyMesh>::FaceFace(mesh);
	vcg::tri::UpdateTopology<MyMesh>::AllocateEdge(mesh);
	vcg::tri::UpdateTopology<MyMesh>::VertexEdge(mesh);

	std::vector<VPointer> decal_verts(4);
	boundary.resize(4);
	if (!find_decal_boundary(mesh, octree, decal_rect, decal_verts, boundary))
		return false;

	//find a seed triangle
	FPointer seed_tri = find_seed_triangle(mesh, octree, decal_rect);
	if (!seed_tri)
		return false;
	
	if (!extract_decal_triangles(mesh, boundary, seed_tri, decal_trigs))
		return false;
	
#if 0
	MyMesh decal_mesh;
	construct_a_mesh(mesh, decal_trigs, decal_mesh);
	string export_file_path = "D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\test_decal_mesh.obj";
	tri::io::Exporter<MyMesh>::Save(decal_mesh, export_file_path.c_str());
#endif
	return true;
}

void DecalPainter::paint_decal()
{
	//read texture image
	cv::Size2i		paint_size(static_cast<int>(m_paint_percent*m_mapping_size.width), static_cast<int>(m_paint_percent*m_mapping_size.height));
	cv::Point2i		decal_pos(static_cast<int>(0.5*(m_mapping_size.width - paint_size.width)), static_cast<int>(0.5*(m_mapping_size.height - paint_size.height)));
	cv::Rect2i		decal_rect(decal_pos.x, decal_pos.y, paint_size.width, paint_size.height);

	const cv::MatSize &tex_size = m_tex_img.size;

	cv::flip(m_decal_img, m_decal_img, 1);
	cv::rotate(m_decal_img, m_decal_img, cv::ROTATE_90_CLOCKWISE);
	cv::resize(m_decal_img, m_decal_img, paint_size, 0, 0, INTER_AREA);

	std::vector<FPointer> decal_trigs;
	std::vector<std::vector<VPointer>> rect_boundary;
	find_3D_mesh_decal_area(m_mesh, m_decal_anchor_corners, decal_trigs, rect_boundary);

	EMatrixX F;
	EMatrixXScalar V_uv;
	parameterizre_decal_rect(m_mesh, decal_trigs, rect_boundary, F, V_uv);

	cv::Mat2f  tex_coords;
	generate_texture_coordinates(decal_trigs, F, V_uv, m_mapping_size, tex_coords);

#if 0
	cv::Mat1b tex_img_triangle = cv::Mat1b(tex_img.size[0], tex_img.size[1]);
	test_draw_triangles_over_texture(tex_img_triangle, local_decal_trigs);
	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\triangle_texture_space.png", tex_img_triangle);

	cv::Mat3b our_masterpiece_1 = test_draw_triangles(size, local_decal_trigs);
	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\rasterization_triangles.png", our_masterpiece_1);

	exit(1);
#endif

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
	textured_ras_img = output_textured_rasterization(size, m_tex_img, decal_img, decal_rect, tex_coords);

	cv::Vec3b bg_color = find_background_color(textured_ras_img);
	int variance = 3;
	cv::Mat3b bg_img = generate_background_image(size, bg_color, cv::Vec3b(variance, variance, variance));

	//textured_ras_img = test_draw_triangles_over_image(textured_ras_img, local_decal_trigs);
	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\rasterize.png", textured_ras_img);
	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\make_up_bacground_image.png", bg_img);
#endif

	cv::Mat3b mod_tex_img;
	cv::Mat1b blended_mask;
	blend_decal_with_texture(m_tex_img, tex_coords, decal_rect, m_decal_img, mod_tex_img, blended_mask);

#if 0
	cv::Mat3b mod_tex_img = m_tex_img.clone();
	cv::Mat1b mask_tex_img = cv::Mat1b(mod_tex_img.size[0], mod_tex_img.size[1], uchar(0));
	cv::Mat3b textured_decal_img = cv::Mat3b(paint_size.width, paint_size.height);
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
				cv::Vec3f decal_pix = m_decal_img(decal_rect.width - i - 1, j);
				const double threshold = 0.0;
				if (true/*decal_pix[0] > threshold && decal_pix[1] > threshold && decal_pix[2] > threshold*/)
				{
					cv::Vec3f tex_pix = m_tex_img(tex_ix, tex_iy);
					textured_decal_img(px, py) = tex_pix;
					mod_tex_img(tex_ix, tex_iy) = decal_pix;
					mask_tex_img(tex_ix, tex_iy) = uchar(255);
				}
			}
		}
	}

	cv::Mat1b mask_triangle_tex_img = cv::Mat1b(m_tex_img.size[0], m_tex_img.size[1], uchar(0));
	cv::Scalar color(255, 255, 255);
	draw_texture_triangle_over_img(mask_triangle_tex_img, decal_trigs, color);
	cv::Point seed_pnt = select_seed_point_texture_space(mask_triangle_tex_img);
	cv::Mat1b mask_inv = mask_triangle_tex_img.clone();
	cv::floodFill(mask_inv, seed_pnt, 255);
	mask_inv = 255 - mask_inv;
	cv::bitwise_or(mask_triangle_tex_img, mask_inv, mask_triangle_tex_img);

	cv::Mat strel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20));
	cv::dilate(mask_triangle_tex_img, mask_triangle_tex_img, strel);

	mask_triangle_tex_img = mask_triangle_tex_img - mask_tex_img;

	cv::inpaint(mod_tex_img, mask_triangle_tex_img, mod_tex_img, 5, cv::INPAINT_NS);

	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\textured_decal.png", textured_decal_img);
	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\laxsquadT_mBBB_xLA4_0430.1001.jpg", mod_tex_img);
	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\texture_decal_blend_mask.png", mask_tex_img);
	//cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\texture_decal_blend_mask_1.png", mask_tex_img_1);
	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\result\\texture_decal_blend_mask_triangle.png", mask_triangle_tex_img);
#endif
}

bool DecalPainter::set_mesh(std::string path)
{
	if (is_file_exist(path))
		return false;

	tri::io::Importer<MyMesh>::Open(m_mesh, path.c_str());
	return true;
}

bool DecalPainter::set_mesh_texture(std::string path)
{
	if (!is_file_exist(path))
		return false;
	cv::Mat3b	tex_img = cv::imread(path);
	return true;
}

void DecalPainter::set_decal_anchor_corners(std::array<vcgPoint3, 4> points)
{
	m_decal_anchor_corners = points;
}

bool DecalPainter::set_decal_anchor_corners(std::string path)
{
	return import_decal_rectangle(path, m_decal_anchor_corners);
}

bool DecalPainter::set_decal_image(std::string path)
{
	if (!is_file_exist(path))
		return false;

	m_decal_img = cv::imread(path, cv::IMREAD_COLOR);
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

int main(int argc, char **argv)
{
	DecalPainter dpainter;
	string mesh_path = "D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\laxsquadT_mBBB_xLA4_0430_front.obj";
	string decal_rect_file_path = "D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\laxsquadT_mBBB_xLA4_0430_decal_rectangle.txt";
	string decal_img_path = "D:\\Projects\\Oh\\data\\3D\\Texture_retargeting\\decal_images\\decal.png";
	string texture_img_path = "D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\laxsquadT_mBBB_xLA4_0430.1001_backup.jpg";

	dpainter.set_mesh(mesh_path);
	dpainter.set_mesh_texture(texture_img_path);

	dpainter.set_decal_anchor_corners(decal_rect_file_path);
	dpainter.set_decal_image(decal_img_path);

	dpainter.paint_decal();

	return 0;
}

