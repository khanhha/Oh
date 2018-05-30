#include "DecalPainter.h"
#include <iostream>

using namespace  std;

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

	cv::Mat3b img_ret;
	int err = dpainter.paint_decal(img_ret);
	//int err = dpainter.erase_decal(img_ret, cv::Rect2d(0.1, 0.1, 0.1, 0.1));

	cv::imwrite("D:\\Projects\\Oh\\data\\3D\\AG_laxsquad_Box\\laxsquadT_mBBB_xLA4_0430.1001.jpg", img_ret);

	char c;
	cout << "done";
	cin >> c;
	return 0;
}
