#include "DecalPainter.h"
#include <iostream>

using namespace  std;

int main(int argc, char **argv)
{
	string DIR = "D:\\Projects\\Oh\\Oh\\src\\mesh\\data\\";
	
	string mesh_path =				DIR + "laxsquadT_mBBB_xLA4_0430_front.obj";
	string decal_rect_file_path =	DIR + "laxsquadT_mBBB_xLA4_0430_decal_rectangle.txt";
	string decal_img_path =			DIR + "decal.png";
	string org_texture_img_path =	DIR + "laxsquadT_mBBB_xLA4_0430.1001_origin.jpg";

	string out_texture_img_path =	DIR + "laxsquadT_mBBB_xLA4_0430.1001.jpg";

	DecalPainter dpainter;
	dpainter.set_mesh(mesh_path);
	dpainter.set_mesh_texture(org_texture_img_path);

	dpainter.set_decal_anchor_corners(decal_rect_file_path);
	dpainter.set_decal_image(decal_img_path);

	cv::Mat3b img_ret;
	//int err = dpainter.paint_decal(img_ret);
	int err = dpainter.erase_decal(img_ret, cv::Rect2d(0.1, 0.1, 0.1, 0.1));

	char c;
	if (err != DecalPainter::NO_ERROR)
	{
		std::cerr << dpainter.error_string(err);
		std::cout << "Failed. press a key to exit." << std::endl;
		std::cin >> c;
	}
	else 
	{
		cv::imwrite(out_texture_img_path, img_ret);
		std::cout << "Done. press a key to exit." << std::endl;
		std::cin >> c;
	}

	return 0;
}
