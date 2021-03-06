#include "DecalPainter.h"
#include <iostream>
#include <string>

using namespace  std;

int main(int argc, char *argv[])
{
	//string DIR = "D:\\Projects\\Oh\\Oh\\src\\mesh\\data\\";
	//string mesh_path = DIR + "laxsquadT_mBBB_xLA4_0430_front.obj";
	//string decal_rect_file_path = DIR + "laxsquadT_mBBB_xLA4_0430_decal_rectangle.txt";
	//string decal_img_path = DIR + "decal.png";
	//string org_texture_img_path = DIR + "laxsquadT_mBBB_xLA4_0430.1001_origin.jpg";

	string mesh_path ;
	string org_texture_img_path;
	string decal_rect_file_path;
	string decal_img_path;
	string out_texture_img_path;
	string type;
	float brightness_val = -1;
	float decal_smooth_sigma = 0.0;
	auto check_valid_params = [&]() -> bool
	{
		if (mesh_path.empty()) 
		{
			std::cout << "missing mesh_path" << std::endl;
			return false;
		}
		if (org_texture_img_path.empty())
		{
			std::cout << "missing org_texture_img_path" << std::endl;
			return false;
		}
		if (decal_rect_file_path.empty()) 
		{
			std::cout << "missing decal_rect_file_path" << std::endl;
			return false;
		}
		if (decal_img_path.empty())
		{
			std::cout << "missing decal_img_path" << std::endl;
			return false;
		}
		if (out_texture_img_path.empty())
		{
			std::cout << "missing out_texture_img_path" << std::endl;
			return false;
		}
		return true;
	};

	for (int i = 1; i < argc; i++) {
		string argv_str(argv[i]);
		if (i + 1 != argc) {
			if (argv_str == "-type")
				type = argv[++i];
			else if (argv_str == "-m")
				mesh_path = argv[++i];
			else if (argv_str == "-t")
				org_texture_img_path = argv[++i];
			else if (argv_str == "-d")
				decal_img_path = argv[++i];
			else if (argv_str == "-r")
				decal_rect_file_path = argv[++i];
			else if (argv_str == "-o")
				out_texture_img_path = argv[++i];
			else if (argv_str == "-br")
				brightness_val = atof(argv[++i]);
			else if (argv_str == "-dsig")
				decal_smooth_sigma = atof(argv[++i]);
			else {
				std::cout << "Not enough or invalid arguments, please try again.\n" << std::endl;
				exit(0);
			}
		}
	}

	if (!check_valid_params())
	{
		std::cout << "Usage is -type <p-paint, e-erase> \n -br <brightness multiplifier, -1 or no set for automatic brightness estimation> \n -m <mesh_file> \n -t <texture_file> \n -d <decal_image>   \n -r <rectangle_text_file>  \n -o <output_texture>\n";
		std::cin.get();
		exit(0);
	}

	std::cout << "running..." << std::endl;
	DecalPainter dpainter;

	std::cout << "importing data..." << std::endl;
	dpainter.set_decal_anchor_corners(decal_rect_file_path);
	dpainter.set_decal_image(decal_img_path);

	dpainter.set_mesh(mesh_path);
	dpainter.set_mesh_texture(org_texture_img_path);

	cv::Mat3b img_ret;
	int err = 0;
	if (type == "p")
	{
		std::cout << "painting decal..." << std::endl;
		err = dpainter.paint_decal(img_ret, brightness_val, decal_smooth_sigma);
	}
	else if (type == "e") 
	{
		std::cout << "erasing decal.." << std::endl;
		err = dpainter.erase_decal(img_ret, cv::Rect2d(0.1, 0.1, 0.1, 0.1), brightness_val);
	}
	else if (type == "ep")
	{
		std::cout << "erasing and painting decal.." << std::endl;
		err = dpainter.erase_paint_decal(img_ret, cv::Rect2d(0.1, 0.1, 0.1, 0.1), brightness_val, decal_smooth_sigma);
	}

	if (err != DecalPainter::NO_ERROR)
	{
		std::cerr << dpainter.error_string(err);
		std::cout << "Failed.";
	}
	else 
	{
		cv::imwrite(out_texture_img_path, img_ret);
		std::cout << "Done." << std::endl;
	}

	return 0;
}
