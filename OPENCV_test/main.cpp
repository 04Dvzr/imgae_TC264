//	/*
#include "pch.h"
#include "all.h"
#include <time.h>

using namespace cv;
using namespace std;

extern int IMG_TEMP[1000][1000];
extern int c_rows, c_cols;
extern int color_mode;			//0为黑白，1为三通道RGB



int main(int argc, char** argv)
{
	int start, stop;
	int a,b,c;
	double	duration;
	String filename = (argc >= 2) ? argv[1] : "../tset_img/g.jpg";
	Mat image;
	image = imread(filename, IMGMODE); // Read the file

	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}
	namedWindow("Display window", WINDOW_GUI_EXPANDED);// Create a window for display.
	imshow("Display window", image); // Show our image inside it.



	Mat noisy(image.size(), CV_8U);
//	/*
	// ADD NOISY
	Mat Noisy(image.size(), CV_32FC1);
	Mat Pic(image.size(), CV_32FC1);

	uchar2float(image, Pic);
	addNoise(sigma, Pic, Noisy);
	float2uchar(Noisy, noisy);
//	*/
//	noisy = addSaltNoise(image, sigma);

//	namedWindow("noisy window", WINDOW_GUI_EXPANDED);
//	imshow("noisy window", noisy);

	int row = image.rows;
	int col = image.cols;

/*
	SQI abc;
	abc.row = row;
	abc.col = col;
	abc.point_col = 0;
	abc.point_row = 0;
	abc.num= 0;

	SQI bc;
*/

	SQ AAAA;

	show_image_matrix(noisy);
	Mat img2(c_rows, c_cols, CV_8UC1);
	start = clock();

//	filter_near(row,col,2,35);
//	filter_near_avg(row, col, 2, 35, 20,18);
//	near_avg_re(row, col, 1);
//	round_plus(row, col, 3, 3, 75, 20);
//	round_plus_2(row,col,3.5,3.5,28,8);
//	lapace_shaper(row, col, 3.85,0.01);
//	filter_mid(row, col, 1);
//	filter_near(row, col, 1, 20);




//	bc=shape_postion2(abc,10,100);
//	a=shape_postion(row,col,0,0,10);
	AAAA = muti_shape(320,240,0,0,AAAA,20,200);


	stop = clock();
	duration = double(stop - start);
	draw_temp_pic(img2);
	namedWindow("Display1 window", WINDOW_GUI_EXPANDED);
	imshow("Display1 window", img2);
	cout << "time for filter_near:" << duration << " ms" << endl;

	for (int i = 0; i != AAAA.count+1; i++)
	{
		cout << "中心点 row " << AAAA.point_row[i] << " col " << AAAA.point_col[i] << endl;
		cout << "row数 " << AAAA.row[i] << " col数 " << AAAA.col[i] << endl;
		cout << i << endl;
	}

//	cout << "中心点 row " << bc.point_row <<" col "<< bc.point_col << endl;
//	cout << "row数 " << bc.row << " col数 " << bc.col << endl;



//对比显示
/*
	show_image_matrix(noisy);
	Mat img3(c_rows, c_cols, CV_8UC1);
	start = clock();

	near_avg_re(row, col, 1);
	round_plus_2(row,col,3.5,3.5,28,8);
//	liner_gray_2(row, col,144);
//	round_plus(row, col, 4, 4, 300, 8);
//	lapace_shaper(row, col, 3.85,0.01);


//	near_avg_re(row, col, 1);
//	svp(row, col, 1.8,3);
//	Perwitt_shaper(row, col,1,0);
//	liner_gray(row, col);
//	svp(row, col, 1.8,3);
//	filter_near(row, col, 2, 35);
//	filter_near(row, col, 1, 20);
//	filter_mid(row, col, 2);
//	filter_cross(row, col, 3, 30, 0.25);
//	filter_near_avg(row, col, 2, 35, 20);
//	shaper_mat(row, col, 2, 50);
	stop = clock();
	duration = double(stop - start);
	draw_temp_pic(img3);
	namedWindow("Display2 window", WINDOW_GUI_EXPANDED);
	imshow("Display2 window", img3);
	cout << "time for filter_cross:" << duration << " ms" << endl;
	*/


	waitKey(0);
	
	return 0;
}
