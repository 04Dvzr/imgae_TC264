


#include "pch.h"
#include "all.h"
#include <time.h>

using namespace cv;
using namespace std;
extern int c_rows, c_cols;


int main()
{
	VideoCapture capture(1); // 打开摄像头
	if (!capture.isOpened()) // 判断是否打开成功
	{
		cout << "open camera failed. " << endl;
		return -1;
	}

	capture.set(CAP_PROP_FRAME_WIDTH, 320);//宽度 
	capture.set(CAP_PROP_FRAME_HEIGHT, 240);//高度


	c_cols = capture.get(CAP_PROP_FRAME_WIDTH);
	c_rows = capture.get(CAP_PROP_FRAME_HEIGHT); 



	namedWindow("camera", WINDOW_GUI_EXPANDED);
	namedWindow("camera1", WINDOW_GUI_EXPANDED);
	namedWindow("after_proess", WINDOW_GUI_EXPANDED);
	
	while (true)
	{
		Mat frame;
		capture >> frame; // 读取图像帧至frame
		if (!frame.empty()) // 判断是否为空
		{
			imshow("camera", frame);

			Mat image;
			Mat show_image(c_rows, c_cols, CV_8UC1);
			Mat rect_image(show_image.size(), CV_8UC3);

			cvtColor(frame, image, COLOR_BGR2GRAY);
			imshow("camera1", image);

			show_image_matrix(image);


			near_avg_re(c_rows, c_cols,1);
			sobel_shaper(c_rows, c_cols,100);
			draw_temp_pic(show_image);


			imshow("after_proess", show_image);




			SQ AAAA;
//			AAAA = muti_shape(320,240,0,0,AAAA,20,200);
			AAAA = muti_search(320,240,4,100,20);
			rect_image = draw_rect(show_image, AAAA);
			namedWindow("after_proess1", WINDOW_GUI_EXPANDED);
			imshow("after_proess1", draw_rect(show_image, AAAA));

/*
			for (int i = 0; i != AAAA.count + 1; i++)
			{
			    cout << "中心点 row " << AAAA.point_row[i] << " col " << AAAA.point_col[i] << endl;
				cout << "row数 " << AAAA.row[i] << " col数 " << AAAA.col[i] << endl;
				cout << i << endl;
			}
*/
		}
		
		if (waitKey(30) > 0) // delay 30 ms等待按键
		{
			break;
		}
	}

	return 0;

}
