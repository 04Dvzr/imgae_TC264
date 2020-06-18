#include "pch.h"
#include "all.h"

using namespace cv;
using namespace std;

int IMG_TEMP[4096][4096];
int IMG_TEMP2[4096][4096];
int c_rows, c_cols;
int color_mode = 0;			//0为黑白，1为三通道RGB
int prime[] = { 2,3,5,7,11,13,17,19,23,29,31,37,41 };

//OPENCV  函数调用、转化
/**************************************************************************************************************************************************************************************/
void img_read(cv::String add)			//看图片
{
	Mat image;
	image = imread(add, IMGMODE); // Read the file

	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << endl;
		return;
	}
	namedWindow("Display window", WINDOW_GUI_EXPANDED);// Create a window for display.
	imshow("Display window", image); // Show our image inside it.

	waitKey(0); // Wait for a keystroke in the window
}
/**************************************************************************************************************************************************************************************/
void show_image_matrix(cv::Mat image)				//RGB矩阵显示
{
//	int row = image.rows;
//	int col = image.cols;

//	c_rows = row;
//	c_cols = col;

	int row = c_rows;
	int col = c_cols;

	int **img = new int *[row];
	for (int i = 0; i < row; i++) 
		img[i] = new int[col];

	if (!IMGMODE)
	{
		color_mode = 0;	//黑白vision
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				img[i][j] = image.at<uchar>(i, j);
				IMG_TEMP[i][j] = img[i][j];
			}
		}
	}
	else {
		color_mode = 1;	//彩色vision
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				img[i][j] = (image.at<uchar>(i, j * 3 + 2) << 16) + (image.at<uchar>(i, j * 3 + 1) << 8) + image.at<uchar>(i, j*3);
				IMG_TEMP[i][j] = img[i][j];
			}
		}
	}
	/*
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			cout << "0x" << hex << IMG_TEMP[i][j] << " ";
		}
		cout << endl;
	}
//	*/
	for (int i = 0; i < row; i++) {
		delete[]img[i];
	}
	delete[] img;

}
/**************************************************************************************************************************************************************************************/
void showHistoCallback(Mat image)						//彩色直方图显示
{
	vector<Mat> bgr;
	split(image, bgr);

	int numbins = 256;

	float range[] = { 0,256 };
	const float* histRange = { range };

	Mat b_hist, g_hist, r_hist;

	calcHist(&bgr[0], 1, 0, Mat(), b_hist, 1, &numbins, &histRange);
	calcHist(&bgr[1], 1, 0, Mat(), g_hist, 1, &numbins, &histRange);
	calcHist(&bgr[2], 1, 0, Mat(), r_hist, 1, &numbins, &histRange);

	int width = 512;
	int height = 300;

	Mat histImage(height, width, CV_8UC3, Scalar(20, 20, 20));

	normalize(b_hist, b_hist, 0, height, NORM_MINMAX);
	normalize(g_hist, g_hist, 0, height, NORM_MINMAX);
	normalize(r_hist, r_hist, 0, height, NORM_MINMAX);

	int binStep = cvRound((float)width / (float)numbins);
	for (int i = 1; i < numbins; i++)
	{
		//B
		line(histImage,
			Point(binStep*(i - 1), height - cvRound(b_hist.at<float>(i - 1))),
			Point(binStep*(i), height - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0));
		//G
		line(histImage,
			Point(binStep*(i - 1), height - cvRound(g_hist.at<float>(i - 1))),
			Point(binStep*(i), height - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0));
		//R
		line(histImage,
			Point(binStep*(i - 1), height - cvRound(r_hist.at<float>(i - 1))),
			Point(binStep*(i), height - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255));
	}
	imshow("直方图", histImage);
}
/**************************************************************************************************************************************************************************************/
void trans_blackwhite_matrix(Mat image)					//灰度矩阵显示
{
	Mat yuv;
	int row = image.rows;
	int col = image.cols;

	if (IMGMODE)
		cvtColor(image, yuv, COLOR_BGR2YCrCb);
	else
		return;

	int **img = new int *[row];
	for (int i = 0; i < row; i++) {
		img[i] = new int[col];
	}

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			img[i][j] = yuv.at<uchar>(i, j);
		}
	}

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			cout << "0x" << hex << img[i][j] << " ";
		}
		cout << endl;
	}
	for (int i = 0; i < row; i++) {
		delete[]img[i];
	}
	delete[] img;
}
/**************************************************************************************************************************************************************************************/
void draw_temp_pic(cv::Mat image)
{
	uchar *p_tmp = NULL;
	for (int i = 0; i < c_rows; i++)
	{
		p_tmp = image.ptr<uchar>(i);
		for (int j = 0; j < c_cols; j++)
		{
			if (!color_mode)
			{
				p_tmp[j] = IMG_TEMP[i][j];
			}
			else {
				p_tmp[j * 3] = (IMG_TEMP[i][j] & 0xff);
				p_tmp[j * 3 + 1] = ((IMG_TEMP[i][j] & 0xff00) >> 8);
				p_tmp[j * 3 + 2] = ((IMG_TEMP[i][j] & 0xff0000) >> 16);
			}

		}
	}

//	namedWindow("NEW TEMP", WINDOW_GUI_EXPANDED);// Create a window for display.
//	imshow("NEW TEMP", image); // Show our image inside it.
}
/**************************************************************************************************************************************************************************************/
Mat addSaltNoise(const Mat srcImage, int n)
{
	Mat dstImage = srcImage.clone();
	for (int k = 0; k < n*500; k++)
	{
		//随机取值行列
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//图像通道判定
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 255;		//盐噪声
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 255;
			dstImage.at<Vec3b>(i, j)[1] = 255;
			dstImage.at<Vec3b>(i, j)[2] = 255;
		}
	}
	for (int k = 0; k < n*500; k++)
	{
		//随机取值行列
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//图像通道判定
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 0;		//椒噪声
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 0;
			dstImage.at<Vec3b>(i, j)[1] = 0;
			dstImage.at<Vec3b>(i, j)[2] = 0;
		}
	}
	return dstImage;
}
/**************************************************************************************************************************************************************************************/
Mat draw_rect(Mat img,SQ pos)
{
	if (pos.count < 0)
		return img;
	Mat out(img.size(), CV_8UC3);
	cvtColor(img, out, COLOR_GRAY2BGR);
	for (int i = 0; i <= pos.count; i++)
	{
		if (pos.row[i] < 0 || pos.col[i] < 0)
			break;
		rectangle(out, Rect((pos.point_row[i]- pos.row[i]/2), (pos.point_col[i] - pos.col[i] / 2), pos.row[i], pos.col[i]), Scalar(0, 0, 255), 1, 1, 0);
	}
	return out;
}





//图像降噪算法
/*
十字均值降噪

row为行数
col为列数
step为处理步长
thres_val为阈值
dif_rate为执行比率
*/
void filter_cross(const int row,const int col,const int step,const int thres_val,const double dif_rate)			
{
	int sum;
	double tmp;
	for (int i = step; i <= row - step; i++)
	{
		for (int j = step; j <= col - step; j++)
		{
			sum = 0;
			for (int k = 1; k <= step; k++)
			{
				if (abs(IMG_TEMP[i][j] - IMG_TEMP[i - k][j]) > thres_val)
					sum++;
				if (abs(IMG_TEMP[i][j] - IMG_TEMP[i][j - k]) > thres_val)
					sum++;
				if (abs(IMG_TEMP[i][j] - IMG_TEMP[i + k][j]) > thres_val)
					sum++;
				if (abs(IMG_TEMP[i][j] - IMG_TEMP[i][j + k]) > thres_val)
					sum++;
 			}
			tmp = (double)sum / (step * 4);
			sum = 0;
			if (tmp > dif_rate)
			{
				for (int k = 1; k <= step; k++) 
				{
					sum += IMG_TEMP[i - k][j];
					sum += IMG_TEMP[i][j - k];
					sum += IMG_TEMP[i + k][j];
					sum += IMG_TEMP[i][j + k];
				}
				IMG_TEMP[i][j] = sum / (4 * step);
			}
		}
	}
}
/**************************************************************************************************************************************************************************************/
/*
米字均值降噪

row为行数
col为列数
step为处理步长
thres_val为阈值
dif_rate为执行比率
*/
void filter_d_cross(const int row, const int col, const int step, const int thres_val, const double dif_rate)
{
	int sum;
	double tmp;
	for (int i = step; i <= row - step; i++)
	{
		for (int j = step; j <= col - step; j++)
		{
			sum = 0;
			for (int k = 1; k <= step; k++)
			{
				if (abs(IMG_TEMP[i][j] - IMG_TEMP[i - k][j]) > thres_val)
					sum++;
				if (abs(IMG_TEMP[i][j] - IMG_TEMP[i][j - k]) > thres_val)
					sum++;
				if (abs(IMG_TEMP[i][j] - IMG_TEMP[i + k][j]) > thres_val)
					sum++;
				if (abs(IMG_TEMP[i][j] - IMG_TEMP[i][j + k]) > thres_val)
					sum++;

				if (abs(IMG_TEMP[i][j] - IMG_TEMP[i - k][j - k]) > thres_val)
					sum++;
				if (abs(IMG_TEMP[i][j] - IMG_TEMP[i - k][j + k]) > thres_val)
					sum++;
				if (abs(IMG_TEMP[i][j] - IMG_TEMP[i + k][j - k]) > thres_val)
					sum++;
				if (abs(IMG_TEMP[i][j] - IMG_TEMP[i + k][j + k]) > thres_val)
					sum++;
			}
			tmp = (double)sum / (step * 8);
			sum = 0;
			if (tmp > dif_rate)
			{
				for (int k = 1; k <= step; k++)
				{
					sum += IMG_TEMP[i - k][j];
					sum += IMG_TEMP[i][j - k];
					sum += IMG_TEMP[i + k][j];
					sum += IMG_TEMP[i][j + k];
					sum += IMG_TEMP[i - k][j - k];
					sum += IMG_TEMP[i - k][j + k];
					sum += IMG_TEMP[i + k][j - k];
					sum += IMG_TEMP[i + k][j + k];
				}
				IMG_TEMP[i][j] = sum / (8 * step);
			}
		}
	}
}
/**************************************************************************************************************************************************************************************/
/*
邻域比较降噪

row为行数
col为列数
div为色彩区分度
step为处理步长
*/
void filter_near(const int row,const int col,const int step,const int div)
{
	int *pix_c = new int[div+1];
	int *count = new int[div+1];
	for (int l = div; l >= 0; l--)
	{
		pix_c[l] = 0;
		count[l] = 0;
	}
	int sum,sp;
	sp = 256 / div;
	for (int i = step; i <= row - step; i++)
	{
		for (int j = step; j <= col - step; j++)
		{
			sum = 0;
			for (int k = 1; k <= step; k++)
			{
				sum += IMG_TEMP[i - k][j];
				sum += IMG_TEMP[i][j - k];
				sum += IMG_TEMP[i + k][j];
				sum += IMG_TEMP[i][j + k];
			}
			sum = sum / (4 * step);
			for (int l = div; l >= 0; l--)
			{
				if (sum >= (sp*l) && sum < (sp*(l + 1)))
				{
					count[l]++;
					pix_c[l] += IMG_TEMP[i][j];
					break;
				}
			}
		}
	}
	for (int l = div; l >= 0; l--)
	{
		if (count[l] == 0)
			count[l] = 1;
		pix_c[l] = pix_c[l] / count[l];
	}
	for (int i = step; i <= row - step; i++)
	{
		for (int j = step; j <= col - step; j++)
		{
			sum = 0;
			for (int k = 1; k <= step; k++)
			{
				sum += IMG_TEMP[i - k][j];
				sum += IMG_TEMP[i][j - k];
				sum += IMG_TEMP[i + k][j];
				sum += IMG_TEMP[i][j + k];
			}
			sum = sum / (4 * step);
			for (int l = div; l >= 0; l--)
			{
				if (sum >= (sp*l) && sum < (sp*(l + 1)))
				{
					IMG_TEMP[i][j]=pix_c[l];
					if (IMG_TEMP[i][j] > 255)
						IMG_TEMP[i][j] = 255;
					else if(IMG_TEMP[i][j] < 0)
						IMG_TEMP[i][j] = 0;
					break;
				}
			}
		}
	}
	delete[] count;
	delete[] pix_c;
}
/**************************************************************************************************************************************************************************************/
/*
邻域比较降噪（大范围）				//没什么用/笑

row为行数
col为列数
div为色彩区分度
step为处理步长
*/
void filter_near2(const int row, const int col, const int step, const int div)
{
	int *pix_c = new int[div + 1];
	int *count = new int[div + 1];
	int **pic_d = new int *[5];
	for (int i = 0; i != 4; i++)
		pic_d[i] = new int[step + step * (step - 1) + 1];
	for (int l = div; l >= 0; l--)
	{
		pix_c[l] = 0;
		count[l] = 0;
	}
	int sum, sp;
	sp = 256 / div;
	for (int i = step; i <= row - step; i++)
	{
		for (int j = step; j <= col - step; j++)
		{
			sum = 0;
			near_serch(i, j, step, pic_d);
			for (int k = 0; k != 4; k++)
			{
				for (int l = 0; l != step + step * (step - 1); l++)
					sum += pic_d[k][l]+ IMG_TEMP[i][j];
			}
			sum = sum / (4 * (step + step * (step - 1)));
			for (int l = div; l >= 0; l--)
			{
				if (sum >= (sp*l) && sum < (sp*(l + 1)))
				{
					count[l]++;
					pix_c[l] += IMG_TEMP[i][j];
					break;
				}
			}
		}
	}
	for (int l = div; l >= 0; l--)
	{
		if (count[l] == 0)
			count[l] = 1;
		pix_c[l] = pix_c[l] / count[l];
	}
	for (int i = step; i <= row - step; i++)
	{
		for (int j = step; j <= col - step; j++)
		{
			sum = 0;
			near_serch(i, j, step, pic_d);
			for (int k = 0; k != 4; k++)
			{
				for (int l = 0; l != step + step * (step - 1); l++)
					sum += pic_d[k][l] + IMG_TEMP[i][j];
			}
			sum = sum / (4 * (step + step * (step - 1)));
			for (int l = div; l >= 0; l--)
			{
				if (sum >= (sp*l) && sum < (sp*(l + 1)))
				{
					IMG_TEMP[i][j] = pix_c[l];
					break;
				}
			}
		}
	}
	for (int i = 0; i != 4; i++)
		delete[] pic_d[i];
	delete[] pic_d;
	delete[] count;
	delete[] pix_c;
}
/**************************************************************************************************************************************************************************************/
/*
邻域搜素

row为行数
col为列数
step为处理步长
buf为定义的头指针
count为计数（迭代用），默认为0
*/
void near_serch(const int row, const int col, int step, int **buf,int count)
{
	if (step == 0)
		return;
	for (int i = 0; i != 2*step - 1; i++)
	{
		buf[0][count] = IMG_TEMP[row + i - 2][col + step] - IMG_TEMP[row][col];
		buf[1][count] = IMG_TEMP[row + step][col + i - 2] - IMG_TEMP[row][col];
		buf[2][count] = IMG_TEMP[row + i - 2][col - step] - IMG_TEMP[row][col];
		buf[3][count] = IMG_TEMP[row - step][col + i - 2] - IMG_TEMP[row][col];
		count++;
	}
	step--;
	near_serch(row,col,step,buf,count);
}
/**************************************************************************************************************************************************************************************/
/*
邻域计算

row为行数
col为列数
step为处理步长
buf为定义的二维数组头指针
buffer为定义的一维数组头指针
count为计数（迭代用），默认为0
base为迭代计数（迭代用），默认为0
*/
void near_cal(const int row, const int col, int step, int **buf,int *buffer, int count,int base)
{
	if (step == 0)
		return;
	for (int i = 0; i != 2 * step - 1; i++)
	{
		if ((count - row) != 0)
		{
			buffer[1 * (base + step) + base] += buf[1][count] * 0.5 / ((count - row)*(count - row));
			buffer[3 * (base + step) + base] += buf[3][count] * 0.5 / ((count - row)*(count - row));
		}
		if ((count - col) != 0)
		{
			buffer[0 * (base + step) + base] += buf[0][count] * 0.5 / ((count - col)*(count - col));
			buffer[2 * (base + step) + base] += buf[2][count] * 0.5 / ((count - col)*(count - col));
		}
		count++;
	}
	step--;
	base++;
	near_cal(row, col, step, buf, buffer, count, base);
}
/**************************************************************************************************************************************************************************************/
/*
邻域赋值

row为行数
col为列数
step为处理步长
buf为定义的二维数组头指针
buffer为定义的一维数组头指针
count为计数（迭代用），默认为0
base为迭代计数（迭代用），默认为0
*/
void near_val(const int row, const int col, int step, int **buf, int *buffer, int count, int base)
{
	if (step == 0)
		return;
	for (int i = 0; i != 4; i++)
	{
		buffer[base] = buf[0][step - 1 + count];
		buffer[3+base] = buf[1][step - 1 + count];
		buffer[6 + base] = buf[2][step - 1 + count];
		buffer[9 + base] = buf[3][step - 1 + count];
	}
	count = 2 * step - 1;
	step--;
	base++;
	near_val(row, col, step, buf, buffer, count, base);
}
/**************************************************************************************************************************************************************************************/
void filter_near_con1(const int row,const int col,const int step,const int dif)
{
	int *pix_c = new int[4 * step + 1];
	int **pix= new int*[5];
	for (int i = 0; i != 4; i++)
		pix[i] = new int[step + step * (step - 1)+1];
	int sum;
	for (int i = step; i <= row - step; i++)
	{
		for (int j = step; j <= col - step; j++)
		{
			near_serch(i, j, step, pix);
			near_val(i, j, step, pix, pix_c);
			near_cal(i, j, step, pix, pix_c);
			sum = 0;
			for (int k = 0; k != step + step * (step - 1); k++)
			{
				if (abs(pix_c[k]) > dif)
					pix_c[k] *= 0.5;
				sum += pix_c[k];
			}
			sum /= step + step * (step - 1);
			IMG_TEMP[i][j] += sum;
		}
	}
	for (int i = 0; i != 4; i++)
		delete[] pix[i];
	delete[] pix;
	delete[] pix_c;
}
/**************************************************************************************************************************************************************************************/
/*
邻域连续性滤波

row为行数
col为列数
step为处理步长
dif_level为色彩区分度
*/
void filter_continue(const int row, const int col,const int step,const int dif_level)
{
	double dif = 256 / dif_level;
	int **pix = new int*[5];
	for (int i = 0; i != 4; i++)
		pix[i] = new int[step + 1];
	double **pix_c = new double*[5];
	for (int i = 0; i != 4; i++)
		pix_c[i] = new double[step + 1];
	for (int i = step; i <= row - step; i++)
	{
		for (int j = step; j <= col - step; j++)
		{
			double sum = 0;
			double sum_p = 0;
			for (int k = 1; k != step + 1; k++)
			{
				pix[0][k - 1] = IMG_TEMP[i][j + k] - IMG_TEMP[i][j];
				pix_c[0][k - 1] = pix[0][k - 1] / dif / 10 + 1;
				sum += pix_c[0][k - 1];

				pix[1][k - 1] = IMG_TEMP[i - k][j] - IMG_TEMP[i][j];
				pix_c[1][k - 1] = pix[1][k - 1] / dif / 10 + 1;
				sum += pix_c[1][k - 1];

				pix[2][k - 1] = IMG_TEMP[i][j - k] - IMG_TEMP[i][j];
				pix_c[2][k - 1] = pix[2][k - 1] / dif / 10 + 1;
				sum += pix_c[2][k - 1];

				pix[3][k - 1] = IMG_TEMP[i + k][j] - IMG_TEMP[i][j];
				pix_c[3][k - 1] = pix[3][k - 1] / dif / 10 + 1;
				sum += pix_c[3][k - 1];
			}
			for (int k = 0; k !=step; k++)
			{
				sum_p += pix_c[0][k] / sum * pix[0][k];
				sum_p += pix_c[1][k] / sum * pix[1][k];
				sum_p += pix_c[2][k] / sum * pix[2][k];
				sum_p += pix_c[3][k] / sum * pix[3][k];
			}
			IMG_TEMP[i][j] += (int)sum_p;
		}
	}
	for (int i = 0; i != 4; i++)
		delete[] pix_c[i];
	delete[] pix_c;
	for (int i = 0; i != 4; i++)
		delete[] pix[i];
	delete[] pix;
}
/**************************************************************************************************************************************************************************************/
/*
邻域连续性滤波
（考虑方向性影响）

row为行数
col为列数
step为处理步长
dif_level为色彩区分度
*/
void filter_continue2(const int row, const int col, const int step, const int dif_level)
{
	double dif = 256 / dif_level;
	int **pix = new int*[5];
	for (int i = 0; i != 4; i++)
		pix[i] = new int[step + 1];
	double **pix_c = new double*[5];
	for (int i = 0; i != 4; i++)
		pix_c[i] = new double[step + 1];
	double *sum_d = new double[6];
	for (int i = step; i <= row - step; i++)
	{
		for (int j = step; j <= col - step; j++)
		{
			for (int k = 0; k != 6; k++)
				sum_d[k] = 0;
			for (int k = 1; k != step + 1; k++)
			{
				pix[0][k - 1] = IMG_TEMP[i][j + k] - IMG_TEMP[i][j];
				pix_c[0][k - 1] = pix[0][k - 1] / dif / 10 + 1;
				sum_d[0] += pix_c[0][k - 1];

				pix[1][k - 1] = IMG_TEMP[i - k][j] - IMG_TEMP[i][j];
				pix_c[1][k - 1] = pix[1][k - 1] / dif / 10 + 1;
				sum_d[1] += pix_c[1][k - 1];

				pix[2][k - 1] = IMG_TEMP[i][j - k] - IMG_TEMP[i][j];
				pix_c[2][k - 1] = pix[2][k - 1] / dif / 10 + 1;
				sum_d[2] += pix_c[2][k - 1];

				pix[3][k - 1] = IMG_TEMP[i + k][j] - IMG_TEMP[i][j];
				pix_c[3][k - 1] = pix[3][k - 1] / dif / 10 + 1;
				sum_d[3] += pix_c[3][k - 1];
			}
			sum_d[4] = sum_d[0] + sum_d[1] + sum_d[2] + sum_d[3];
			for (int k = 0; k != 4; k++)
				sum_d[k] = sum_d[k] / sum_d[4] + 1;
			for (int k = 0; k != step; k++)
			{
				sum_d[5] += pix_c[0][k] / sum_d[4] * sum_d[0] * pix[0][k];
				sum_d[5] += pix_c[1][k] / sum_d[4] * sum_d[1] * pix[1][k];
				sum_d[5] += pix_c[2][k] / sum_d[4] * sum_d[2] * pix[2][k];
				sum_d[5] += pix_c[3][k] / sum_d[4] * sum_d[3] * pix[3][k];
			}
			IMG_TEMP[i][j] += (int)sum_d[5];
		}
	}
	for (int i = 0; i != 4; i++)
		delete[] pix_c[i];
	delete[] pix_c;
	for (int i = 0; i != 4; i++)
		delete[] pix[i];
	delete[] pix;
	delete[] sum_d;
}
/**************************************************************************************************************************************************************************************/
/*
中值滤波

row为行数
col为列数
step为处理步长
*/
void filter_mid(const int row, const int col, const int step)
{
	int side = step * 2 + 1;
	int *tmp = new int[side*side + 1];
	int count;
	for (int i = step; i <= row - step; i++)
	{
		for (int j = step; j <= col - step; j++)
		{
			count = 0;
			for (int k = -step; k != step+1; k++)
			{
				for (int l = -step; l != step+1; l++)
				{
					tmp[count] = IMG_TEMP[i + k][j + l];
					if (count)
					{
						int m;
						if (tmp[count] < tmp[count - 1])
						{
							int temp = tmp[count];
							for (m = count - 1; m >= 0 && temp < tmp[m]; m--)
								tmp[m + 1] = tmp[m];
							tmp[m + 1] = temp;
						}
					}
					count++;
				}
			}
			IMG_TEMP[i][j] = tmp[side*side / 2];
		}
	}
	delete[] tmp;
}
/**************************************************************************************************************************************************************************************/
/*
邻域比较高斯分布降噪

row为行数
col为列数
div为色彩区分度
step为处理步长
sigma为高斯核sigma
*/
void filter_near_avg(const int row, const int col, const int step, const int div,const double sigma,int white_balance)
{
	double w_b = ((double)white_balance / 100) + 1.9999;
	int sp = 256 / div;
	double guss_core = -0.5 / (sigma * sigma);
	int *pix_c = new int[div + 1];
	int *count = new int[div + 1];
	double *guss = new double[sp + 1];
	for (int l = div; l >= 0; l--)
	{
		pix_c[l] = 0;
		count[l] = 0;
	}
	for (int i = 0; i != sp; i++)
		guss[i] = exp(i * i * guss_core);
	int sum;
	double r_h = row / 2, c_h = col / 2;
	for (int i = step; i <= row - step; i++)
	{
		for (int j = step; j <= col - step; j++)
		{
			sum = 0;
			for (int k = 1; k <= step; k++)
			{
				sum += IMG_TEMP[i - k][j];
				sum += IMG_TEMP[i][j - k];
				sum += IMG_TEMP[i + k][j];
				sum += IMG_TEMP[i][j + k];
			}
			sum = sum / (4 * step);
			for (int l = div; l >= 0; l--)
			{
				if (sum >= (sp*l) && sum < (sp*(l + 1)))
				{
					count[l]++;
					pix_c[l] += IMG_TEMP[i][j];
					break;
				}
			}
		}
	}
	for (int l = div; l >= 0; l--)
	{
		if (count[l] == 0)
			count[l] = 1;
		pix_c[l] = pix_c[l] / count[l];
	}
	for (int i = step; i <= row - step; i++)
	{
		for (int j = step; j <= col - step; j++)
		{
			sum = 0;
			for (int k = 1; k <= step; k++)
			{
				sum += IMG_TEMP[i - k][j];
				sum += IMG_TEMP[i][j - k];
				sum += IMG_TEMP[i + k][j];
				sum += IMG_TEMP[i][j + k];
			}
			sum = sum / (4 * step);
			for (int l = div; l >= 0; l--)
			{
				if (sum >= (sp*l) && sum < (sp*(l + 1)))
				{
					sum = (int)((abs(i - r_h) + abs(j - c_h)) / ((abs(i - r_h) + abs(j - c_h)) / sp + 1));
					IMG_TEMP[i][j] = pix_c[l] * (w_b - guss[sum]);
					if (IMG_TEMP[i][j] > 255)
						IMG_TEMP[i][j] = 255;
					else if (IMG_TEMP[i][j] < 0)
						IMG_TEMP[i][j] = 0;
					break;
				}
			}
		}
	}
	delete[] count;
	delete[] pix_c;
	delete[] guss;
}
/**************************************************************************************************************************************************************************************/
/*
邻域平均代替降噪

row为行数
col为列数
step为处理步长
*/
void near_avg_re(const int row, const int col, const int step)
{
	int sum,count;
	for (int i = 0; i != row; i++)
	{
		for (int j = 0; j != col; j++)
		{
			sum = 0;
			count = 0;
			for (int k = -step; k <= step; k++)
			{
				for (int l = -step; l <= step; l++)
				{
					if (i + k < 0 || j + l < 0)
						continue;
					else if (i + k >= row || j + l >= col)
						continue;
					else if (k == 0 && l == 0)
						continue;
					sum += IMG_TEMP[i + k][j + l];
					count++;
				}
			}
			if (count != 0)
			{
				IMG_TEMP[i][j] = sum / count;
			}
		}
	}
}
/**************************************************************************************************************************************************************************************/
/*
平滑算法

row为行数
col为列数
integ为处理步长
*/
void svp(const int row, const int col, const double integ,int step)
{
	if (step < 0)
		return;
	int sp = prime[step];
	int sum,count;
	for (int i = 0; i < row; i+= sp)
	{
		for (int j = 0; j < col; j+= sp)
		{
			sum = 0;
			count = 0;
			for (int k = 0; k != sp; k++)
			{
				for (int l = 0; l != sp; l++)
				{
					sum += IMG_TEMP[i + k][j + l] - IMG_TEMP[i][j];
					count++;
				}
			}
			if (abs(sum) <= (integ * (sp * sp - 1)))
			{
				IMG_TEMP[i][j] = (count * IMG_TEMP[i][j] + sum) / count;
				for (int k = 0; k != sp; k++)
				{
					for (int l = 0; l != sp; l++)
					{
						IMG_TEMP[i + k][j + l] = IMG_TEMP[i][j];
					}
				}
			}
		}
	}
	step--;
	svp(row, col, integ, step);
}



//图像增强、识别算法
/**************************************************************************************************************************************************************************************/
/*
拉普拉斯算法

row为行数
col为列数
lapace_val为拉普拉斯算子倍数
div为区分度
*/
void lapace_shaper(const int row, const int col,double lapace_val,double div)
{
	int sum,count,tot_val;
	double lapace_1[3][3] = { -1,-1,-1,
					 	 -1,lapace_val + div,-1,
						 -1,-1,-1 };
	double lapace_2[3][3] = { -1,-1,-1,
						 -1,lapace_val - div,-1,
						 -1,-1,-1 };
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			sum = 0;
			count = 0;
			for (int k = -1; k != 2; k++)
			{
				for (int l = -1; l != 2; l++)
				{
					if (i + k < 0 || j + l < 0)
						continue;
					else if (i + k >= row || j + l >= col)
						continue;
					sum += IMG_TEMP[i + k][j + l]* lapace_1[k+1][l+1];
					count++;
				}
			}
			sum /= count;
			tot_val = sum;
			sum = 0;
			for (int k = -1; k != 2; k++)
			{
				for (int l = -1; l != 2; l++)
				{
					if (i + k < 0 || j + l < 0)
						continue;
					else if (i + k >= row || j + l >= col)
						continue;
					sum += IMG_TEMP[i + k][j + l] * lapace_2[k + 1][l + 1];
					count++;
				}
			}
			sum /= count;
			tot_val -= sum;
			IMG_TEMP[i][j] = tot_val/2;
		}
	}
}
/**************************************************************************************************************************************************************************************/
/*
线性增强算法

row为行数
col为列数
*/
void liner_gray(const int row, const int col)
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (IMG_TEMP[i][j] < 48)
				IMG_TEMP[i][j] = IMG_TEMP[i][j] / 1.85;
			else if (IMG_TEMP[i][j] > 191)
				IMG_TEMP[i][j] = (IMG_TEMP[i][j] - 192) * 0.38 + 231;
			else IMG_TEMP[i][j] = (IMG_TEMP[i][j] - 38) * 1.5;


		}
	}


}
/**************************************************************************************************************************************************************************************/
/*
线性二极化算法

row为行数
col为列数
*/
void liner_gray_2(const int row, const int col,const int weight)
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (IMG_TEMP[i][j] < weight)
				IMG_TEMP[i][j] = 0;
			else
				IMG_TEMP[i][j] = 255;


		}
	}


}
/**************************************************************************************************************************************************************************************/
/*
邻域差分强化（整型）

row为行数
col为列数
step_r为纵处理步长
step_c为横处理步长
weight为权重
range为范围
*/
void round_plus(const int row, const int col, const int step_r, const int step_c, const int weight,const int range)
{ 
	if (step_r <= 2 || step_c <= 2 || range > 127)
		return;
	int sum, count, temp,s_r, s_rf, s_c, s_cf;
	temp = step_r % 2;
	if (temp == 1)
	{
		s_r = step_r / 2;
		s_rf = step_r / 2;
	}
	else 
	{
		s_r = step_r / 2;
		s_rf = step_r / 2 + 1;
	}
	temp = step_c % 2;
	if (temp == 1)
	{
		s_c = step_c / 2;
		s_cf = step_c / 2;
	}
	else
	{
		s_c = step_c / 2;
		s_cf = step_c / 2 + 1;
	}
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			sum = 0;
			count = 0;
			for (int k = -s_r; k <= s_rf; k++)
			{
				for (int l = -s_c; l <= s_cf; l++)
				{
					if (i + k < 0 || j + l < 0)
						continue;
					sum = IMG_TEMP[i + k][j + l];
					count++;
				}
			}
			if (count == 1)
				continue;
			else
			{
				sum = (sum - IMG_TEMP[i][j]) / (count - 1);
				IMG_TEMP[i][j] = IMG_TEMP[i][j] + sum * weight;
			}
			if (IMG_TEMP[i][j] > range && IMG_TEMP[i][j] < 255- range)
				IMG_TEMP[i][j] = 0;
			else
				IMG_TEMP[i][j] = 255;
		}
	}
}
/**************************************************************************************************************************************************************************************/
/*
邻域差分强化（浮点型）

row为行数
col为列数
step_r为纵处理步长
step_c为横处理步长
weight为权重
range为范围
*/
void round_plus_2(const int row, const int col,double step_r,double step_c, const double weight, const int range)
{
	if (step_r < 3 || step_c < 3 || range > 127)
		return;
	int  count, s_r, s_c;
	double sum;

	s_r = step_r;
	if (s_r % 2 != 1)
		s_r -= 1;
	step_r = (step_r - s_r) / 2;
	s_r /= 2;
	double *r_t = new double[s_r+2];
	for (int i = 0; i != s_r+1; i++)
	{
		r_t[s_r + 1 - i] = sqrt(step_r/(i+1)) / 3.5;
	}

	s_c = step_c;
	if (s_c % 2 != 1)
		s_c -= 1;
	step_c = (step_c - s_c) / 2;
	s_c /= 2;
	double *c_t = new double[s_c + 1];
	for (int i = 0; i != s_c + 1; i++)
	{
		c_t[s_c + 1 - i] = sqrt(step_r / (i + 1)) / 3.5;
	}


	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			sum = 0;
			count = 0;
			for (int k = -s_r; k <= s_r; k++)
			{
				for (int l = -s_c; l <= s_c; l++)
				{
					if (i + k < 0 || j + l < 0)
						continue;
					sum = IMG_TEMP[i + k][j + l] * (1 - r_t[abs(k)] - c_t[abs(l)]) + r_t[abs(k)] * IMG_TEMP[i + k + 1][j + l] + r_t[abs(k)] * IMG_TEMP[i + k - 1][j + l] + c_t[abs(l)] * IMG_TEMP[i + k][j + l + 1] + c_t[abs(l)] * IMG_TEMP[i + k][j + l - 1];
					count++;
				}
			}
			if (count == 1)
				continue;
			else
			{
				sum = (sum - IMG_TEMP[i][j]) / (count);
				IMG_TEMP[i][j] = IMG_TEMP[i][j] + sum * weight;
			}
			if (IMG_TEMP[i][j] > range && IMG_TEMP[i][j] < 255 - range)
				IMG_TEMP[i][j] = 0;
			else
				IMG_TEMP[i][j] = 255;
		}
	}
}
/**************************************************************************************************************************************************************************************/
void sobel_shaper(const int row, const int col,const int thres_val)
{
	int sum_r,sum_c, sum = 0;
	int sobel_row[3][3] = { 1, 2, 1,
						     0, 0, 0,
						    -1, -2, -1 };

	int sobel_col[3][3] = { -1, 0, 1,
						   -2, 0, 2,
						   -1, 0, 1 };

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			sum_r = 0;
			sum_c = 0;
			for (int k = -1; k != 2; k++)
			{
				for (int l = -1; l != 2; l++)
				{
					if (i + k < 0 || j + l < 0)
						continue;
					else if (i + k >= row || j + l >= col)
						continue;
					sum_r += IMG_TEMP[i + k][j + l] * sobel_row[k + 1][l + 1];

					sum_c += IMG_TEMP[i + k][j + l] * sobel_col[k + 1][l + 1];

				}
			}
			IMG_TEMP2[i][j] = abs(sum_r) + abs(sum_c);
			sum += IMG_TEMP2[i][j];

		}
	}
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (i == 0 || i == row - 1 || j == 0 || j == col - 1)
				IMG_TEMP[i][j] = 0;
			else if (abs(IMG_TEMP2[0][0] - IMG_TEMP2[i][j]) > thres_val)
				IMG_TEMP[i][j] = 255;
			else
				IMG_TEMP[i][j] = 0;
		}
	}
}




/**************************************************************************************************************************************************************************************/
/*
单对称目标中心点寻找

row为行数
col为列数
st_r为初始纵坐标
st_c为初始横坐标
tol为容差

返回为横纵坐标拼接
横坐标左移16位+纵坐标
*/
int shape_postion(int row,int col,int st_r,int st_c,const int tol)
{
	int temp = -1, core_r = 0, core_c = 0, count = 0;
	for (int i = st_r; i < row; i++)
	{
		for (int j = st_c; j < col; j++)
		{
			if (IMG_TEMP[i][j] - tol > IMG_TEMP[0][0] || IMG_TEMP[i][j] + tol < IMG_TEMP[0][0])
			{
				temp += j;
				count++;
			}
		}
		if (temp != -1)
		{
			core_c = (temp + 1) / count;
			temp = i;
			break;
		}
	}
	if (temp == -1)
		return -1;
	count = 0;
	int state = 1;
	for (int i = temp; i < row; i++)
	{
		if (state)
		{
			if (IMG_TEMP[i][core_c] - tol > IMG_TEMP[0][0] || IMG_TEMP[i][core_c] + tol < IMG_TEMP[0][0])
			{
				count++;
				core_r += i;
				state = 0;
			}
			if (count == 2)
			{
				core_r /= 2;
				break;
			}
		}
		else
		{
			if (IMG_TEMP[i][core_c] - tol > IMG_TEMP[temp][core_c] || IMG_TEMP[i][core_c] + tol < IMG_TEMP[temp][core_c])
				state = 1;
		}
	}
	if (count < 2)
		return -1;
	else
		return (core_c << 16) + core_r;
}
/**************************************************************************************************************************************************************************************/
/*
单目标寻找(方框选区)

使用SQI结构体
tol为容差
suqre_lim为面积阈值

返回为新SQI结构体
*/
SQI shape_postion2(SQI Base, const int tol,const int suqre_lim)
{
	if (Base.num > 5)
		return Base;

	int temp = -1, core_r = 0, core_c = 0, count = 0;
	for (int i = Base.point_row; i < Base.row; i++)
	{
		for (int j = Base.point_col; j < Base.col; j++)
		{
			if (IMG_TEMP[i][j] - tol > IMG_TEMP[0][0] || IMG_TEMP[i][j] + tol < IMG_TEMP[0][0])
			{
				temp += j;
				count++;
			}
		}
		if (temp != -1)
		{
			core_c = (temp + 1) / count;
			temp = i;
			break;
		}
	}
	if (temp == -1)
		return Base;

	SQI Gen;
	Gen.num = Base.num + 1;
	Gen.point_col = core_c;

	count = 0;
	int state = 1;
	for (int i = temp; i < Base.row; i++)
	{
		if (state)
		{
			if (IMG_TEMP[i][core_c] - tol > IMG_TEMP[0][0] || IMG_TEMP[i][core_c] + tol < IMG_TEMP[0][0])
			{
				count++;
				core_r += i;
				state = 0;
			}
			if (count == 2)
			{
				core_r /= 2;
				break;
			}
		}
		else
		{
			if (IMG_TEMP[i][core_c] - tol > IMG_TEMP[temp][core_c] || IMG_TEMP[i][core_c] + tol < IMG_TEMP[temp][core_c])
				state = 1;
		}
	}
	if (count < 2)
		return Base;

	Gen.point_row = core_r;
	Gen.row = core_r * 2 - 2 * (temp - 1);
	Gen.col = 0;
	count = core_c;
	while (count<Base.col)
	{
		if (IMG_TEMP[core_r][count] - tol > IMG_TEMP[0][0] || IMG_TEMP[core_r][count] + tol < IMG_TEMP[0][0])
			break;
		count++;
		Gen.col++;
	}

	temp = core_c;
	while (temp > 0)
	{
		if (IMG_TEMP[core_r][temp] - tol > IMG_TEMP[0][0] || IMG_TEMP[core_r][temp] + tol < IMG_TEMP[0][0])
			break;
		temp--;
		Gen.col++;
	}
	Gen.col++;

	if(Gen.col*Gen.row< suqre_lim)
		return Base;


	return Gen;
}
/**************************************************************************************************************************************************************************************/
/*
多目标寻找(严格方框选区)

row_a为横行扫描终点
col_a为纵行扫描终点
row_p为横行扫描起点
col_p为纵行扫描起点
tol为容差
suqre_lim为面积阈值

返回为SQ结构体
*/
SQ muti_shape(int row_a, int col_a,int row_p,int col_p ,SQ Base, const int tol, const int suqre_lim)
{
	if (Base.count > 10)
		return Base;
	int base_row= row_p;
	int base_col= col_p;

	int temp = -1, core_r = 0, core_c = 0, count = 0;
	for (int i = base_row; i < row_a; i++)
	{
		for (int j = base_col; j < col_a; j++)
		{
			if (IMG_TEMP[i][j] - tol > IMG_TEMP[0][0] || IMG_TEMP[i][j] + tol < IMG_TEMP[0][0])
			{
				temp += j;
				count++;
			}
		}
		if (temp != -1)
		{
			core_c = (temp + 1) / count;
			temp = i;
			break;
		}
	}
	if (temp == -1)
		return Base;

	Base.point_col[Base.count] = core_c;

	count = 0;
	int state = 1;
	for (int i = temp; i < row_a; i++)
	{
		if (state)
		{
			if (IMG_TEMP[i][core_c] - tol > IMG_TEMP[0][0] || IMG_TEMP[i][core_c] + tol < IMG_TEMP[0][0])
			{
				count++;
				core_r += i;
				state = 0;
			}
			if (count == 2)
			{
				core_r /= 2;
				break;
			}
		}
		else
		{
			if (IMG_TEMP[i][core_c] - tol > IMG_TEMP[temp][core_c] || IMG_TEMP[i][core_c] + tol < IMG_TEMP[temp][core_c])
				state = 1;
		}
	}
	if (count < 2)
		return Base;

	Base.point_row[Base.count] = core_r;
	Base.row[Base.count]= core_r * 2 - 2 * (temp - 1);
	Base.col[Base.count] = 0;
	count = core_c;
	while (count < col_a)
	{
		if (IMG_TEMP[core_r][count] - tol > IMG_TEMP[0][0] || IMG_TEMP[core_r][count] + tol < IMG_TEMP[0][0])
			break;
		count++;
		Base.col[Base.count]++;
	}

	temp = core_c;
	while (temp > 0)
	{
		if (IMG_TEMP[core_r][temp] - tol > IMG_TEMP[0][0] || IMG_TEMP[core_r][temp] + tol < IMG_TEMP[0][0])
			break;
		temp--;
		Base.col[Base.count]++;
	}
	Base.col[Base.count]++;

	if (Base.col[Base.count] * Base.row[Base.count] > suqre_lim)
	{
		Base.count++;

		Base = muti_shape(row_a, Base.point_col[Base.count - 1] - Base.col[Base.count - 1] / 2 - 1, Base.point_row[Base.count - 1] - Base.row[Base.count - 1] / 2 + 1, col_p, Base, tol, suqre_lim);

		Base = muti_shape(row_a, Base.point_col[Base.count - 1] + Base.col[Base.count - 1] / 2 - 1, Base.point_row[Base.count - 1] + Base.row[Base.count - 1] / 2 + 1, Base.point_col[Base.count - 1] - Base.col[Base.count - 1] / 2 + 1, Base, tol, suqre_lim);

		Base = muti_shape(row_a, col_a, Base.point_row[Base.count - 1] - Base.row[Base.count - 1] / 2 + 1, Base.point_col[Base.count - 1] + Base.col[Base.count - 1] / 2 + 1, Base, tol, suqre_lim);
	}
	
	return Base;
}



PP_S* outside_force(const uint16_t row, const uint16_t col,uint8_t dir,PP_S *all, const uint16_t all_row, const uint16_t all_col)
{
	uint8_t loop,dloop;
	bool state = 0;
	if (dir == LEFT)
	{
		loop = -1;
		dloop = 2;
	}
	else if (dir == RIGHT)
	{
		loop = 1;
		dloop = 1;
	}
	else
	{
		loop = -1;
		dloop = 3;
	}

	for (int8_t i = loop; i < loop + dloop; i++)
	{
		for (int8_t j = loop; j < loop + dloop; j++)
		{
			if (i == 0 && j == 0)
				break;
			else if (IMG_TEMP2[row + i][col + j] - IMG_TEMP2[row][col] < 20)
			{
				uint8_t temp = 1;
				do
				{
					temp++;
					if ((row + temp * i > all_row) || (col + temp * i > all_col)|| (row + temp * i < 0) || (col + temp * i < 0))
						return all;
				} while (abs(IMG_TEMP2[row + temp * i][col + temp * j] - IMG_TEMP2[row][col]) < 20);

				if (row + i * temp >= all->row_max)
				{
					all->row_max = row + i * temp;
					state = 1;
				}
				else if (row + i * temp <= all->row_min)
				{
					all->row_min = row + i * temp;
					state = 1;
				}

				if (col + j * temp >= all->col_max)
				{
					all->col_max = col + j * temp;
					state = 1;
				}
				else if (col + j * temp <= all->col_min)
				{
					all->col_min = col + j * temp;
					state = 1;
				}
			}
		}
	}
	if (state)
	{
		if (dir == LEFT)
			all = outside_force(row, col, LEFT, all, all_row, all_col);
		else if (dir == RIGHT)
			all = outside_force(row, col, RIGHT, all, all_row, all_col);
		else
		{
			all = outside_force(row, col, LEFT, all, all_row, all_col);
			all = outside_force(row, col, RIGHT, all, all_row, all_col);
		}
	}

	return all;
}


SQ approach(int row_a, int col_a, int row_p, int col_p ,SQ pos)
{
	if ((row_a >= row_p) || (col_a >= col_p) || (pos.count >= 10))
		return pos;

	PP_S Core;
	PP_S *core = &Core;

	for (int i = row_a; i < row_p; i++)
	{
		for (int j = col_a; j < col_p; j++)
		{
			if (IMG_TEMP2[i][j] > 240)
			{
				core = outside_force(i,j,ZERO,core, row_p, col_p);
				pos.point_row[pos.count] = (core->row_max + core->row_min) / 2;
				pos.point_col[pos.count] = (core->col_max + core->col_min) / 2;
				pos.col[pos.count] = core->row_max - core->row_min;
				pos.row[pos.count] = core->col_max - core->col_min;
				pos.count++;
				goto end_loop;
			}
		}
	}
	end_loop:
	pos = approach(row_a, core->col_min + 1, core->row_min - 1, col_p, pos);
	pos = approach(core->row_min, core->col_min, core->row_max, core->col_max, pos);
	pos = approach(core->row_max+1, core->col_min + 1, core->row_min, col_p, pos);


	return pos;
}


PP_S outside_search(int point_x, int point_y, const int step)
{
	PP_S ALL;
	if (point_x < step || point_x> c_rows - step || point_y< step || point_y> c_cols - step)
		return ALL;

	ALL.col_max = point_y;
	ALL.col_min = point_y;
	ALL.row_max = point_x;
	ALL.row_min = point_x;

	bool state = FALSE;

_LEFT:
	for (int j = -step; j <= 0; j++)
	{
		for (int i = -step; i <= step; i++)
		{
			if (IMG_TEMP[point_x + i][point_y + j] > 200)
			{
				if (i == 0 && j == 0)
					break;

				point_x = point_x + i;
				point_y = point_y + j;
				goto _LEFT;

			}
		}
	}
	ALL.col_min = point_y;

_DOWN:
	for (int i = step; i >= 0; i--)
	{
		for (int j = -step; j <= step; j++)
		{
			if (IMG_TEMP[point_x + i][point_y + j] > 200)
			{
				if (i == 0 && j == 0)
					break;

				point_x = point_x + i;
				point_y = point_y + j;
				goto _DOWN;

			}
		}
	}
	ALL.row_max = point_x;
	point_x = ALL.row_min;
	point_y = ALL.col_max;

_RIGHT:
	for (int j = step; j >= 0; j--)
	{
		for (int i = -step; i <= step; i++)
		{
			if (IMG_TEMP[point_x + i][point_y + j] > 200)
			{
				if (i == 0 && j == 0)
					break;

				point_x = point_x + i;
				point_y = point_y + j;
				goto _RIGHT;

			}
		}
	}
	ALL.col_max = point_y;

	return ALL;
}



SQ muti_search(int row, int col, const int step, const int space_area,const int offset)
{
	SQ ALL;
	PP_S TEMP;
	int point_x = offset, point_y = 0;
	bool state = FALSE;
	int temp;


	for (int i = point_x; i < row - offset; i++)
	{
		for (int j = point_y; j < col; j++)
		{
			if (ALL.count != 0)
			{
				temp = ALL.count;
				while (temp > 0)
				{
					if (i >= ALL.point_row[temp - 1] - ALL.row[temp - 1] / 2 && i <= ALL.point_row[temp - 1] + ALL.row[temp - 1] / 2 &&
						j >= ALL.point_col[temp - 1] - ALL.col[temp - 1] / 2 && j <= ALL.point_col[temp - 1] + ALL.col[temp - 1] / 2)
					{
						goto _NEXT;
					}
					temp--;
				}
			}

			if (IMG_TEMP[i][j] > 200)
			{
				if (ALL.count >= 9)
					return ALL;
				TEMP = outside_search(i, j, 4);
				if ((TEMP.col_max - TEMP.col_min)*(TEMP.row_max - TEMP.row_min) > space_area && ALL.count < 10)
				{
					ALL.row[ALL.count] = TEMP.row_max - TEMP.row_min + 4;
					ALL.col[ALL.count] = TEMP.col_max - TEMP.col_min + 4;
					ALL.point_row[ALL.count] = TEMP.row_min + (TEMP.row_max - TEMP.row_min) / 2;
					ALL.point_col[ALL.count] = TEMP.col_min + (TEMP.col_max - TEMP.col_min) / 2;
					ALL.count++;
				}
			}

		_NEXT:
			{

			}
		}
	}

	return ALL;
}





void Bilateral_Filter(int r, double sigma_d, double sigma_r)
{
	int i, j, m, n, k;
	int nx = c_rows, ny = c_cols;
	int w_filter = 2 * r + 1; // 滤波器边长
	int m_nChannels = 3;
	double gaussian_d_coeff = -0.5 / (sigma_d * sigma_d);
	double gaussian_r_coeff = -0.5 / (sigma_r * sigma_r);

	double** d_metrix = new double *[w_filter];// 空间权重
	for (int i = 0; i < w_filter; i++) {
		d_metrix[i] = new double[w_filter];
	}

	double r_metrix[256];  // 相似权重
	int m_imgData[3];

	// copy the original image
	double* img_tmp = new double[nx * ny * 3];

	for (i = 0; i < ny; i++)
		for (j = 0; j < nx; j++)
		{
			img_tmp[i * nx + j * 3] = (IMG_TEMP[i][j] & 0xff);
			img_tmp[i * nx + j * 3 + 1] = ((IMG_TEMP[i][j] & 0xff00) >> 8);
			img_tmp[i * nx + j * 3 + 2] = ((IMG_TEMP[i][j] & 0xff0000) >> 16);
		}

	// 计算空间权重
	for (i = -r; i <= r; i++)
		for (j = -r; j <= r; j++)
		{
			int x = j + r;
			int y = i + r;

			d_metrix[y][x] = exp((i * i + j * j) * gaussian_d_coeff);
		}

	// 计算相似权重
	for (i = 0; i < 256; i++)
	{
		r_metrix[i] = exp(i * i * gaussian_r_coeff);
	}

	//	/*
		// bilateral filter
	for (i = 0; i < ny; i++)
		for (j = 0; j < nx; j++)
		{
			for (k = 0; k < m_nChannels; k++)
			{
				double weight_sum, pixcel_sum;
				weight_sum = pixcel_sum = 0.0;

				for (m = -r; m <= r; m++)
					for (n = -r; n <= r; n++)
					{
						if (m*m + n * n > r*r)
							continue;

						int x_tmp = j + n;
						int y_tmp = i + m;

						x_tmp = x_tmp < 0 ? 0 : x_tmp;
						x_tmp = x_tmp > nx - 1 ? nx - 1 : x_tmp;   // 边界处理，replicate
						y_tmp = y_tmp < 0 ? 0 : y_tmp;
						y_tmp = y_tmp > ny - 1 ? ny - 1 : y_tmp;

						int pixcel_dif = (int)abs(img_tmp[y_tmp * m_nChannels * nx + m_nChannels * x_tmp + k] - img_tmp[i * m_nChannels * nx + m_nChannels * j + k]);
						double weight_tmp = d_metrix[m + r][n + r] * r_metrix[pixcel_dif];  // 复合权重

						pixcel_sum += img_tmp[y_tmp * m_nChannels * nx + m_nChannels * x_tmp + k] * weight_tmp;
						weight_sum += weight_tmp;
					}

				pixcel_sum = pixcel_sum / weight_sum;
				m_imgData[i * m_nChannels * nx + m_nChannels * j + k] = (uchar)pixcel_sum;

			} // 一个通道
			IMG_TEMP[i][j] = (m_imgData[2] << 16) + (m_imgData[1] << 8) + m_imgData[0];
		} // END ALL LOOP

//	*/
//	UpdateImage();
//	DeleteDoubleMatrix(d_metrix, w_filter, w_filter);
	delete[] img_tmp;
	for (int i = 0; i < w_filter; i++)
	{
		delete[]d_metrix[i];
	}
	delete[] d_metrix;

}
