
#ifndef ALL_H
#define ALL_H



#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdlib.h>
#include "bm3d.h"


#define IMGMODE IMREAD_GRAYSCALE		//IMREAD_GRAYSCALE,IMREAD_COLOR
constexpr auto sigma = 1;						//value of sigma;

void img_read(cv::String add);
void show_image_matrix(cv::Mat image);
void showHistoCallback(cv::Mat image);
void trans_blackwhite_matrix(cv::Mat image);
void draw_temp_pic(cv::Mat image);

cv::Mat addSaltNoise(const cv::Mat srcImage, int n);

void Bilateral_Filter(int r, double sigma_d, double sigma_r);
void filter_cross(const int row, const int col, const int step, const int thres_val, const double dif_rate);
void filter_d_cross(const int row, const int col, const int step, const int thres_val, const double dif_rate);
void filter_near(const int row, const int col, const int step, const int div);
void filter_near2(const int row, const int col, const int step, const int div);
void near_serch(const int row, const int col, int step, int **buf, int count = 0);
void near_cal(const int row, const int col, int step, int **buf, int *buffer, int count = 0, int base = 0);
void near_val(const int row, const int col, int step, int **buf, int *buffer, int count = 0, int base = 0);
void filter_near_con1(const int row, const int col, const int step, const int dif);
void filter_continue(const int row, const int col, const int step, const int dif_level);
void filter_continue2(const int row, const int col, const int step, const int dif_level);
void filter_mid(const int row, const int col, const int step);
void filter_near_avg(const int row, const int col, const int step, const int div, const double sigma, int white_balance = 0);
void near_avg_re(const int row, const int col, const int step);
void svp(const int row, const int col, const double integ, int step = 12);


void lapace_shaper(const int row, const int col, double prewitt_val, double div);
void liner_gray(const int row, const int col);
void round_plus(const int row, const int col, const int step_r, const int step_c, const int weight, const int range = 5);
void round_plus_2(const int row, const int col, double step_r, double step_c, const double weight, const int range = 5);
void liner_gray_2(const int row, const int col,const int weight);
int shape_postion(int row, int col, int st_r, int st_c, const int tol);

#endif
