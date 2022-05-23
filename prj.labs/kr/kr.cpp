#include <opencv2/opencv.hpp>

cv::Mat makePicture() {
  std::vector<int> rect_colors = { 0, 127, 250, 127, 250, 0 };
  std::vector<int> circle_colors = { 127, 0, 127, 250, 0, 250 };
  cv::Mat picture(300, 450, CV_8UC1);
  picture = 0;

  //draw rectangles
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      cv::rectangle(picture, cv::Rect(j * 150, i * 150, 150, 150), rect_colors[i * 3 + j], -1);
    }
  }

  //draw circles
  int r = 60;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      cv::circle(picture, cv::Point(j * 2 * 75 + 75 , i * 2 * 75 + 75), r, circle_colors[i * 3 + j], -1);
    }
  }

  return picture;
}

void applyFilter(cv::Mat& src, cv::Mat& dst, cv::Mat& kernel) {
  cv::filter2D(src, dst, -1, kernel, cv::Point(1, 1));
}

//create mat equals to geometrical mean of two mats
void makeGeometricMean(cv::Mat& dst, cv::Mat& m1, cv::Mat& m2) {
  int cols = m1.cols;
  int rows = m1.rows;
  for (int i = 0; i < cols; i++) {
    for (int j = 0; j < rows; j++) {
      dst.at<uchar>(cv::Point(i, j)) = round(sqrt(pow(m1.at<uchar>(cv::Point(i, j)), 2) + pow(m2.at<uchar>(cv::Point(i, j)), 2)));
    }
  }
}

int main() {
  //make original picture
  cv::Mat original = makePicture();
  cv::imwrite("I0.png", original);

  //apply 1 filter and make I1
  cv::Mat I1;
  std::vector<float> k_vector_1 = {1, 0, -1, 2, 0, -2, 1, 0, -1};
  cv::Mat kernel_1(3, 3, CV_32FC1);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      kernel_1.at<float>(j, i) = k_vector_1[i * 3 + j];
    }
  }
  applyFilter(original, I1, kernel_1);
  cv::imwrite("I1.png", I1);

  //apply 2 filter and make I2
  cv::Mat I2;
  std::vector<float> k_vector_2 = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
  cv::Mat kernel_2(3, 3, CV_32FC1);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      kernel_2.at<float>(j, i) = k_vector_2[i * 3 + j];
    }
  }
  applyFilter(original, I2, kernel_2);
  cv::imwrite("I2.png", I2);

  //geometrical mean of I1 and I2
  cv::Mat geometric_mean(original.size(), CV_8UC1);
  makeGeometricMean(geometric_mean, I1, I2);
  cv::imwrite("I3.png", geometric_mean);
}