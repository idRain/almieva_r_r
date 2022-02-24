#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>

uchar func(int x) {
  return cv::saturate_cast<uchar>(100 * cos(x * M_PI / 510) + 155);
}

void createTable(uchar(&table)[256], int n) {
  for (int i = 0; i < n; i++) {
    table[i] = func(i);
  }
}

void drawGraph(uchar(&table)[256], int n, int p, cv::Mat& graph) {
  graph = 255;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      for (int k = 0; k < p; k++) {
        graph.at<uchar>(511 - table[i] * p - j, i * p + k) = 0;
      }
    }
  }
}

int main() {
  //rgb
  cv::Mat rgb_img = cv::imread("../../../data/cross_0256x0256.png");
  cv::imwrite("lab03_rgb.png", rgb_img);

  //graiscale
  cv::Mat gre_img(256, 256, CV_8UC1);
  cv::cvtColor(rgb_img, gre_img, cv::COLOR_BGR2GRAY);
  cv::imwrite("lab03_gre.png", gre_img);

  //func vizualization
  uchar table[256] = { 0 };
  createTable(table, 256);

  cv::Mat graph(512, 512, CV_8UC1);
  drawGraph(table, 256, 2, graph);
  cv::imwrite("lab03_viz_func.png", graph);

  cv::Mat lut(1, 256, CV_8UC1, table);

  //func to greyscale
  cv::Mat gre_res(256, 256, CV_8UC1);
  cv::LUT(gre_img, lut, gre_res);
  cv::imwrite("lab03_gre_res.png", gre_res);

  //func to rgb
  cv::Mat rgb[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      //zero Mat
      cv::Mat zero(256, 256, CV_8UC1);
      zero = 0;
      rgb[i][j] = zero;
    }
  }
  
  cv::Mat channels[3];
  cv::split(rgb_img, channels);
  for (int i = 0; i < 3; i++) {
    cv::LUT(channels[i], lut, rgb[i][i]);
  }

  cv::Mat rgb_res(256, 768, CV_8UC3);
  for (int i = 0; i < 3; i++) {
    cv::merge(rgb[i], 3, rgb_res(cv::Rect(i * 256, 0, 256, 256)));
  }
  cv::imwrite("lab03_rgb_res.png", rgb_res);
}