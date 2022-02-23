#include <opencv2/opencv.hpp>

void createMosaicOfChannels(cv::Mat &src, cv::Mat(&src_bgr)[3], cv::Mat &mosaic) {
  cv::Mat zero(256, 256, CV_8UC1);
  zero = 0;

  cv::split(src, src_bgr);

  cv::Mat src_b[3] = { src_bgr[0], zero, zero };
  cv::Mat src_g[3] = { zero, src_bgr[1], zero };
  cv::Mat src_r[3] = { zero, zero, src_bgr[2] };

  cv::Mat src_channels[3];

  cv::merge(src_b, 3, src_channels[0]);
  cv::merge(src_g, 3, src_channels[1]);
  cv::merge(src_r, 3, src_channels[2]);

  src_channels[0].copyTo(mosaic(cv::Rect(256, 256, 256, 256)));
  src_channels[1].copyTo(mosaic(cv::Rect(0, 256, 256, 256)));
  src_channels[2].copyTo(mosaic(cv::Rect(256, 0, 256, 256)));
  src.copyTo(mosaic(cv::Rect(0, 0, 256, 256)));
}

void createDiagram(cv::Mat(&bgr)[3], int n, int m, cv::Mat &diagram) {
  //coefficient for making diagram wider
  int K_x = 10;

  int channels[3][256] = { 0 };
  int max = 0;
  for (int k = 0; k < 3; k++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        int current_count = channels[k][bgr[k].at<uchar>(j, i)] + 1;
        channels[k][bgr[k].at<uchar>(j, i)] = current_count;
        if (current_count > max) max = current_count;
      }
    }
  }
  max += 1;
  cv::Mat c_diagram(max + 1, 768 * K_x, CV_8UC3);
  cv::Mat bgr_diagrams[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      cv::Mat x(max + 1, 256 * K_x, CV_8UC1);
      x = 0;
      bgr_diagrams[i][j] = x;
    }
  }


  for (int k = 0; k < 3; k++) {
    for (int i = 0; i < 256; i++) {
      for (int j = max; j > max - channels[k][i]; j--) {
        for (int p = 0; p < K_x; p++) {
          bgr_diagrams[k][k].at<uchar>(j, i * K_x + p) = 255;
        }
      }
    }
  }

  cv::merge(bgr_diagrams[0], 3, c_diagram(cv::Rect(0, 0, 256 * K_x, max + 1)));
  cv::merge(bgr_diagrams[1], 3, c_diagram(cv::Rect(256 * K_x, 0, 256 * K_x, max + 1)));
  cv::merge(bgr_diagrams[2], 3, c_diagram(cv::Rect(512 * K_x, 0, 256 * K_x, max + 1)));

  c_diagram.copyTo(diagram);
}

int main() {
  cv::Mat src_png = cv::imread("../../../data/cross_0256x0256.png");
  cv::imwrite("cross_0256x0256_025.jpg", src_png, { cv::ImwriteFlags::IMWRITE_JPEG_QUALITY, 25 });
  cv::Mat copy_jpeg_25 = cv::imread("cross_0256x0256_025.jpg");

  //src_png
  cv::Mat src_mosaic(512, 512, CV_8UC3);
  cv::Mat src_bgr[3];

  //mosaic for src_png
  createMosaicOfChannels(src_png, src_bgr, src_mosaic);

  cv::imwrite("cross_0256x0256_png_channels.png", src_mosaic);

  //diagram for src_png
  cv::Mat src_diagram;
  createDiagram(src_bgr, 256, 256, src_diagram);

  //copy_jpeg_25
  cv::Mat copy_mosaic(512, 512, CV_8UC3);
  cv::Mat copy_bgr[3];

  //mosaic for copy_jpeg_25
  createMosaicOfChannels(copy_jpeg_25, copy_bgr, copy_mosaic);

  cv::imwrite("cross_0256x0256_jpg_channels.png", copy_mosaic);

  //diagram for copy_jpeg
  cv::Mat copy_diagram;
  createDiagram(copy_bgr, 256, 256, copy_diagram);

  //Join diagrams
  src_diagram.push_back(copy_diagram);
  imwrite("cross_0256x0256_hists.png", src_diagram);
}