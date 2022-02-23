#include <opencv2/opencv.hpp>
#include <chrono>

//Greate histogram
cv::Mat createHistogram(cv::Mat &image) {
  int max = 0;
  int counts[256] = { 0 };
  for (int i = 0; i < 768; i++) {
    int current_sum = counts[image.at<uchar>(0, i)] + 1;
    counts[image.at<uchar>(0, i)] = current_sum;
    if (max < current_sum) max = current_sum;
  }
  max *= 3;
  max += 1;
  cv::Mat hist(max + 1, 768, CV_8UC1);
  hist = 240;

  for (int i = 0; i < 256; i++) {
    for (int j = max; j > (max - 3 * counts[i]); j--) {
      hist.at<uchar>(j, i * 3) = 0;
      hist.at<uchar>(j, i * 3 + 1) = 0;
      hist.at<uchar>(j, i * 3 + 2) = 0;
    }
  }
  return hist;
}

int main() {
  //Gamma correction coefficient
  double K = 2.2;

  //
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point stop;

  //I_1
  cv::Mat I_1(60, 768, CV_8UC1);
  for (int i = 0; i < 256; i++) {
    for (int j = 0; j < 60; j++) {
      I_1.at<uchar>(j, i * 3) = i;
      I_1.at<uchar>(j, i * 3 + 1) = i;
      I_1.at<uchar>(j, i * 3 + 2) = i;
    }
  }

  //G_1
  start = std::chrono::steady_clock::now();

  cv::Mat G_1_0;
  cv::Mat G_1_1;
  cv::Mat G_1;
  I_1.convertTo(G_1_0, CV_32F, 1.0 / 255);
  cv::pow(G_1_0, K, G_1_1);
  G_1_1.convertTo(G_1, CV_8UC1, 255);

  stop = std::chrono::steady_clock::now();

  std::cout << "G_1: " << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() << std::endl;

  //G_2
  start = std::chrono::steady_clock::now();

  cv::Mat G_2(60, 768, CV_8UC1);
  for (int i = 0; i < 768; i++) {
    for (int j = 0; j < 60; j++) {
      double pixel = pow(I_1.at<uchar>(j, i) / 255.0, K) * 255;
      G_2.at<uchar>(j, i) = cv::saturate_cast<uchar>(pixel);
    }
  }

  stop = std::chrono::steady_clock::now();

  std::cout << "G_2: " << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() << std::endl;


  //Greate image of histograms
  cv::Mat hist = createHistogram(G_1);
  cv::Mat hist_2 = createHistogram(G_2);

  hist.push_back(hist_2);
  cv::imwrite("histograms.png", hist);
  
  //Join images
  I_1.push_back(G_1);
  I_1.push_back(G_2);
  // save result
  cv::imwrite("lab01.png", I_1);
}
