#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

//retrive array of frames from videos
void makeFramesList(cv::Mat(&images)[15], int num) {
  for (int i = 1; i <= num; i++) {
    cv::VideoCapture cap("../../../data/money/" + std::to_string(i) + ".mp4");
    int frames_count_dev_5 = ((int) cap.get(cv::CAP_PROP_FRAME_COUNT)) / 5;
    for (int j = 0; j <= frames_count_dev_5 * 2; j++) {
      cap.read(images[(i - 1) * 3]);
    }
    for (int j = 0; j <= frames_count_dev_5; j++) {
      cap.read(images[(i - 1) * 3 + 1]);
    }
    for (int j = 0; j <= frames_count_dev_5; j++) {
      cap.read(images[(i - 1) * 3 + 2]);
    }
    cap.release();
  }
}

//binarize original and inverted
void binarization(cv::Mat& gray, cv::Mat& bin, cv::Mat& bin_inv) {
  cv::ximgproc::niBlackThreshold(gray, bin, 255, cv::THRESH_BINARY, 221, 0.2);
  cv::ximgproc::niBlackThreshold(gray, bin_inv, 255, cv::THRESH_BINARY_INV, 221, 0.2);
}

//dilatation
void morfology(cv::Mat& bin_img) {
  cv::Mat structure_elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::dilate(bin_img, bin_img, structure_elem);
}

//find component
int chooseMaxSizeComponent(int count, cv::Mat& stats, int w, int h) {
  int max_ind = 0;
  int max_value = 0;
  for (int i = 2; i < count; i++) {
    int left = stats.at<int>(i, cv::CC_STAT_LEFT);
    int top = stats.at<int>(i, cv::CC_STAT_TOP);
    int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
    int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
    if (left != 0 && left + width != w && top != 0 && top + height != h) {
      int current_value = width * height;
      if (current_value > max_value) {
        max_value = current_value;
        max_ind = i;
      }
    }
  }
  return max_ind;
}

//fill black areas inside component
void makeMask(int ind, cv::Mat& labels, cv::Mat& mask) {
  mask = 0;
  for (int i = 0; i < labels.cols; i++) {
    for (int j = 0; j < labels.rows; j++) {
      if (labels.at<int>(j, i) == ind) {
        mask.at<uchar>(j, i) = 255;
      }
    }
  }

  //morphologization
  morfology(mask);

  mask = 255 - mask;
  cv::Mat labels_inv;
  cv::connectedComponents(mask, labels_inv);
  int ind_inv = labels_inv.at<int>(0, 0);
  for (int i = 0; i < labels_inv.cols; i++) {
    for (int j = 0; j < labels_inv.rows; j++) {
      if (labels_inv.at<int>(j, i) == ind_inv) {
        mask.at<uchar>(j, i) = 0;
      }
      else {
        mask.at<uchar>(j, i) = 255;
      }
    }
  }
}

//color mask
void makeColoredLayer(cv::Mat& mask, int channel) {
  cv::Mat channels_arr[3];
  for (int i = 0; i < 3; i++) {
    channels_arr[i] = cv::Mat(mask.size(), CV_8UC1);
  }
  channels_arr[channel] = mask;
  cv::merge(channels_arr, 3, mask);
}

//count white pixels on binarized image
int countOfWhitePixels(cv::Mat& bin) {
  int sum = 0;
  for (int i = 0; i < bin.cols; i++) {
    for (int j = 0; j < bin.rows; j++) {
      if (bin.at<uchar>(j, i) == 255) {
        sum++;
      }
    }
  }
  return sum;
}

int main() {
  cv::Mat images[15];
  int num = 5;

  //retrieve frames
  makeFramesList(images, num);
  
  for (int i = 0; i < num * 3; i++) {
    cv::Mat gray, bin, bin_inv;

    cv::imwrite("bgr" + std::to_string(i + 1) + ".png", images[i]);

    //color reduction
    cv::cvtColor(images[i], gray, cv::COLOR_BGR2GRAY);
    cv::imwrite("gray" + std::to_string(i + 1) + ".png", gray);

    //binarization
    binarization(gray, bin, bin_inv);

    //find money component
    cv::Mat labels, stats, centroids, labels_inv, stats_inv, centroids_inv;

    int w = bin.cols;
    int h = bin.rows;
    cv::Mat mask(h, w, CV_8UC1);
    int components_count = cv::connectedComponentsWithStats(bin, labels, stats, centroids);
    int components_count_inv = cv::connectedComponentsWithStats(bin_inv, labels_inv, stats_inv, centroids_inv);
    int max_ind = chooseMaxSizeComponent(components_count, stats, w, h);
    int max_inv_ind = chooseMaxSizeComponent(components_count_inv, stats_inv, w, h);
    int max = stats.at<int>(max_ind, cv::CC_STAT_WIDTH) * stats.at<int>(max_ind, cv::CC_STAT_HEIGHT);
    int max_inv = stats_inv.at<int>(max_inv_ind, cv::CC_STAT_WIDTH) * stats_inv.at<int>(max_inv_ind, cv::CC_STAT_HEIGHT);

    if (max > max_inv) {
      makeMask(max_ind, labels, mask);
    }
    else {
      makeMask(max_inv_ind, labels_inv, mask);
    }
    cv::imwrite("mask" + std::to_string(i + 1) + ".png", mask);

    //Read right mask
    cv::Mat right_mask = cv::imread("../../../data/money_masks/" + std::to_string(i + 1) + ".png", cv::IMREAD_GRAYSCALE);

    //Calculate precision
    cv::Mat joined_masks, intersectioned_masks;
    cv::bitwise_or(mask, right_mask, joined_masks);
    cv::bitwise_and(mask, right_mask, intersectioned_masks);
    double precision = 1.0 * countOfWhitePixels(intersectioned_masks) / countOfWhitePixels(joined_masks);
    std::cout << precision * 100 << "%" << std::endl;

    //Visualize masks areas
    cv::Mat changed_image;
    
    makeColoredLayer(mask, 2);
    cv::addWeighted(images[i], 1, mask, 1, 0, changed_image);

    makeColoredLayer(right_mask, 0);
    cv::addWeighted(changed_image, 1, right_mask, 1, 0, changed_image);

    cv::imwrite("original_with_areas" + std::to_string(i) + ".png", changed_image);
  }
}