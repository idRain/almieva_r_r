#include <opencv2/opencv.hpp>
#include <string>
#include "json.hpp"
#include <fstream>
using json = nlohmann::json;

//retrive array of frames from videos
void makeFramesList(cv::Mat(&images)[15], int num) {
  for (int i = 1; i <= num; i++) {
    cv::VideoCapture cap("../../../data/money/" + std::to_string(i) + ".mp4");
    int frames_count_dev_5 = ((int)cap.get(cv::CAP_PROP_FRAME_COUNT)) / 5;
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

//calculate distance between two points
double calculateDistance(cv::Point p1, cv::Point p2) {
  return sqrt(pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2));
}

//calculate perimetr
double getPerimetr(std::vector<cv::Point> points) {
  double perimetr = 0;
  for (int i = 0; i < points.size(); i++) {
    perimetr += calculateDistance(points[i], points[(i + 1) % 4]);
  }
  return perimetr;
}

//get corners map
cv::Mat getPointsMap(std::vector<cv::Point>& points, cv::Size img_size, bool flag) {
  if (!flag) {
    cv::Mat points_mat(img_size, CV_8UC1);
    points_mat = 0;
    for (int i = 0; i < points.size(); i++) {
      points_mat.at<uchar>(points[i]) = 255;
    }
    return points_mat;
  }
  else {
    cv::Mat points_mat(img_size, CV_8UC3);
    points_mat = 0;
    for (int i = 0; i < points.size(); i++) {
      cv::line(points_mat, cv::Point(points[i].x - 3, points[i].y), cv::Point(points[i].x + 3, points[i].y), cv::Scalar(255, 255, 255));
      cv::line(points_mat, cv::Point(points[i].x, points[i].y - 3), cv::Point(points[i].x, points[i].y + 3), cv::Scalar(255, 255, 255));
      cv::circle(points_mat, points[i], 10, cv::Scalar(255, 255, 255), 2);
    }
    return points_mat;
  }
  
}

//chose start or end
void choseStartOrEnd(int a1, int a2, int& b1, int& b2) {
  if (a1 < a2) {
    b1 = a1;
    b2 = a2;
  }
  else {
    b1 = a2;
    b2 = a1;
  }
}

//comparision
bool compare(std::pair<double, cv::Point> a, std::pair<double, cv::Point> b) {
  if (a.first < b.first) return true;
  return false;
}

//count of white points in area between 2 points
int getCountOfWhitePoints(cv::Point& point1, cv::Point& point2, cv::Mat& map) {
  int start_x, end_x, start_y, end_y;
  choseStartOrEnd(point1.x, point2.x, start_x, end_x);
  choseStartOrEnd(point1.y, point2.y, start_y, end_y);

  int count = 0;
  for (int i = start_y; i <= end_y; i++) {
    for (int j = start_x; j <= end_x; j++) {
      if (map.at<uchar>(i, j) == 255) count++;
    }
  }
  return count;
}

//find document corners from points of document
void findDocumentCorners(std::vector<cv::Point> points, cv::Mat& map, cv::Size size, std::vector<cv::Point>& corners) {
  std::vector<std::pair<double, cv::Point>> possible_corners;
  
  //find points on edges
  std::vector<int> counts(4, 0);
  for (int i = 0; i < points.size(); i++) {
    counts[0] = getCountOfWhitePoints(cv::Point(0, size.height), points[i], map);
    counts[1] = getCountOfWhitePoints(cv::Point(0, 0), points[i], map);
    counts[2] = getCountOfWhitePoints(cv::Point(size.width, 0), points[i], map);
    counts[3] = getCountOfWhitePoints(cv::Point(size.width, size.height), points[i], map);

    for (int j = 0; j < 4; j++) {
      if (counts[j] <= 1) {
        std::pair <double, cv::Point> current_pair(0, points[i]);
        possible_corners.push_back(current_pair);
        break;
      }
    }
  }
  
  //calculate max values of corners 
  for (int i = 0; i < possible_corners.size(); i++) {
    for (int j = 0; j < possible_corners.size() && j != i; j++) {
      for (int k = j + 1; k < possible_corners.size() && k != i; k++) {
        double x1 = possible_corners[j].second.x - possible_corners[i].second.x;
        double x2 = possible_corners[k].second.x - possible_corners[i].second.x;
        double y1 = possible_corners[j].second.y - possible_corners[i].second.y;
        double y2 = possible_corners[k].second.y - possible_corners[i].second.y;
        double current_corner = acos((x1 * x2 + y1 * y2) / calculateDistance(possible_corners[i].second, possible_corners[j].second) / calculateDistance(possible_corners[i].second, possible_corners[k].second));
        if (possible_corners[i].first < current_corner) possible_corners[i].first = current_corner;
      }
    }
  }

  //sort corners regarding to their values
  std::sort(possible_corners.begin(), possible_corners.end(), compare);

  //consider average value for 4 points
  cv::Point average_point(0, 0);
  for (int i = 0; i < 4; i++) {
    average_point.x += possible_corners[i].second.x;
    average_point.y += possible_corners[i].second.y;
  }
  average_point.x /= 4;
  average_point.y /= 4;

  //place 4 points in necessary order
  for (int i = 0; i < 4; i++) {
    if (possible_corners[i].second.x < average_point.x && possible_corners[i].second.y > average_point.y) {
      corners[0] = possible_corners[i].second;
    }
    else if (possible_corners[i].second.x < average_point.x && possible_corners[i].second.y < average_point.y) {
      corners[1] = possible_corners[i].second;
    }
    else if (possible_corners[i].second.x > average_point.x && possible_corners[i].second.y < average_point.y) {
      corners[2] = possible_corners[i].second;
    }
    else {
      corners[3] = possible_corners[i].second;
    }
  }
}

//change points positions
void changePosition(std::vector<cv::Point>& points, int width, int height) {
  for (int i = 0; i < points.size(); i++) {
    points[i].x = points[i].x * 8;
    points[i].y = points[i].y * 8;
  }
}

//calculate mean
double getMean(std::vector<double>& errors) {
  double sum = 0;
  for (int i = 0; i < errors.size(); i++) {
    sum += errors[i];
  }
  return sum / errors.size();
}

//calculate stdev
double getStdev(std::vector<double>& errors) {
  double sum = 0;
  double mean = getMean(errors);
  for (int i = 0; i < errors.size(); i++) {
    sum += pow(errors[i] - mean, 2);
  }
  return sum / errors.size();
}


int main() {
  cv::Mat images[15];
  int num = 5;

  //for precission estimate
  std::vector<double> errors;

  //retrieve frames
  makeFramesList(images, num);

  for (int i = 0; i < num * 3; i++) {
    cv::Mat gray, filtered;
    std::vector<cv::Point> points;

    cv::imwrite("bgr" + std::to_string(i + 1) + ".png", images[i]);

    //color reduction
    cv::cvtColor(images[i], gray, cv::COLOR_BGR2GRAY);
    cv::imwrite("gray" + std::to_string(i + 1) + ".png", gray);

    //resize image
    cv::Mat resized;
    cv::resize(gray, resized, gray.size() / 8, 0, 0);

    //Gauss filter
    cv::GaussianBlur(resized, filtered, cv::Size(3, 3), 1);

    //find corners' points
    cv::goodFeaturesToTrack(filtered, points, 30, 0.0001, 1, cv::Mat(), 7, true);

    //vizualize points map
    cv::Mat points_map = getPointsMap(points, filtered.size(), false);

    //variables for corners
    std::vector<cv::Point> corners(4);

    //finding probably corners of document
    findDocumentCorners(points, points_map, filtered.size(), corners);

  /*-- Start. Estimate calculated edges precission -- */

    //array for real points
    std::vector<cv::Point> real_corners(4);

    //read JSON
    std::ifstream j("../../../data/money.json", std::ifstream::binary);
    json r_points;
    j >> r_points;
    for (int j = 0; j < 4; j++) {
      json current_point = r_points.at("bgr" + std::to_string(i + 1)).at("regions")[j].at("shape_attributes");
      int x = current_point.at("cx");
      int y = current_point.at("cy");
      real_corners[j] = cv::Point(x, y);
    }

    //change points positions for original size
    changePosition(points, 8, 4);
    changePosition(corners, 8, 4);

    //Calculate accurancy
    double max_destinition = -1;
    for (int k = 0; k < 4; k++) {
      double current_dest = calculateDistance(real_corners[k], corners[k]);
      if (current_dest > max_destinition) max_destinition = current_dest;
    };
    errors.push_back(max_destinition / getPerimetr(real_corners));

  /*-- End. Estimate calculated edges precission -- */

    //vizualization of points map
    cv::imwrite("points_map" + std::to_string(i + 1) + ".png", getPointsMap(points, images[i].size(), true));

    //vizualizaton of calculated edge
    cv::Mat document_edges(images[i].size(), CV_8UC3);
    cv::cvtColor(gray, document_edges, cv::COLOR_GRAY2BGR);
    cv::polylines(document_edges, corners, true, cv::Scalar(0, 0, 255), 2);
    cv::polylines(document_edges, real_corners, true, cv::Scalar(0, 255, 0), 2);
    cv::imwrite("corners" + std::to_string(i + 1) + ".png", document_edges);
  }


  //write standart deviation
  std::cout << "Average error: " << getMean(errors) << std::endl;
  std::cout << "Precission: " << getStdev(errors) << std::endl;
}