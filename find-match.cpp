//*******************surf.cpp******************//
//********** SURF implementation in OpenCV*****//
//**loads image, computes SURF keypoints and descriptors **//


#include <stdio.h>
#include  <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "affine2d.h"

#include <iostream>
#define MAX_MATCHES 100

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define CONFIDENCE 0.99
#define DISTANCE 1.2

/** @function readme */
void readme()
{
  printf("Usage: ./SURF_detector <EO IMG> <IR IMG> <hessian>\n");
}

// A basic symmetry test
void symmetryTest(const std::vector<cv::DMatch> &matches1, const std::vector<cv::DMatch> &matches2, std::vector<cv::DMatch> &symMatches)
{
  symMatches.clear();
  for (vector<DMatch>::const_iterator matchIterator1 = matches1.begin(); matchIterator1 != matches1.end(); ++matchIterator1) {
    for (vector<DMatch>::const_iterator matchIterator2 = matches2.begin(); matchIterator2 != matches2.end(); ++matchIterator2) {
      if ((*matchIterator1).queryIdx == (*matchIterator2).trainIdx && (*matchIterator2).queryIdx == (*matchIterator1).trainIdx) {
        symMatches.push_back(DMatch((*matchIterator1).queryIdx, (*matchIterator1).trainIdx, (*matchIterator1).distance));
        break;
      }
    }
  }
}

// ransac test for stereo images
// void ransacTest(const std::vector<cv::DMatch> matches, const std::vector<cv::KeyPoint> &keypoints1,
//                 const std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> &goodMatches, double distance,
//                 double confidence, double minInlierRatio)
// {
//   goodMatches.clear();
//   // Convert keypoints into Point
//   std::vector<cv::Point> points1, points2;
//   for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it) {
//     // Get the position of left keypoints
//     float x = keypoints1[it->queryIdx].pt.x;
//     float y = keypoints1[it->queryIdx].pt.y;
//     points1.push_back(cv::Point(x, y));
//     // Get the position of right keypoints
//     x = keypoints2[it->trainIdx].pt.x;
//     y = keypoints2[it->trainIdx].pt.y;
//     points2.push_back(cv::Point(x, y));
//   }
//   // Compute F matrix using RANSAC
//   std::vector<uchar> inliers(points1.size(), 0);
//   cv::Mat fundemental = cv::findFundamentalMat(cv::Mat(points1), cv::Mat(points2), inliers, CV_FM_RANSAC,
//                         distance, confidence); // confidence probability

//   // extract the surviving (inliers) matches
//   std::vector<uchar>::const_iterator itIn = inliers.begin();
//   std::vector<cv::DMatch>::const_iterator itM = matches.begin();
//   // for all matches
//   for (; itIn != inliers.end(); ++itIn, ++itM) {
//     if (*itIn) {
//       // it is a valid match
//       goodMatches.push_back(*itM);
//     }
//   }
// }


// Matching descriptor vectors using FLANN matcher
int flannMatcher(const cv::Mat descriptors_1, const cv::Mat descriptors_2, std::vector<cv::DMatch> &good_matches)
{
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches, final_1, final_2;

  matcher.match(descriptors_1, descriptors_2, matches);

  // -- Quick calculation of max and min distances between keypoints
  double max_dist = 0; double min_dist = 100;
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = matches[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }
  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  // -- Select good matches, distance <2*min_dist,
  // -- or a small arbitary value ( 0.02 ) in the event that min_dist is very
  // -- small

  for (int i = 0; i < descriptors_1.rows; i++) {

    if (matches[i].distance <= max(1.2 * min_dist, 0.02)) {
      good_matches.push_back(matches[i]);
    }
  }


  // matcher.match(descriptors_2, descriptors_1, matches);

  // max_dist = 0; min_dist = 100;
  // for (int i = 0; i < descriptors_2.rows; i++) {
  //   double dist = matches[i].distance;
  //   if (dist < min_dist) min_dist = dist;
  //   if (dist > max_dist) max_dist = dist;
  // }
  // printf("-- Max dist : %f \n", max_dist);
  // printf("-- Min dist : %f \n", min_dist);

  // for (int i = 0; i < descriptors_2.rows; i++) {

  //   if (matches[i].distance <= max(2 * min_dist, 0.02)) {
  //     final_2.push_back(matches[i]);
  //   }
  // }

  // symmetryTest(final_1, final_2, matches);

}

//Match descriptors based on knn FLANN
int flannKnn(const cv::Mat descriptors_1, const cv::Mat descriptors_2, std::vector<cv::DMatch> &good_matches)
{

  FlannBasedMatcher matcher;
  std::vector< vector<DMatch> > matches_1, matches_2;
  matcher.knnMatch(descriptors_1, descriptors_2, matches_1, 2, noArray(), false);

  matcher.knnMatch(descriptors_2, descriptors_1, matches_2, 2, noArray(), false);

  std::vector< DMatch > filtered_matches_1, filtered_matches_2;

  // Keep matches <0.8 (Lowe's paper), discard rest
  const float ratio = 0.76;
  for (int i = 0; i < matches_1.size(); ++i) {
    if (matches_1[i][0].distance < (ratio * matches_1[i][1].distance)) {
      printf("Dist 1 %f vs Dist 2 %f , Ratio %f\nn", matches_1[i][0].distance,
             matches_1[i][1].distance, matches_1[i][0].distance / matches_1[i][1].distance);
      filtered_matches_1.push_back(matches_1[i][0]);
    }
  }
  printf("--------------------------------------------------\n");
  for (int i = 0; i < matches_2.size(); ++i) {
    if (matches_2[i][0].distance < (ratio * matches_2[i][1].distance)) {
      // printf("Dist 1 %f vs Dist 2 %f , Ratio %f\nn", matches_2[i][0].distance, matches_2[i][1].distance, matches_2[i][0].distance / matches_2[i][1].distance);
      filtered_matches_2.push_back(matches_2[i][0]);
    }
  }
  printf("Set 1 %d Set 2 %d \n", filtered_matches_1.size(), filtered_matches_2.size());
  symmetryTest(filtered_matches_1, filtered_matches_2, good_matches);
}

int bffKnn(const cv::Mat descriptors_1, const cv::Mat descriptors_2, std::vector<cv::DMatch> &good_matches)
{

  BFMatcher matcher;
  std::vector< vector<DMatch> > matches_1, matches_2;
  matcher.knnMatch(descriptors_1, descriptors_2, matches_1, 2, noArray(), false);

  matcher.knnMatch(descriptors_2, descriptors_1, matches_2, 2, noArray(), false);

  std::vector< DMatch > filtered_matches_1, filtered_matches_2;

  // Keep matches <0.8 (Lowe's paper), discard rest
  float ratio = 0.81;
  for (int i = 0; i < matches_1.size(); ++i) {
    if (matches_1[i][0].distance < (ratio * matches_1[i][1].distance)) {
      // printf("Dist 1 - %f vs Dist 2 - %f , Ratio - %f\n", matches_1[i][0].distance,
      // matches_1[i][1].distance, matches_1[i][0].distance / matches_1[i][1].distance);
      filtered_matches_1.push_back(matches_1[i][0]);
    }
  }
  printf("-----------------------------------------------------\n");
  ratio = 0.8;
  for (int i = 0; i < matches_2.size(); ++i) {
    if (matches_2[i][0].distance < (ratio * matches_2[i][1].distance)) {
      // printf("Dist 1 - %f vs Dist 2 - %f , Ratio - %f\n", matches_2[i][0].distance,
      // matches_2[i][1].distance, matches_2[i][0].distance / matches_2[i][1].distance);
      filtered_matches_2.push_back(matches_2[i][0]);
    }
  }
  printf("Set 1 %d Set 2 %d \n", filtered_matches_1.size(), filtered_matches_2.size());
  symmetryTest(filtered_matches_1, filtered_matches_2, good_matches);
}

cv::Mat ransacTest(
  const std::vector<cv::DMatch> &matches,
  const std::vector<cv::KeyPoint> &keypoints1,
  const std::vector<cv::KeyPoint> &keypoints2,
  std::vector<cv::DMatch> &outMatches)
{
  // Convert keypoints into Point - Make it a function
  std::vector<cv::Point> points1, points2;
  for (std::vector<cv::DMatch>::
       const_iterator it = matches.begin();
       it != matches.end(); ++it) {
    // Get the position of left keypoints
    float x = keypoints1[it->queryIdx].pt.x;
    float y = keypoints1[it->queryIdx].pt.y;
    points1.push_back(cv::Point(x, y));
    // Get the position of right keypoints
    x = keypoints2[it->trainIdx].pt.x;
    y = keypoints2[it->trainIdx].pt.y;
    points2.push_back(cv::Point(x, y));
  }
  // Compute F matrix using RANSAC
  std::vector<uchar> inliers(points1.size(), 0);

  cv::Mat fundemental = cv::findFundamentalMat(
                          cv::Mat(points1), cv::Mat(points2), // matching points
                          inliers,                          // match status (inlier or outlier)
                          CV_FM_RANSAC, // RANSAC method
                          DISTANCE,                          // distance to epipolar line
                          CONFIDENCE); // confidence probability

  // extract the surviving (inliers) matches
  std::vector<uchar>::const_iterator
  itIn = inliers.begin();
  std::vector<cv::DMatch>::const_iterator
  itM = matches.begin();
  // for all matches
  for (; itIn != inliers.end(); ++itIn, ++itM) {
    if (*itIn) { // it is a valid match
      outMatches.push_back(*itM);
    }
  }
  // The F matrix will be recomputed with
  // all accepted matches
  // Convert keypoints into Point
  // for final F computation
  points1.clear();
  points2.clear();
  for (std::vector<cv::DMatch>::
       const_iterator it = outMatches.begin();
       it != outMatches.end(); ++it) {
    // Get the position of left keypoints 
    float x = keypoints1[it->queryIdx].pt.x;
    float y = keypoints1[it->queryIdx].pt.y;
    points1.push_back(cv::Point(x, y));
    // Get the position of right keypoints
    x = keypoints2[it->trainIdx].pt.x;
    y = keypoints2[it->trainIdx].pt.y;
    points2.push_back(cv::Point(x, y));
  }
  // Compute 8-point F from all accepted matches
  fundemental = cv::findFundamentalMat(
                  cv::Mat(points1), cv::Mat(points2), // matches
                  CV_FM_8POINT); // 8-point method
  cout << fundemental.type() << endl; 
  return fundemental;
}

cv::Mat findAffineMat(
  const std::vector<cv::DMatch> &matches,
  const std::vector<cv::KeyPoint> &keypoints1,
  const std::vector<cv::KeyPoint> &keypoints2)
{

  // Calculate the Affine LS Homography

  int **final_matches;

  final_matches = (int **)malloc(sizeof(int *)*matches.size());
  for (int i = 0; i < matches.size(); i++) {
    final_matches[i] = (int *)malloc(sizeof(int) * 4);
  }

  for (int i = 0; i < matches.size(); i++) {

    final_matches[i][0] = keypoints1[matches[i].queryIdx].pt.x;
    final_matches[i][1] = keypoints1[matches[i].queryIdx].pt.y;
    final_matches[i][2] = keypoints2[matches[i].trainIdx].pt.x;
    final_matches[i][3] = keypoints2[matches[i].trainIdx].pt.y;
  }

  float X[100];

  findAffine(matches.size(), final_matches, X);

  // printf("-----------------------------------------------------\n");
  // printf("\n[%f , %f , %f \n%f , %f , %f \n%f , %f , %f ]\n", X[0], X[1],
  //        X[2], X[3] , X[4], X[5], X[6], X[7], X[8]);

  Mat H = Mat(2, 3, CV_32FC1, &X);
  cout << H << endl;

  return H;
  
}


int main(int argc, char **argv)
{
  if (argc != 4)
  { readme(); return -1; }

  Mat img_1 = imread(argv[1]);
  if (!img_1.data) {
      std::cout << " --(!) Error reading : %s"<< argv[1] << std::endl;
      return -1;
    }
  Mat img_2 = imread(argv[2]);

  if (!img_2.data) {
    std::cout << " --(!) Error reading : %s"<< argv[2] << std::endl;
    return -1;
  }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = atoi(argv[3]);
  Ptr<SURF> detector = SURF::create();
  detector->setHessianThreshold(minHessian);
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;

  detector->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1);

  minHessian = minHessian / 2;
  detector->setHessianThreshold(minHessian);
  detector->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2);

  #if 0
  // -- Draw keypoints
  Mat img_keypoints_1; Mat img_keypoints_2;

  drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

  //-- Show detected (drawn) keypoints
  imshow("Keypoints 1", img_keypoints_1);
  imshow("Keypoints 2", img_keypoints_2);
  #endif

  // -- Perform Matching
  vector <DMatch> good_matches;
  bffKnn(descriptors_1, descriptors_2, good_matches);

  // -- Draw Matches
  Mat img_matches;

  // Mat mask = Mat::zeros(img_1.size(), CV_8U);  // type of mask is CV_8U
  // Mat roi(mask, cv::Rect(231,130,630,334));
  // roi = Scalar(255, 255, 255);
  // drawMatches(img_2, keypoints_2, img_1, keypoints_1, good_matches,
  //             img_matches, Scalar(0, 255, 0), Scalar(0, 0, 255),
  //             vector< char >(), DrawMatchesFlags::DEFAULT);


  drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches,
              img_matches, Scalar(0, 255, 0), Scalar(0, 0, 255),
              vector< char >(), DrawMatchesFlags::DEFAULT);

  printf("Found %d good matches\n", (int)good_matches.size());
  imshow("Good Matches & Object detection", img_matches);

  vector<DMatch> ransac_match;
  #if 0
  // Calculate the Affine LS Homography
  ransacTest(good_matches, keypoints_1, keypoints_2, ransac_match, DISTANCE, CONFIDENCE, 0.25);

  int **final_matches;

  final_matches = (int **)malloc(sizeof(int *)*ransac_match.size());
  for (int i = 0; i < ransac_match.size(); i++) {
    final_matches[i] = (int *)malloc(sizeof(int) * 4);
  }

  for (int i = 0; i < ransac_match.size(); i++) {

    final_matches[i][0] = keypoints_1[ransac_match[i].queryIdx].pt.x;
    final_matches[i][1] = keypoints_1[ransac_match[i].queryIdx].pt.y;
    final_matches[i][2] = keypoints_2[ransac_match[i].trainIdx].pt.x;
    final_matches[i][3] = keypoints_2[ransac_match[i].trainIdx].pt.y;
  }

  float X[100];

  findAffine(ransac_match.size(), final_matches, X);

  printf("Before RS %d After RS %d\n", good_matches.size(), ransac_match.size());

  drawMatches(img_1, keypoints_1, img_2, keypoints_2, ransac_match,
              img_matches, Scalar(0, 255, 0), Scalar(0, 0, 255),
              vector< char >(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  imshow("After ransac", img_matches);

  printf("\n[%f , %f , %f \n%f , %f , %f \n%f , %f , %f ]\n", X[0], X[1],
         X[2], X[3] , X[4], X[5], X[6], X[7], X[8]);

  Mat H = Mat(2, 3, CV_32FC1, &X);
  Mat timage;

  warpAffine(img_2, timage, H, Size(720 , 576), INTER_LINEAR,
             BORDER_CONSTANT, Scalar());
  imshow("Transformed", timage);

  // void ransacTest(const std::vector<cv::DMatch> matches, const std::vector<cv::KeyPoint> &keypoints1,
  //   const std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> &goodMatches, double distance,
  //   double confidence, double minInlierRatio)

  // warpPerspective(img_2, timage, tH, Size(720 , 576), INTER_LINEAR,
  //   BORDER_CONSTANT, Scalar());

  #else
  // -- Calculate Projective RANSAC

  Mat tmat = ransacTest(good_matches, keypoints_1, keypoints_2, ransac_match); 
  cout << tmat << endl;

  drawMatches(img_1, keypoints_1, img_2, keypoints_2, ransac_match,
              img_matches, Scalar(0, 255, 0), Scalar(0, 0, 255),
              vector< char >(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  imshow("After ransac", img_matches);


  Mat fimg;

  warpPerspective(img_2, fimg, tmat, Size(720 , 576), INTER_LINEAR,
                  BORDER_CONSTANT, Scalar());
  
  imshow("RANSAC", fimg);

  Mat affinemat = Mat::zeros(2,3,CV_32FC1);
  affinemat = findAffineMat(ransac_match,keypoints_1,keypoints_2);
  cout << affinemat << endl;

  // warpAffine(img_2, fimg, affinemat, Size(720 , 576), INTER_LINEAR,
  //                 BORDER_CONSTANT, Scalar());
  
  // imshow("Affine", fimg);


  #endif

  waitKey(0);

  return 0;
}

