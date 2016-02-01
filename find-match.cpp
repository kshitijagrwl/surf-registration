//*******************surf.cpp******************//
//********** SURF implementation in OpenCV*****//
//**loads video from webcam, grabs frames computes SURF keypoints and descriptors**//  //** and marks them**//

//****author: achu_wilson@rediffmail.com****//

#include <stdio.h>
#include  <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "affine2d.h"

#define MAX_MATCHES 100

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

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

//ransac test for stereo images
void ransacTest(const std::vector<cv::DMatch> matches, const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> &goodMatches, double distance, double confidence, double minInlierRatio)
{
  goodMatches.clear();
  // Convert keypoints into Point2f
  std::vector<cv::Point2f> points1, points2;
  for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it) {
    // Get the position of left keypoints
    float x = keypoints1[it->queryIdx].pt.x;
    float y = keypoints1[it->queryIdx].pt.y;
    points1.push_back(cv::Point2f(x, y));
    // Get the position of right keypoints
    x = keypoints2[it->trainIdx].pt.x;
    y = keypoints2[it->trainIdx].pt.y;
    points2.push_back(cv::Point2f(x, y));
  }
  // Compute F matrix using RANSAC
  std::vector<uchar> inliers(points1.size(), 0);
  cv::Mat fundemental = cv::findFundamentalMat(cv::Mat(points1), cv::Mat(points2), inliers, CV_FM_RANSAC, distance, confidence); // confidence probability
  // extract the surviving (inliers) matches
  std::vector<uchar>::const_iterator
  itIn = inliers.begin();
  std::vector<cv::DMatch>::const_iterator
  itM = matches.begin();
  // for all matches
  for (; itIn != inliers.end(); ++itIn, ++itM) {
    if (*itIn) {
      // it is a valid match
      goodMatches.push_back(*itM);
    }
  }
}

int main(int argc, char **argv)
{
  if (argc != 4)
  { readme(); return -1; }

  Mat img_1 = imread(argv[1], IMREAD_GRAYSCALE);
  Mat img_2 = imread(argv[2], IMREAD_GRAYSCALE);

  if (!img_1.data || !img_2.data) {
    // std::cout << " --(!) Error reading images " << std::endl;
    printf("Error reading images\n");
    return -1;
  }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = atoi(argv[3]);
  Ptr<SURF> detector = SURF::create();
  detector->setHessianThreshold(minHessian);
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;

  detector->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1);

  //Hessian made twice for IR
  minHessian = minHessian;
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
  #else

  //-- Step 2: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  
  // std::vector< DMatch > matches;
  // matcher.match(descriptors_1, descriptors_2, matches);

  // double max_dist = 0; double min_dist = 100;
  //-- Quick calculation of max and min distances between keypoints
  // for (int i = 0; i < descriptors_1.rows; i++) {
  //   double dist = matches[i].distance;
  //   if (dist < min_dist) min_dist = dist;
  //   if (dist > max_dist) max_dist = dist;
  // }
  // printf("-- Max dist : %f \n", max_dist);
  // printf("-- Min dist : %f \n", min_dist);

  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
  //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
  //-- small)
  // for (int i = 0; i < descriptors_1.rows; i++) {

  //   if (matches[i].distance <= max(2*min_dist, 0.02)) {
  //     good_matches.push_back(matches[i]);
  //   }
  // }

  std::vector< vector<DMatch> > matches_1, matches_2;
  matcher.knnMatch(descriptors_1, descriptors_2, matches_1, 2, noArray(), false);

  matcher.knnMatch(descriptors_2, descriptors_1, matches_2, 2, noArray(), false);

  std::vector< DMatch > filtered_matches_1, filtered_matches_2, good_matches;

  // Keep matches <0.8 (Lowe's paper), discard rest
  const float ratio = 0.80;
  for (int i = 0; i < matches_1.size(); ++i) {
    if (matches_1[i][0].distance < (ratio * matches_1[i][1].distance)) {
      printf("Dist 1 %f vs Dist 2 %f , Ratio %f\nn", matches_1[i][0].distance, matches_1[i][1].distance, matches_1[i][0].distance / matches_1[i][1].distance);
      filtered_matches_1.push_back(matches_1[i][0]);
    }
  }

  for (int i = 0; i < matches_2.size(); ++i) {
    if (matches_2[i][0].distance < (ratio * matches_2[i][1].distance)) {
      printf("Dist 1 %f vs Dist 2 %f , Ratio %f\nn", matches_2[i][0].distance, matches_2[i][1].distance, matches_2[i][0].distance / matches_2[i][1].distance);
      filtered_matches_2.push_back(matches_2[i][0]);
    }
  }

  symmetryTest(filtered_matches_1, filtered_matches_2, good_matches);

  //-- Draw only "good" matches
  Mat img_matches;

  Mat mask = Mat::zeros(img_1.size(), CV_8U);  // type of mask is CV_8U
  Mat roi(mask, cv::Rect(231,130,630,334));

  drawMatches(img_1, keypoints_1, img_2, keypoints_2,
              good_matches, img_matches, Scalar(0, 255, 0), Scalar(0,0,255),
              roi, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  printf("Found %d good matches\n", (int)good_matches.size());


  // Calculate the (Affine) Homography
  int **final_matches;

  final_matches = (int **)malloc(sizeof(int *)*good_matches.size());
  for (int i = 0; i < good_matches.size(); i++) {
    final_matches[i] = (int *)malloc(sizeof(int) * 4);
  }

  for (int i = 0; i < good_matches.size(); i++) {

    final_matches[i][0] = keypoints_1[good_matches[i].queryIdx].pt.x;
    final_matches[i][1] = keypoints_1[good_matches[i].queryIdx].pt.y;
    final_matches[i][2] = keypoints_2[good_matches[i].trainIdx].pt.x;
    final_matches[i][3] = keypoints_2[good_matches[i].trainIdx].pt.y;
  }

  float X[100];
  findAffine(good_matches.size(), final_matches, X);

  printf("\n[%f , %f , %f \n%f , %f , %f \n0.000000 0.000000 1.000000 ]\n", X[0], X[1], X[2], X[3] , X[4], X[5]);

  Mat H = Mat(2, 3, CV_32FC1, &X);
  Mat timage;

  // cvWarpAffine(&img_2, &timage, &H,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0));
  warpAffine(img_2, timage, H, Size(720 , 576), INTER_LINEAR, BORDER_CONSTANT, Scalar());
  imshow("Transformed", timage);
  //-- Show detected matches
  // std::vector<Point2f> obj;
  // std::vector<Point2f> scene;

  // for (int i = 0; i < (int)good_matches.size(); i++) {
  //   printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
  //   obj.push_back(keypoints_2[ good_matches[i].queryIdx ].pt);
  //   scene.push_back(keypoints_1[ good_matches[i].trainIdx ].pt);
  // }

  // Mat H = findHomography(obj, scene, RANSAC);

  // std::vector<Point2f> obj_corners(4);
  // obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img_2.cols, 0);
  // obj_corners[2] = cvPoint(img_2.cols, img_2.rows); obj_corners[3] = cvPoint(0, img_2.rows);
  // std::vector<Point2f> scene_corners(4);
  // perspectiveTransform(obj_corners, scene_corners, H);

  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
  // line( img_matches, scene_corners[0] + Point2f( img_2.cols, 0), scene_corners[1] + Point2f( img_2.cols, 0), Scalar(0, 255, 0), 4 );
  // line( img_matches, scene_corners[1] + Point2f( img_2.cols, 0), scene_corners[2] + Point2f( img_2.cols, 0), Scalar( 0, 255, 0), 4 );
  // line( img_matches, scene_corners[2] + Point2f( img_2.cols, 0), scene_corners[3] + Point2f( img_2.cols, 0), Scalar( 0, 255, 0), 4 );
  // line( img_matches, scene_corners[3] + Point2f( img_2.cols, 0), scene_corners[0] + Point2f( img_2.cols, 0), Scalar( 0, 255, 0), 4 );
  //-- Show detected matches

  imshow("Good Matches & Object detection", img_matches);

  #endif

  waitKey(0);

  return 0;
}

