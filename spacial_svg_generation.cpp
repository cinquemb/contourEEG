#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <map>
#include <cmath>
#include <iterator>
#include <string>
#include <streambuf>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>

#include <librsvg/rsvg.h>
#include <Magick++.h>

#include <magics/magics_api.h>
#include <json/json.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

const long double PI = acos(-1.0L);
int max_leads = 128;

//offset: 2 for new files, 0 for old files
const int offset_node = 0;
//factor to convert biosemi values into uv
double biosemi_microvoltage_factor = 8192;

//skip lines in mining files
int skip_count = 100;

//samples to create
int max_samples = 90;

//maxnumber of lines to mine in file
int max_lines = skip_count * max_samples;

//should speed up computation time at the expense of the number of x/y points
int width_factor = 7;

//number of colors
int contouring_bins = 15;

std::vector<double> x_vec;
std::vector<double> y_vec;
//temp voltage
std::vector<double> z_vec;

std::vector<std::vector<std::vector<double>  > > point_combinations;
std::vector<std::vector<double>  > temp_point_combinations;
std::vector<std::vector<std::vector<double>>> plane_triangles;
std::map<std::string, int> z_point_map;

std::vector<std::string> string_split(std::string const &input) { 
    std::istringstream buffer(input);
    std::vector<std::string> ret((std::istream_iterator<std::string>(buffer)), 
                                 std::istream_iterator<std::string>());
    return ret;
}

struct RetrieveKey
{
    template <typename T>
    typename T::first_type operator()(T keyValuePair) const
    {
        return keyValuePair.first;
    }
};


void print_vector(std::vector<double>& v){
	int vector_len = v.size();
	for(int i = vector_len - 1;i >= 0; i--){
        if(i > 0)
            std::cout << v[i] << ',';
        else
            std::cout << v[i] << '\n';
    }
		
    std::cout << '\n';
}

template <class T1, class T2, class Pred = std::less<T2> >
struct sort_pair_second {
    bool operator()(const std::pair<T1,T2>&left, const std::pair<T1,T2>&right) {
        Pred p;
        return p(left.second, right.second);
    }
};

bool point_sort_delaunay_triangulation(std::vector<double>& i, std::vector<double>& j){
	if(i[0] == j[0])
		return (i[1] < j[1]);
	else
		return (i[0] < j[0]);
}

bool is_point_in_triange(std::vector<std::vector<double>  >& tri, std::vector<double>& point){
	for(int i=0; i<tri.size();i++){
		if(tri[i][0] == point[0] && tri[i][1] == point[1])
			return true;
	}
	return false;
}

bool sort_find_bottom_point_left(std::vector<double>& i, std::vector<double>& j){
	if(i[1] == j[1])
		return (i[0] > j[0]);
	else
		return (i[1] < j[1]);
}

bool sort_find_bottom_point_right(std::vector<double>& i, std::vector<double>& j){
	if(i[1] == j[1])
		return (i[0] < j[0]);
	else
		return (i[1] < j[1]);
}
double compute_angle_left(std::vector<std::vector<double>  >& tri){
	std::vector<double> u = {tri[2][0]-tri[0][0],tri[2][1]-tri[0][1]};
	std::vector<double> v = {tri[1][0]-tri[0][0],tri[1][1]-tri[0][1]};

	double dot_u_v = (u[0]*v[0]) + (u[1]*v[1]);
	double mag_u_v = std::sqrt(std::pow(u[0],2) + std::pow(u[1],2)) * std::sqrt(std::pow(v[0],2) + std::pow(v[1],2));
	double costheta = dot_u_v/mag_u_v;
	//std::cout << "dot_u_v: "<< dot_u_v<< std::endl;
	//std::cout << "mag_u_v: "<< mag_u_v<< std::endl;
	//std::cout << "dot_u_v/mag_u_v: "<< (dot_u_v/mag_u_v) << std::endl;
	return (std::acos(costheta) * (180/PI));
}

double compute_angle_right(std::vector<std::vector<double>  >& tri){
	std::vector<double> u = {tri[0][0]-tri[2][0],tri[0][1]-tri[2][1]};
	std::vector<double> v = {tri[0][0]-tri[1][0],tri[0][1]-tri[1][1]};

	double dot_u_v = (u[0]*v[0]) + (u[1]*v[1]);
	double mag_u_v = std::sqrt(std::pow(u[0],2) + std::pow(u[1],2)) * std::sqrt(std::pow(v[0],2) + std::pow(v[1],2));
	double costheta = dot_u_v/mag_u_v;
	//std::cout << "dot_u_v: "<< dot_u_v<< std::endl;
	//std::cout << "mag_u_v: "<< mag_u_v<< std::endl;
	//std::cout << "dot_u_v/mag_u_v: "<< (dot_u_v/mag_u_v) << std::endl;
	return (std::acos(costheta) * (180/PI));
}

bool is_ccw(std::vector<std::vector<double>  >& tri){
	return ( ( tri[1][0]-tri[0][0] )*( tri[1][1]+tri[0][1] ) + ( tri[2][0]-tri[1][0] )*( tri[2][1]+tri[1][1] ) + ( tri[0][0]-tri[2][0] )*( tri[0][1]+tri[2][1] ) ) < 0;
}

bool circumcircle_contains_point(std::vector<std::vector<double>  >& tri, std::vector<double>& point){
	Eigen::MatrixXf m(4,4);
	std::vector<std::vector<double>  > points(tri.begin(), tri.end());
	points.push_back(point);
	for(int i=0; i<4; i++){
		m(i,0) = points[i][0];
		m(i,1) = points[i][1];
		m(i,2) = std::pow(points[i][0],2) + std::pow(points[i][1],2);
		m(i,3) = 1;
	}
    return (m.determinant() > 0);
}

bool convex_hull_determinant(std::vector<std::vector<double>  > points){
	/*
	Calc. determinant of a special matrix with three 2D points.

    The sign, "-" or "+", determines the side, right or left,
    respectivly, on which the point r lies, when measured against
    a directed vector from p to q.
    */

    std::vector<double> p = points[0];
    std::vector<double> q = points[1];
    std::vector<double> r = points[2];

    // We use Sarrus' Rule to calculate the determinant.
    // (could also use the Numeric package...)

    double sum1 = q[0]*r[1] + p[0]*q[1] + r[0]*p[1];
    double sum2 = q[0]*p[1] + r[0]*q[1] + p[0]*r[1];
    return (sum1 - sum2) < 0;
}
    

std::vector<std::vector<double>  > convex_hull_points(std::vector<std::vector<double>  > points){
	std::vector<std::vector<double>  > c_points_upper;
	std::vector<std::vector<double>  > c_points_lower;
	std::vector<std::vector<double>  > hull_points_lower;

	// Get a local list copy of the points and sort them lexically.
	// sort points by x, if x equivalnt, sory by y;
	if(points.size() < 4)
		return points;

	std::sort(points.begin(), points.end(), point_sort_delaunay_triangulation);

    // Build upper half of the hull.
    c_points_upper.push_back(points[0]);
    c_points_upper.push_back(points[1]);

    //std::cout << "before loop" << std::endl;
    for(int i = 2; i< points.size(); i++){
    	c_points_upper.push_back(points[i]);
    	std::vector<std::vector<double>  > temp_point_vec(c_points_upper.end()-3, c_points_upper.end());
    	while(c_points_upper.size() > 2 and !convex_hull_determinant(temp_point_vec))
    		c_points_upper.erase(c_points_upper.end()-2);
    }
    //std::cout << "after loop" << std::endl;
	

    // Build lower half of the hull.
    std::reverse(points.begin(), points.end());
    c_points_lower.push_back(points[0]);
    c_points_lower.push_back(points[1]);

    for(int i = 2; i< points.size(); i++){
    	c_points_lower.push_back(points[i]);
    	std::vector<std::vector<double>  > temp_point_vec(c_points_lower.end() -3, c_points_lower.end());
    	while(c_points_lower.size() > 2 and !convex_hull_determinant(temp_point_vec))
    		c_points_lower.erase(c_points_lower.end() -2);
    }

    //std::cout << "removing dupes" << "\n";

    // Remove duplicates
    hull_points_lower.insert(hull_points_lower.end(), c_points_lower.begin()+1, c_points_lower.end()-1);

    // Concatenate both halfs and return.
    c_points_upper.insert(c_points_upper.end(), hull_points_lower.begin(), hull_points_lower.end());
    return c_points_upper;
}


double area_of_triangle(double& x1, double& y1, double& x2, double& y2, double& x3, double& y3){
   return std::abs((x1*(y2-y3) + x2*(y3-y1)+ x3*(y1-y2))/2.0);
}

bool is_point_inside_triangle(std::vector<std::vector<double>  >& tri, double& x, double& y){   
	double x1 = tri[0][0]; double x2 = tri[1][0]; double x3 = tri[2][0]; double y1 = tri[0][1]; double y2 = tri[1][1]; double y3 = tri[2][1];
	/* Calculate area of triangle ABC */
	double A = area_of_triangle(x1, y1, x2, y2, x3, y3);
	/* Calculate area of triangle PBC */   
	double A1 = area_of_triangle(x, y, x2, y2, x3, y3);
	/* Calculate area of triangle PAC */   
	double A2 = area_of_triangle(x1, y1, x, y, x3, y3);
	/* Calculate area of triangle PAB */    
	double A3 = area_of_triangle(x1, y1, x2, y2, x, y);
	/* Check if sum of A1, A2 and A3 is same as A */ 
	return (A == A1 + A2 + A3);
}
    

std::vector<std::string> make_color_gradeint(double freq1, double freq2, double freq3, double phase1, double phase2, double phase3, int count){
	double center=128;
	double width=127;
	std::vector<std::string> colors;
	for(int i=0;i<count;i++){
		//"rgb(0.95,0.65,0.95)"
		float red = ((float)((std::sin(freq1*i + phase1) * width) + center))/255;
        float grn = ((float)((std::sin(freq2*i + phase2) * width) + center))/255;
        float blu = ((float)((std::sin(freq3*i + phase3) * width) + center))/255;

        std::string color_string = "RGB(" + std::to_string(red) + "," + std::to_string(grn) + "," + std::to_string(blu) + ")";
        colors.push_back(color_string);
	}
	return colors;
}

double calculate_stdv(std::vector<double>& v, double& _mean){
    double stdv_exp = 2;
    double _accum = 0.0;
    std::for_each (v.begin(), v.end(), [&](const double d) {
        _accum += std::pow((d - _mean), stdv_exp);
    });
    double _stdev = std::sqrt(_accum/ ((double)v.size()-1));
    return _stdev;
}

std::map<int, std::vector<double> > generate_spacial_map(){
	std::string map_file = "xy.txt";
	std::map<int, std::vector<double> > lead_map_x_y;

    std::string line;
    std::ifstream in(map_file.c_str());

    int lead_count = 0;
    if (!in.is_open()) return lead_map_x_y;

    while (std::getline(in,line)){
        if(line.size() > 1){
        	std::vector<std::string> data_mapping_string = string_split(line);
        	std::vector<double> temp_xy;
        	
        	for(int i=0;i< data_mapping_string.size();i++){
        		double coord_val;
        		std::istringstream(data_mapping_string[i]) >> coord_val;
        		temp_xy.push_back(coord_val);
        	}

        	std::cout << "x: " << temp_xy[0] << " y: " << temp_xy[1] << std::endl;
        	lead_map_x_y[lead_count] = temp_xy;
        	++lead_count;
        }
            
    }
    return lead_map_x_y;
}

Magick::Image OpenCV2Magick(cv::Mat opencvImage){
	// to convert opencv image to imagemagick to render svg
	Magick::Image mgk(opencvImage.rows, opencvImage.cols, "BGR", Magick::CharPixel, (char *)opencvImage.data);
	return mgk;
}

cv::Mat create_image(){
	int width = 500;
	int height = 500;
	cv::Mat test_image(width,height,CV_8UC4);
	
	int black = 255;

	test_image.rows = height;
	test_image.cols = width;

	for(int y=0; y< height; y++){
		for(int x=0; x < width; x++){
			test_image.at<cv::Vec3b>(y,x)[0] = black;
			test_image.at<cv::Vec3b>(y,x)[1] = 0;
			test_image.at<cv::Vec3b>(y,x)[2] = black;
		}
	}

	std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    return test_image;
}

double gcp(int index, std::vector<double>& vector){
	return vector[index-1];
}

double b_spline(double& x, std::vector<double>& control_points, int power, int current_control_point){
	if(power == 1){
		if(x >= gcp(current_control_point, control_points) && x < gcp(current_control_point+1, control_points))
			return 1;
		else
			return 0;
	}

	double alpha;

	double numeratorA = (x - gcp(current_control_point, control_points));
	double denominatorA = (gcp(current_control_point+power-1, control_points)  - gcp(current_control_point, control_points));

	double numeratorB = (gcp(current_control_point+power, control_points) - x);
	double denominatorB = (gcp(current_control_point+power, control_points) - gcp(current_control_point+1, control_points));

	if(current_control_point-1 < 0)
		std::cout << "current_control_point: " << current_control_point << std::endl;
	if(current_control_point+power < 0 || control_points.size() < current_control_point+power )
		std::cout << "current_control_point+power-1 " << current_control_point+power  << std::endl;

	double sub_weight_a = 0;
	double sub_weight_b = 0;
	double zero = 0.0;
		
	if (std::abs(denominatorA) > zero){
		alpha = numeratorA/denominatorA;

		if(std::numeric_limits<int>::max() < std::abs(alpha))
			sub_weight_a = 0;
		else
			sub_weight_a = (alpha * b_spline(x, control_points, power-1, current_control_point));
	}
		

	if(std::abs(denominatorB) > zero){
		alpha = numeratorB/denominatorB;
		//std::cout << "sub_weight_b alpha: " << alpha << std::endl;
		if(std::numeric_limits<int>::max() < std::abs(alpha))
			sub_weight_b = 0;
		else
			sub_weight_b = (alpha * b_spline(x, control_points, power-1, current_control_point+1));
	}

	//std::cout << "sub_weight_a: " << sub_weight_a << std::endl;
	//std::cout << "sub_weight_b: " << sub_weight_b << std::endl;
	return sub_weight_a + sub_weight_b;
}

double nearest_neighbour_weighted_interpolation(double& x_value, double& y_value, std::vector<double>& raw_eeg_voltage, double& min_z, double& range_z, int& power){
	//http://paulbourke.net/miscellaneous/interpolation/
	double denominator_sum = 0.0;
	double numerator_sum = 0.0;
	for(int i=0; i<max_leads;i++){

		if(x_value == x_vec[i] && y_value == y_vec[i]){
			return raw_eeg_voltage[i];
		}else{
			double x_ = x_value - x_vec[i];
			double y_ = y_value - y_vec[i];
			double real_p = ((double)power)/((double)2);
			double tmp_denominator = std::pow((std::pow(x_,2) +  std::pow(y_,2)), real_p);

			if(tmp_denominator == 0){
				denominator_sum += 0;
				numerator_sum += 0;
			}else{
				denominator_sum += raw_eeg_voltage[i]/tmp_denominator;
				numerator_sum += 1/tmp_denominator;
			}
		}		
	}

	if(denominator_sum ==0)
		return 0;
	else
		return numerator_sum/denominator_sum;
}

std::vector<std::pair<int,double> > soreted_voltage_indexes(std::map<int, std::vector<double> >& brain_spacial_map, double& x_value, double& y_value){
	double min_value = std::numeric_limits<int>::max();
	int min_index = 0;
	std::vector<std::pair<int,double> > index_distance_pairs;
	for(int i=0; i<max_leads;i++){
		double x_ = x_value - brain_spacial_map[i][0];
		double y_ = y_value - brain_spacial_map[i][1];

		double tmp_distance = std::sqrt(std::pow(x_,2) +  std::pow(y_,2));
		if(min_value > tmp_distance){
			min_value = tmp_distance;
			min_index = i;
		}
		std::pair<int,double> tmp_pair = std::make_pair(i,tmp_distance);
		index_distance_pairs.push_back(tmp_pair);
	}

	std::sort(index_distance_pairs.begin(), index_distance_pairs.end(), sort_pair_second<int,double>());
	return index_distance_pairs;
}

std::vector<double> calculate_coeffecients_b_spline(int& power, int& coefficient_length, std::vector<double>& control_points, std::vector<double>& sample_points){
	std::vector<double> _coeff(coefficient_length,0);
	for(int i=1; i<coefficient_length; i++){
		double weight = 0;
		int control_vector_size = control_points.size()-2;
		for(int j = 1;j < control_vector_size; j++){
			weight += b_spline(sample_points[i], control_points, power, j);
			
		}
		_coeff[i] = weight;
	}
	return _coeff;
}
/*
void compute_merge(std::vector<std::vector<double>  >&left_hand_points, std::vector<std::vector<double>  >&right_hand_points, std::map<std::vector<std::vector<double>  >, int>& triangle_mapping)
	*/
void compute_merge(std::vector<std::vector<double>  >&right_hand_points, std::map<std::vector<std::vector<double>  >, int>& triangle_mapping){


	int left_start_index = 0;
	int right_start_index = 0;
	std::vector<std::vector<double>  > left_temp_tri;
	std::vector<double> temp_left_cand;
	std::vector<std::vector<double>  > right_temp_tri;
	std::vector<double> temp_right_cand;
	
	while (1){
		std::vector<std::vector<double>> old_left_hand_points;
		std::vector<std::vector<std::vector<double>>> tmp_triangles;
		std::map<std::vector<double>, int> edge_points_map;
		// Retrieve all keys which are the triangles
		transform(triangle_mapping.begin(), triangle_mapping.end(), back_inserter(tmp_triangles), RetrieveKey());
		//std::cout << "triangle_mapping: " <<  triangle_mapping.size()<< std::endl;
		//for each triangle, sort points and place in map
		for(int i=0;i<tmp_triangles.size();i++){
			for(int j=0; j< tmp_triangles[i].size(); j++){
				edge_points_map[tmp_triangles[i][j]] = 1;
			}
		}
		

		// Retrieve all keys which are the lefthand points
		transform(edge_points_map.begin(), edge_points_map.end(), back_inserter(old_left_hand_points), RetrieveKey());
		
		//old_left_hand_points.insert(old_left_hand_points.end(), left_hand_points.begin(), left_hand_points.end());
		
		//std::cout << "before convex_hull_points" << std::endl;
		std::vector<std::vector<double>  > hull_points_left = convex_hull_points(old_left_hand_points);
		std::sort(hull_points_left.begin(), hull_points_left.end(), sort_find_bottom_point_right);
		std::cout << "old_left_hand_points.size(): " << old_left_hand_points.size() << std::endl;

		int left_candidate = 0;
		int right_candidate = 0;

		for(int i=left_start_index;i< hull_points_left.size()-1;i++){
			if(i+2 <= hull_points_left.size()-1){
				//base LR edge
				//first candidate, second candidate (to check if inside triangle of first)
				left_temp_tri = {hull_points_left[i], hull_points_left[i+1], right_hand_points[right_start_index]};

				std::cout << "compute_angle_left(left_temp_tri): " << compute_angle_left(left_temp_tri) << std::endl;
				if(compute_angle_left(left_temp_tri) < 180){
					if(!circumcircle_contains_point(left_temp_tri, hull_points_left[i+2])){
						temp_left_cand = hull_points_left[i+1];
						left_candidate = 1;
						break;
					}else{
						//delete triangle left_temp_tri from map if in map
						std::sort(left_temp_tri.begin(), left_temp_tri.end(), point_sort_delaunay_triangulation);
						if (triangle_mapping.count(left_temp_tri)>0){
							triangle_mapping.erase(left_temp_tri);
						}	
					}
				}

			}else{
				//no more candidates for left
				break;
			}
		}

		for(int i=right_start_index;i< right_hand_points.size()-1;i++){
			if(i+2 <= right_hand_points.size()-1){
				//base LR edge
				//first candidate, second candidate (to check if inside triangle of first)
				right_temp_tri = {right_hand_points[i], right_hand_points[i+1], hull_points_left[left_start_index]};

				std::cout << "compute_angle_right(right_temp_tri): " << compute_angle_right(right_temp_tri) << std::endl;
				if(compute_angle_right(right_temp_tri) < 180){
					if(!circumcircle_contains_point(right_temp_tri, right_hand_points[i+2])){
						temp_right_cand = right_hand_points[i+1];
						right_candidate = 1;
						break;
					}else{
						//delete triangle left_temp_tri from map if in map
						std::sort(right_temp_tri.begin(), right_temp_tri.end(), point_sort_delaunay_triangulation);
						if (triangle_mapping.count(right_temp_tri)>0){
							triangle_mapping.erase(right_temp_tri);
						}	
					}
				}
			}else{
				//no more candidates for right
				break;
			}
		}

		if(right_candidate == 1 && left_candidate == 1){

			if(!circumcircle_contains_point(left_temp_tri, temp_right_cand)){
				triangle_mapping[left_temp_tri] = 1;
				++left_start_index;
			}else{
				triangle_mapping[right_temp_tri] = 1;
				++right_start_index;
			}

		}else if(right_candidate == 1 && left_candidate == 0){
			triangle_mapping[right_temp_tri] = 1;
			++right_start_index;
		}else if(right_candidate == 0 && left_candidate == 1){
			triangle_mapping[left_temp_tri] = 1;
			++left_start_index;
		}else{
			break;
		}
	}
}

void combinations(int offset, int k, std::vector<std::vector<double> >& points) {
  if (k == 0) {
  	point_combinations.push_back(temp_point_combinations);
    return;
  }
  for (int i = offset; i <= points.size() - k; ++i) {
    temp_point_combinations.push_back(points[i]);
    combinations(i+1, k-1, points);
    temp_point_combinations.pop_back();
  }
}

std::vector<std::vector<std::vector<double>  > > compute_triangles_from_delaunay_triangulation_divide_and_conquer(std::vector<std::vector<double> > v){
	int range_segments = 2;
	int max_points = v.size();
	std::vector<std::vector<double> > temp_segments_cache;
	int max_segments = (int)std::ceil(max_points/range_segments);
	// sort points by x, if x equivalnt, sory by y;
	std::sort(v.begin(), v.end(), point_sort_delaunay_triangulation);

	// triangles
	std::vector<std::vector<std::vector<double>  > > triangles;

	//segmented triangles init
	std::vector<std::vector<std::vector<double>  > > segemented_triangles_init;

	//temp segmented triangles
	std::vector<std::vector<std::vector<double>  > > temp_segements_triangles;

	std::map<std::vector<std::vector<double>  >, int> map_triangles;

	int init_segment = 0;

	for(int i = 0; i <max_points; i++){
		if(i % 3 == 0 && i != 0){
			//std::cout << "temp_segments: " << temp_segments_cache.size() << std::endl;
			//push back temp segment onto segments
			segemented_triangles_init.push_back(temp_segments_cache);
			temp_segments_cache.clear();
			temp_segments_cache.push_back(v[i]);
		}else{
			temp_segments_cache.push_back(v[i]);
		}
	}

	
	if(temp_segments_cache.size() > 0){
		//std::cout << "temp_segments: " << temp_segments_cache.size() << std::endl;
		segemented_triangles_init.push_back(temp_segments_cache);
		temp_segments_cache.clear();
	}

	//iterate over segments and when combination of segments is two, merge segemnts (compute triangles and push back new onto triangle vector)
	int total = 0;
	map_triangles[segemented_triangles_init[0]] =1;
	//temp_segements_triangles.push_back(segemented_triangles_init[0]);
	for(int i =1; i<segemented_triangles_init.size();i++){
		/*
		if(i % 2 == 0){
			std::sort(temp_segements_triangles[0].begin(),temp_segements_triangles[0].end(), sort_find_bottom_point_left);
			std::sort(temp_segements_triangles[1].begin(),temp_segements_triangles[1].end(), sort_find_bottom_point_right);
				compute merge and delete triangles from map if LL/RR are violating
				return new triangles, store triangles in map of triangles
				push back temp segment onto segments
			compute_merge(temp_segements_triangles[0], temp_segements_triangles[1], map_triangles);
			// Retrieve all keys which are the triangles
			transform(map_triangles.begin(), map_triangles.end(), back_inserter(triangles), RetrieveKey());
			total += map_triangles.size();
			std::cout << "map_triangles: " << map_triangles.size() << std::endl;
			std::cout << "total: " << total << std::endl;

			temp_segements_triangles.clear();
			temp_segements_triangles.push_back(segemented_triangles_init[i]);
		}else{
			temp_segements_triangles.push_back(segemented_triangles_init[i]);
		}*/

		std::sort(segemented_triangles_init[i].begin(),segemented_triangles_init[i].end(), sort_find_bottom_point_left);
		compute_merge(segemented_triangles_init[i], map_triangles);
		// Retrieve all keys which are the triangles
		
		total += map_triangles.size();
		std::cout << "map_triangles: " << map_triangles.size() << std::endl;
		std::cout << "total: " << total << std::endl;
	}
	// Retrieve all keys which are the triangles
	//transform(map_triangles.begin(), map_triangles.end(), back_inserter(triangles), RetrieveKey());
	transform(map_triangles.begin(), map_triangles.end(), back_inserter(triangles), RetrieveKey());
	return triangles;
}

std::vector<std::vector<std::vector<double>  > > compute_triangles_from_delaunay_triangulation_brute_force(std::vector<std::vector<double> > v){

	std::vector<std::vector<std::vector<double>  > > _triangles;
	combinations(0,3,v);

	for(int i=0; i<point_combinations.size();i++){
		bool triangle_empty = true;

		for(int j=0;j<v.size(); j++){

			if(is_point_in_triange(point_combinations[i], v[j]))
				continue;

			if(is_ccw(point_combinations[i]) == circumcircle_contains_point(point_combinations[i], v[j])){
				triangle_empty=false;
				break;
			}
			
		}
		if(triangle_empty)
			_triangles.push_back(point_combinations[i]);
	}

	return _triangles;
}

std::vector<double> compute_plane_coeff(std::vector<std::vector<double> >& tri){
	std::vector<double> a = tri[0]; std::vector<double> b = tri[1]; std::vector<double> c = tri[2];
	//length of four, ax +by + cz = d, vector = a,b,c,d
	std::vector<double> a_v = {a[0]-c[1], a[1]-c[1], a[2]-c[2]};
	std::vector<double> b_v = {b[0]-c[1], b[1]-c[1], b[2]-c[2]};
	std::vector<double> a_b_c_d_vector = {(a_v[1]*b_v[2]) - (a_v[2]*b_v[1]), (a_v[2]*b_v[0]) + (a_v[0]*b_v[2]), (a_v[0]*b_v[1]) + (a_v[1]*b_v[0])};
	double d = a_b_c_d_vector[0]*c[0] + a_b_c_d_vector[1]*c[1] + a_b_c_d_vector[2]*c[2];
	a_b_c_d_vector.push_back(d);

	return a_b_c_d_vector;
}

double compute_z_value_from_plane(std::vector<double>& plane, double& x, double& y){
	if(plane[2] == 0)
		return 0.0;
	return  (plane[3] - plane[0]*x - plane[1]*y)/(plane[2]);
}

double find_z_value_on_trianglar_mesh(double& x,  double&y, std::vector<double>& raw_eeg_values_vector){
	double z_value = 0.0;
	for(int i=0; i<plane_triangles.size();i++){
		//std::cout << "trying find point in triangles" << "\n";
		//print_vector(plane_triangles[i][0]);
		//print_vector(plane_triangles[i][1]);
		//print_vector(plane_triangles[i][2]);
		if(is_point_inside_triangle(plane_triangles[i], x, y)){
			//plane_triangles[i] need to find the z values for each point
			//std::cout << "in point found z" << "\n";
			std::vector<double> p1 = plane_triangles[i][0]; p1.push_back(raw_eeg_values_vector[z_point_map[std::to_string(p1[0]) + "_" + std::to_string(p1[1])]]);
			std::vector<double> p2 = plane_triangles[i][1]; p2.push_back(raw_eeg_values_vector[z_point_map[std::to_string(p2[0]) + "_" + std::to_string(p2[1])]]);
			std::vector<double> p3 = plane_triangles[i][2]; p3.push_back(raw_eeg_values_vector[z_point_map[std::to_string(p3[0]) + "_" + std::to_string(p3[1])]]);
			//std::cout << "after point mods found z" << "\n";

			std::vector<std::vector<double> > plane_points = {p1, p2, p3};
			std::vector<double> plane_coeff = compute_plane_coeff(plane_points);
			z_value = compute_z_value_from_plane(plane_coeff, x, y);

			break;
		}
	}
	return z_value;
}

double weighted_z_mean(int& nearest_points, double& numerical_edge_factor, double& min_z, double& range_z, std::vector<double>& raw_eeg_voltage, std::vector<std::pair<int,double> >& nearest_voltage_indexes){
	double z_value;
	double weight_sum = 0;
	double raw_z_value = 0.0;
	std::vector<double> temp_raw_values;
	std::vector<double> temp_raw_values_weights;
	for(int l=0; l<nearest_points;l++){
		double temp_raw_value = raw_eeg_voltage[nearest_voltage_indexes[l].first];
		temp_raw_values.push_back(temp_raw_value);
		double temp_raw_value_weight = nearest_voltage_indexes[l].second;
		temp_raw_values_weights.push_back(temp_raw_value_weight);
		weight_sum += temp_raw_value_weight;
	}
	for(int l=0; l<nearest_points;l++){
		raw_z_value += ((((temp_raw_values[l] - min_z)/range_z)/numerical_edge_factor) * temp_raw_values_weights[l]);
	}
	z_value = raw_z_value/weight_sum;
	return z_value;
}

double inner_kernal(double& mean, double& stdv, double& value){
	double inner_kernal = std::exp(-( std::pow(value-mean,2)/(2 *std::pow(stdv,2)) ));
	return inner_kernal;
}

void generate_magics_plot(std::string& svg_file_name, double& min_x, double&max_x, double& min_y, double& max_y, double*& gradientized_voltages, int& y_amount_dy, int& x_amount_dx, double*& contour_level_list, int& num_contouring_bins, const char**& contour_shade_colour_list_const, double& min_contour_shade_level, double& max_contour_shade_level){

	double page_y_scale_ratio = (max_y-min_y)/(max_x-min_x);
	double x_page_length = 5;
	double y_page_length = page_y_scale_ratio * x_page_length;

	mag_new("super_page");
	mag_setc("page_frame","off");

	mag_setr("page_x_length", x_page_length);
	mag_setr("page_y_length", y_page_length);
	mag_setr("super_page_x_length", x_page_length);
	mag_setr("super_page_y_length", y_page_length);
	mag_setc("page_id_line", "off");

	//const char *formats[2] = {"png", "svg"};
    //mag_set1c("output_formats", formats, 2);
    //mag_setc ("output_name",    "multiformat");
	mag_setc("output_format","svg");
	//mag_setc("output_format","png");
	mag_setc("output_name",svg_file_name.c_str());

	
	// Area specification

	mag_setc("subpage_map_projection","cartesian");
	mag_setc("subpage_x_axis_type","regular");
	mag_setc("subpage_y_axis_type","regular");

	mag_setr("subpage_x_min", min_x);
	mag_setr("subpage_x_max", max_x);
	mag_setr("subpage_y_min", min_y);
	mag_setr("subpage_y_max", max_y);
	mag_setc("subpage_frame","off");
	mag_setc("map_coastline","off");

	//Vertical axis
	mag_setc("axis_orientation","vertical");
	mag_setc("axis_grid","off");
	//mag_setc("axis_grid_colour","grey");
	//mag_seti("axis_grid_thickness",1);
	//mag_setc("axis_grid_line_style","dot");
	mag_setc("axis_line", "off");
	mag_setc("axis_tick", "off");
	mag_setc("axis_tick_label", "off");
	mag_axis();

	//#Horizontal axis
	mag_setc("axis_orientation","horizontal");
	mag_setc("axis_grid","off");
	//mag_setc("axis_grid_colour","grey");
	//mag_seti("axis_grid_thickness",1);
	//mag_setc("axis_grid_line_style","dot");
	mag_setc("axis_line", "off");
	mag_setc("axis_tick", "off");
	mag_setc("axis_tick_label", "off");
	mag_axis();

	// load the data
	mag_set2r("input_field", gradientized_voltages, y_amount_dy, x_amount_dx);
	mag_setr("input_field_initial_x", min_x);
	mag_setr("input_field_final_x", max_x);
	mag_setr("input_field_initial_y", min_y);
	mag_setr("input_field_final_y", max_y);
	
	mag_import();

	//define the contouring parameters

	mag_setc("contour_shade","on");
	mag_setc("legend","of");
	mag_setc("contour_shade_method","area_fill");
	mag_setc("legend_display_type","continuous");
	mag_setc("contour", "off");
	mag_setc("contour_level_selection_type","level_list");
	mag_setc("contour_shade_colour_method", "list");
	mag_setc("contour_shade","on");
	mag_set1r("contour_level_list", contour_level_list, num_contouring_bins);
	mag_setc("contour_hilo", "off");
	mag_setc("contour_highlight", "off");
	mag_setc("contour_label","off");
	mag_set1c("contour_shade_colour_list",contour_shade_colour_list_const, num_contouring_bins);
	mag_setr("contour_shade_min_level", min_contour_shade_level);
	mag_setr("contour_shade_max_level", max_contour_shade_level);

	//drawing data
	mag_cont();

};

void create_matrix_and_generate_svgs(int& max_samples, double**& gradientized_voltages, double& min_x, double&max_x, double& min_y, double& max_y, double& min_z, double& max_z, int& y_amount_dy, int& x_amount_dx, double*& contour_level_list, int& num_contouring_bins, const char**& contour_shade_colour_list_const, std::vector<std::vector<double> >& raw_eeg_voltages, std::map<int, std::vector<double> >& brain_spacial_map, std::vector<double>& x_coeoff, std::vector<double>& y_coeoff, std::vector<double>& x_vector_magics_c, std::vector<double>& y_vector_magics_c, int& line_counter){

	/*
	TO DO:
	COMPUTE NEAREST CONTROL POINTS NEAR EACH SAMPLE X,Y combo using soreted_voltage_indexes()
	http://geog.uoregon.edu/bartlein/old_courses/geog414s06/lectures/lec16.htm
	https://pimiddy.wordpress.com/2011/01/20/n-dimensional-interpolation/
	*/

	double range_z = (max_z - min_z);
	std::vector<std::vector<std::pair<int,double> > > raw_values;
	raw_values.reserve(y_amount_dy*x_amount_dx);
	std::vector<double> raw_z_values(y_amount_dy*x_amount_dx,0);

	double mean = 0.0;

	//to make it away from edge, should be around 1
	double numerical_edge_factor = 1.1;

	int number_of_flatten_samples = (int)(max_leads*raw_eeg_voltages.size());
	std::vector<double> raw_eeg_voltages_flatten(number_of_flatten_samples,0);
	std::vector<std::vector<double> > beta_coeffs;
	beta_coeffs.resize(max_samples);


	/*centroids for computing distance regression planes to solve overdetermined linear system for use as kernal but need to compute for every sample point instead of just once for all the data */

	//http://mathforum.org/library/drmath/view/69103.html
	//http://mathforum.org/library/drmath/view/63765.html
	double x0 = (double)std::accumulate(x_vec.begin(), x_vec.end(), 0.0)/(double)max_leads;
	double y0 = (double)std::accumulate(y_vec.begin(), y_vec.end(), 0.0)/(double)max_leads;
	std::vector<double> z0_centroids(max_leads,0);


	for(int i = 0; i<max_samples;i++){
		double temp_mean = (double)std::accumulate(raw_eeg_voltages[i].begin(), raw_eeg_voltages[i].end(), 0.0);
		z0_centroids.push_back(temp_mean/(double)max_leads);
		mean += temp_mean;

		
		for(int j=0;j<raw_eeg_voltages[i].size();j++){
			raw_eeg_voltages_flatten.push_back( ((raw_eeg_voltages[i][j] - min_z)/range_z) );
		}
		
	}

	
	mean = ((((double)(mean/(double)number_of_flatten_samples)) - min_z)/range_z);
	double stdv = calculate_stdv(raw_eeg_voltages_flatten, mean);

	double h_bar = std::pow(((4 * std::pow(stdv,5))/(3 * number_of_flatten_samples)), 1/5);

	std::cout << "mean: " << mean << std::endl;
	std::cout << "stdv: " << stdv << std::endl;
	int power_interop = 1;

	
	for(int i=0;i<max_samples;i++){
		Eigen::MatrixXf m(128,3);
		for(int j =0; j < max_leads;j++){
			double x_diff = x_vec[j] - x0;
			double y_diff = y_vec[j] - y0;
			double z_diff = ((raw_eeg_voltages[i][j] - z0_centroids[i])- min_z)/range_z;
			m(j,0) = x_diff;
			m(j,1) = y_diff;
			m(j,2) = z_diff;
		}
		//std::cout << "M:\n" << m << std::endl;
		Eigen::JacobiSVD<Eigen::MatrixXf> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::MatrixXf m_s_v = svd.matrixV();
		Eigen::MatrixXf m_s_u = svd.matrixU();
		//Eigen::VectorXf sing_values = svd.singularValues();
		std::vector<double> t_beta_coeffs = {m_s_v(0,0),m_s_v(1,0),m_s_v(2,0)};
		//std::vector<double> t_beta_coeffs = {sing_values(0),sing_values(1),sing_values(2)};
		//std::cout << "t_beta_coeffs: b1: " << t_beta_coeffs[0] << " b2: " << t_beta_coeffs[1] << " b3: " << t_beta_coeffs[2] << std::endl;
		//std::cout << "V:\n" << m_s_v << std::endl;
		//std::cout << "U:\n" << m_s_u << std::endl;
		//std::cout << "Its singular values are:\n" << svd.singularValues() << std::endl;
		

		beta_coeffs[i] = t_beta_coeffs;
	}


	int matrix_dim = 0;
	double plot_z_min = 0;
	double plot_z_max = 1;




	//number of points to weight
	//int nearest_points = 2;

	for(int i=0; i<x_amount_dx; i++){
		for(int j=0; j<y_amount_dy; j++){
			//if(std::abs(x_coeoff[i]) > 0 && std::abs(y_coeoff[j]) > 0){
				std::vector<std::pair<int,double> > nearest_voltage_indexes;
				for(int k =0; k< max_samples; k++){
					double temp_z_value;
					if(raw_values[matrix_dim].size() == 0){
						//nearest_voltage_indexes = soreted_voltage_indexes(brain_spacial_map,  x_vector_magics_c[i],  y_vector_magics_c[j]);
						//std::cout << "before z" << "\n";
						temp_z_value = find_z_value_on_trianglar_mesh(x_vector_magics_c[i],  y_vector_magics_c[j], raw_eeg_voltages[k]);
						//std::cout << "after z" << "\n";
						//raw_values[matrix_dim] = nearest_voltage_indexes;
						raw_z_values[matrix_dim] = temp_z_value;
						
					}else{
						//nearest_voltage_indexes = raw_values[matrix_dim];
						temp_z_value = raw_z_values[matrix_dim];
					}

					//double t_k_value = beta_coeffs[k][0] + (beta_coeffs[k][1] * x_vector_magics_c[i]) + (beta_coeffs[k][2] * y_vector_magics_c[j]);
					//double t_k_value = nearest_neighbour_weighted_interpolation(x_vector_magics_c[i], y_vector_magics_c[i], raw_eeg_voltages[k], min_z, range_z, power_interop);

					  
					//std::cout << "t_k_value: " << t_k_value << std::endl;
					/*
					double z_value = weighted_z_mean(nearest_points, numerical_edge_factor, min_z, range_z, raw_eeg_voltages[k], nearest_voltage_indexes);
					*/
					
					//double z_value = ((raw_eeg_voltages[k][nearest_voltage_indexes[0].first] - min_z)/range_z)/numerical_edge_factor;

					double z_value = ((temp_z_value - min_z)/range_z)/numerical_edge_factor;
					if(z_value > 1){
						//std::cout << z_value << "\n";
						gradientized_voltages[k][matrix_dim] = 1/numerical_edge_factor;
					}else if(z_value < 0){
						//std::cout << z_value << "\n";
						gradientized_voltages[k][matrix_dim] = 0;
					}else{
						gradientized_voltages[k][matrix_dim] = z_value;
					}
						

					//double k_value = (t_k_value)/numerical_edge_factor;
					//double k_value = ((t_k_value - min_z)/range_z)/numerical_edge_factor;

					/*
					double z_value = 0.0;
					for(int l =0;l<max_leads;l++){
						double tmp_z_value = (k_value - (((z_vec[l]-min_z)/range_z)/numerical_edge_factor))/h_bar;
						double tmp_value = inner_kernal(mean, stdv, tmp_z_value);
						z_value += tmp_value;
					}

					z_value = z_value/(max_leads*h_bar);
					*/
					//double data = k_value;
					//std::cout << "data: " << data << std::endl;
					
				}
			/*}else{

				for(int k =0; k< max_samples; k++){
					gradientized_voltages[k][matrix_dim] = .5/numerical_edge_factor;
				}
			}*/
			if(matrix_dim % 100000 == 0)
				std::cout << "matrix_dim: " << matrix_dim << std::endl;
			++matrix_dim;
		}
	}

	//open magics
	mag_open();

	for(int k =0; k< max_samples; k++){
		double countour_widths = 1/((double)num_contouring_bins);
		for(int i=0;i<num_contouring_bins;i++){

			contour_level_list[i] = (i * countour_widths);
		}
		std::string svg_file_name = "test_eeg_data_skip_count_" + std::to_string(skip_count) + "_" + std::to_string(k) + "_" + std::to_string(line_counter);
		generate_magics_plot(svg_file_name, min_x, max_x, min_y, max_y, gradientized_voltages[k], y_amount_dy,  x_amount_dx, contour_level_list, num_contouring_bins, contour_shade_colour_list_const, plot_z_min, plot_z_max);
	}
	//closings magics api
	mag_close();
}

void mine_file(std::string& file_name, int& max_samples, double**& gradientized_voltages, double& min_x, double&max_x, double& min_y, double& max_y, double& min_z, double& max_z, int& y_amount_dy, int& x_amount_dx, double*& contour_level_list, int& num_contouring_bins, const char**& contour_shade_colour_list_const, std::map<int, std::vector<double> >& brain_spacial_map, std::vector<double>& x_coeoff, std::vector<double>& y_coeoff, std::vector<double>& x_vector_magics_c, std::vector<double>& y_vector_magics_c){
	std::vector<double> data_vector_out;
	std::vector<std::vector<double> > raw_eeg_voltages_cache;
	std::string line;
	int line_counter = 0;
	
	int skip_counter = 0;

	std::ifstream in(file_name.c_str());
	if (!in.is_open()) exit(0);

	while (std::getline(in,line)){
		if(line_counter > max_lines)
			break;

		if(line_counter % skip_count == 0){
			data_vector_out.clear();
	    	std::vector<std::string> data_vector_string = string_split(line);
	    	for(int i =offset_node; i<offset_node+max_leads;i++){
	    		if (data_vector_string[i].size() > 0){
	                double node_val;
	                std::stringstream(data_vector_string[i]) >> node_val;
	                data_vector_out.push_back(node_val/biosemi_microvoltage_factor);
	            }      
	    	}
	    	data_vector_string.clear();
	    	if(max_samples == 1){
	    		raw_eeg_voltages_cache.push_back(data_vector_out);
	    		std::cout << "Generating svgs: " << skip_counter << std::endl;
	    		create_matrix_and_generate_svgs(max_samples, gradientized_voltages, min_x, max_x, min_y, max_y, min_z, max_z, y_amount_dy, x_amount_dx, contour_level_list, num_contouring_bins, contour_shade_colour_list_const, raw_eeg_voltages_cache, brain_spacial_map, x_coeoff, y_coeoff, x_vector_magics_c, y_vector_magics_c, skip_counter);
	    		raw_eeg_voltages_cache.clear();
	    	}else if(((skip_counter % max_samples) == 0) && skip_counter != 0){
	    		std::cout << "Generating svgs: " << skip_counter << std::endl;
	    		create_matrix_and_generate_svgs(max_samples, gradientized_voltages, min_x, max_x, min_y, max_y, min_z, max_z, y_amount_dy, x_amount_dx, contour_level_list, num_contouring_bins, contour_shade_colour_list_const, raw_eeg_voltages_cache, brain_spacial_map, x_coeoff, y_coeoff, x_vector_magics_c, y_vector_magics_c, skip_counter);
	    		raw_eeg_voltages_cache.clear();
	    		raw_eeg_voltages_cache.push_back(data_vector_out);
	    	}else{
	    		raw_eeg_voltages_cache.push_back(data_vector_out);
	    	}
	    	++skip_counter;
		}
    	
    	++line_counter;
    }

    if(max_samples ==1 ){
    	if(raw_eeg_voltages_cache.size() > 0){
	    	int temp_max_samples = (int)raw_eeg_voltages_cache.size();
	    	std::cout << "Generating svgs: end of file" << std::endl;
	    	create_matrix_and_generate_svgs(temp_max_samples, gradientized_voltages, min_x, max_x, min_y, max_y, min_z, max_z, y_amount_dy, x_amount_dx, contour_level_list, num_contouring_bins, contour_shade_colour_list_const, raw_eeg_voltages_cache, brain_spacial_map, x_coeoff, y_coeoff, x_vector_magics_c, y_vector_magics_c, skip_counter);
	    	raw_eeg_voltages_cache.clear();
	    }
    }
    
}

std::vector<double> min_max_z(std::string& file_name){
	std::cout << "Finding Min and Max" << std::endl;
	std::vector<double> out_vector(2,0);
	std::string line;
	std::ifstream in(file_name.c_str());
	double min_z = (double)std::numeric_limits<int>::max();
	double max_z = (double)std::numeric_limits<int>::min();
	int line_counter = 0;
	
	if (!in.is_open()) exit(0);

	while (std::getline(in,line)){
		if(line_counter > max_lines)
			break;

		if(line_counter % skip_count == 0){
	    	std::vector<std::string> data_vector_string = string_split(line);
	    	for(int i =offset_node; i<offset_node+max_leads;i++){
	    		if (data_vector_string[i].size() > 0){
	                double node_val;
	                std::stringstream(data_vector_string[i]) >> node_val;
	                double raw_value = node_val/biosemi_microvoltage_factor;

	                if(min_z > raw_value)
	                	min_z = raw_value;
	                if(max_z < raw_value)
	                	max_z = raw_value;
	            }      
	    	}
	    }

	    ++line_counter;
    }

    out_vector[0] = min_z;
    out_vector[1] = max_z;

    return out_vector;
}


int main(int argc, char *arvg[]){
	
	double min_x = 0;
	double min_y = 0;

	double max_x = 0;
	double max_y = 0;
	
	std::vector<std::vector<double> > grid_points;



	std::map<int, std::vector<double> > brain_spacial_map = generate_spacial_map();

	for(int i=0; i< max_leads; i++){
		double temp_x = brain_spacial_map[i][0];
		double temp_y = brain_spacial_map[i][1];
		std::vector<double> temp_grid_point = {temp_x, temp_y};
		z_point_map[std::to_string(temp_x) + "_" + std::to_string(temp_y)] = i;
		grid_points.push_back(temp_grid_point);

		x_vec.push_back(temp_x);
		y_vec.push_back(temp_y);
		z_vec.push_back(brain_spacial_map[i][2]);

		if(min_x > temp_x) min_x = temp_x;
		if(min_y > temp_y) min_y = temp_y;
		if(max_x < temp_x) max_x = temp_x;
		if(max_y < temp_y) max_y = temp_y;
	}

	//computing delaunay trianglation with  with O(n*log(n))
	//std::vector<std::vector<std::vector<double>>> plane_triangles = compute_triangles_from_delaunay_triangulation_divide_and_conquer(grid_points);
	

	//std::cout << "point_combinations: " << point_combinations.size() << std::endl;

	//computing delaunay trianglation with  with O(n^4)
	plane_triangles = compute_triangles_from_delaunay_triangulation_brute_force(grid_points);
	std::cout << "plane_triangles: " << plane_triangles.size() << std::endl;
	/*
	for(int i=0; i<plane_triangles.size(); i++){
		std::cout << "plane_triangles " << i << std::endl;
		for(int j=0;j<3;j++){
			print_vector(plane_triangles[i][j]);
		}		
	}*/

	exit(0);

	double x_width = max_x;
	double y_width = max_y;

	std::sort(x_vec.begin(), x_vec.end());
	std::sort(y_vec.begin(), y_vec.end());


	for(int i=0;i<x_vec.size()-1;i++){
		double diff_x = std::abs(x_vec[i+1] - x_vec[i]);
		if((x_width > diff_x) && (diff_x > 0)){
			x_width = diff_x;
		}
		double diff_y = std::abs(y_vec[i+1] - y_vec[i]);
		if((y_width > diff_y)  && (diff_y > 0)){
			y_width = diff_y;
		}
	}
	y_width = y_width*width_factor;
	x_width = x_width*width_factor;

	int x_amount_dx = (int)std::ceil((max_x-min_x)/x_width);
	int y_amount_dy = (int)std::ceil((max_y-min_y)/y_width);
	max_x = min_x + (x_width * x_amount_dx);
	max_y = min_y + (y_width * y_amount_dy);

	std::vector<double> x_vector_magics_c(x_amount_dx,0);
	std::vector<double> y_vector_magics_c(y_amount_dy,0);

	std::cout << "x_amount_dx: " << x_amount_dx << "\n";
	std::cout << "y_amount_dy: " << y_amount_dy << "\n";
	
	for(int i=0;i<x_amount_dx;i++){
		x_vector_magics_c[i] = min_x + (i * x_width);
	}

	for(int i=0;i<y_amount_dy;i++){
		y_vector_magics_c[i] = min_y + (i * y_width);
	}

	//comput NURBS coeffs for x and y vectors
	//int power = 3;
	std::vector<double> x_coeoff;// = calculate_coeffecients_b_spline(power, x_amount_dx, x_vec, x_vector_magics_c);
	std::vector<double> y_coeoff;// = calculate_coeffecients_b_spline(power, y_amount_dy, y_vec, y_vector_magics_c);
	
	
	/*
		Generating color gradients
		non repeating color set
	*/

	
	double freq = round(((PI * 2)/ (contouring_bins * 1.5))*1000)/1000.00; 
	double* contour_level_list = (double*)malloc(contouring_bins*sizeof(double));
	std::vector<std::string> color_gradients = make_color_gradeint(freq, freq, freq, 4, 0, 2, contouring_bins);
	char** contour_shade_colour_list = (char**)malloc(contouring_bins *sizeof(char*));
	for(int i=0;i<contouring_bins;i++){
		const char* c = color_gradients[i].c_str();
		contour_shade_colour_list[i] = (char*) malloc((strlen(color_gradients[i].c_str())/sizeof(char))+1);
		strlcpy(contour_shade_colour_list[i], c, (strlen(color_gradients[i].c_str())/sizeof(char)));
	}
	const char** contour_shade_colour_list_const = (const char**)contour_shade_colour_list;

	//24 MB times the number of samples -- memory constrained
	double** gradientized_voltages = (double**) malloc(max_samples *sizeof(double*));
	for(int i=0; i< max_samples; i++){
		gradientized_voltages[i] = (double*) malloc(x_amount_dx * y_amount_dy *sizeof(double));
	} 

	std::string file_name = "/Users/cinquemb/Desktop/Jud_run3_run1/Jud_run3_042114_1251_Run1_raw.txt";

	std::vector<double> min_max_vector = min_max_z(file_name);
	double min_z = min_max_vector[0];
	double max_z = min_max_vector[1];

	mine_file(file_name, max_samples, gradientized_voltages, min_x, max_x, min_y, max_y, min_z, max_z, y_amount_dy, x_amount_dx, contour_level_list, contouring_bins, contour_shade_colour_list_const, brain_spacial_map, x_coeoff, y_coeoff, x_vector_magics_c, y_vector_magics_c);
	
	if(gradientized_voltages) free(gradientized_voltages);
	if(contour_level_list) free(contour_level_list);
	if(contour_shade_colour_list) free(contour_shade_colour_list);
	

    return 0;
}  