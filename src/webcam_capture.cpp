//opencv
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//Libraries 
#include "cv.h"
#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
//#include "highgui.h"
//std
#include <iostream>
#include <cstdlib>



//Variables
using namespace cv;
using namespace std;

void detectAndDisplay(Mat frame);
//Clasificador XML:
	String face_cascade_name = "/home/felix/webcam_capture_px/data/haarcascades/haarcascade_frontalface_default.xml";
	String eyes_cascade_name = "/home/felix/webcam_capture_px/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

	CascadeClassifier eyes_cascade;
	CascadeClassifier face_cascade;

//Cargar imágenes sombrero y bigote 
    Mat hat = imread("/home/felix/webcam_capture_px/img/hat.png", -1);  
    Mat hat_resized; //Imagen de sombrero redimensionada
    Mat moustache = imread("/home/felix/webcam_capture_px/img/moustache.png", -1);
    Mat moustache_resized; //Imagen de bigote redimensionada

int main(int argc, char *argv[]) 
{
    //Craga cascade
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    if(! hat.data ) // Mirar si la imagen craga
    {
        cout <<  "Could not open or find the image <- hat.png ->" << endl ;
        return -1;
    }

    if(! moustache.data ) // Mirar si la imagen craga
    {
        cout <<  "Could not open or find the image <- moustache.png ->" << endl ;
        return -1;
}
	
	int cam_id; 
	Mat src, dst;
    cv::VideoCapture camera; //OpenCV video capture
    cv::Mat image; //OpenCV image object
    cv::Mat gray_image;
    cv::Point center;
	
	//Definicion de 3 variables para obtener la combinación de píxeles para el sombrero y bigote
    double color_pixel_0, color_pixel_1, color_pixel_2;


	//check user args
	switch(argc)
	{
		case 1: //no argument provided, so try /dev/video0
			cam_id = 0;  
			break; 
		case 2: //an argument is provided. Get it and set cam_id
			cam_id = atoi(argv[1]);
			break; 
		default: 
			std::cout << "Invalid number of arguments. Call program as: webcam_capture [video_device_id]. " << std::endl; 
			std::cout << "EXIT program." << std::endl; 
			break; 
	}
	
	//advertising to the user 
	std::cout << "Opening video device " << cam_id << std::endl;

    //open the video stream and make sure it's opened
    if( !camera.open(cam_id) ) 
	{
        std::cout << "Error opening the camera. May be invalid device id. EXIT program." << std::endl;
        return -1;
    }

    //capture loop. Out of user press a key
    while(1)
	{
		//Read image and check it. Blocking call up to a new image arrives from camera.
        if(!camera.read(image)) 
		{
            std::cout << "No frame" << std::endl;
            cv::waitKey();
        }

         // escala de grises
        cv::cvtColor(image, gray_image, CV_BGR2GRAY);
        equalizeHist(gray_image,gray_image); //Este algoritmo normaliza el brillo y aumenta el contraste de la imagen


	     // Detecion de caras
	    std::vector<Rect> faces;
	    face_cascade.detectMultiScale( image, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

        // Detectcion de ojos
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale( image, eyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	    
	    // Dibujar circulos en las caras ojos y superponer sombrero y bigote
	    for( int i = 0; i < faces.size(); i++ )
	    {
	    	 Rect face_i = faces[i];

			Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        	cv::circle( image, center, faces[i].width/2, cv::Scalar(0,0,255), 3, 8, 0 );// Circulo rojo en la cara


        	int facew = face_i.width; //face ancho
		    int faceh = face_i.height; //face alto
	        Size hat_size(facew,faceh); //Redimensionar imagen de sombrero dando la misma anchura de la cara
	        resize(hat, hat_resized, hat_size );

	        Size moustache_size(facew/2,faceh/2); // Redimensionar el cuadro de bigote dando la mitad del ancho de la cara
	        resize(moustache, moustache_resized, moustache_size );

	        double hat_locate = 0.50; //Variable para subir el sombrero desde la posición de la cara
	        double moustache_locate_y = 0.50; //Variable para mover el bigote desde la posición de la cara
	        double moustache_move_x = (facew - moustache_resized.size[0])/2; //Variable para mover el bigote desde la posición de la cara

            // Dibuja circulos verdes en los ojos
            for( int i = 0; i < eyes.size(); i++ )
            {
                Point center( eyes[i].x + eyes[i].width*0.5, eyes[i].y + eyes[i].height*0.5 );
                cv::circle( image, center, eyes[i].width/2, cv::Scalar(0,255,0), 3, 8, 0 );// circle verde
            }

	        //Superposición del sombrero y bigote
        for ( int j = 0; j < faceh ; j++)
	    {
                for ( int k = 0; k < facew; k++)
                {
                    // Determina la posicon usando alpha para superrponer el sombrero
                    double alpha_hat = hat_resized.at<cv::Vec4b>(j, k)[3] / 255.0;
                    color_pixel_0 = (hat_resized.at<cv::Vec4b>(j, k)[0] * (alpha_hat)) + ((image.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[0])* (1.0-alpha_hat));
                    color_pixel_1 = (hat_resized.at<cv::Vec4b>(j, k)[1] * (alpha_hat)) + ((image.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[1])* (1.0-alpha_hat));
                    color_pixel_2 = (hat_resized.at<cv::Vec4b>(j, k)[2] * (alpha_hat)) + ((image.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[2])* (1.0-alpha_hat));

                    if((face_i.y +j-(faceh*hat_locate))>0){
                        image.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[0] = color_pixel_0 ;
                        image.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[1] = color_pixel_1 ;
                        image.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[2] = color_pixel_2 ;
                    }

                    if((j<(faceh/2))&&(k<(facew/2))){
                        //  Determina la posicon usando alpha para superrponer el bigote
                        double alpha_moustache = moustache_resized.at<cv::Vec4b>(j, k)[3] / 255.0;
                        color_pixel_0 = (moustache_resized.at<cv::Vec4b>(j, k)[0] * (alpha_moustache)) + ((image.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[0])* (1.0-alpha_moustache));
                        color_pixel_1 = (moustache_resized.at<cv::Vec4b>(j, k)[1] * (alpha_moustache)) + ((image.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[1])* (1.0-alpha_moustache));
                        color_pixel_2 = (moustache_resized.at<cv::Vec4b>(j, k)[2] * (alpha_moustache)) + ((image.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[2])* (1.0-alpha_moustache));

                        image.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[0] = color_pixel_0 ;
                        image.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[1] = color_pixel_1 ;
                        image.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[2] = color_pixel_2 ;
                    }
                }
            }

	    }

        //show image in a window
        cv::imshow("Output Window", image);
		
		//Waits 1 millisecond to check if a key has been pressed. If so, breaks the loop. Otherwise continues.
        if(cv::waitKey(1) >= 0) break;
    }   
}