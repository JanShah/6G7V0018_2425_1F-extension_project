#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import argparse
import os
import threading
import time
from geometry_msgs.msg import PoseStamped

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    rospy.logwarn("Ultralytics YOLO not available. Object detection will be disabled.")

class CameraDetection:
    def __init__(self, camera_name="head_camera", enable_detection=True):
        self.camera_name = camera_name
        self.bridge = CvBridge()
        self.image_sub = None
        self.latest_image = None
        self.image_received = False
        self.frame_count = 0
        self.running = True
        self.enable_detection = enable_detection and YOLO_AVAILABLE
        self.fps = 0
        self.last_time = time.time()
        self.model = None
        self.detection_results = None

        self.pose_pub = rospy.Publisher('/detected_object_pose', PoseStamped, queue_size=10)
        self.save_dir = os.path.expanduser(f"~/camera_detection/{camera_name}")
        os.makedirs(self.save_dir, exist_ok=True)

        if self.enable_detection:
            self.load_model()

        topic = f"/io/internal_camera/{camera_name}/image_raw"
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        rospy.loginfo(f"Subscribed to {topic}")

        self.display_thread = threading.Thread(target=self.display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()

        rospy.loginfo(f"Camera detection initialised for {camera_name}")
        rospy.loginfo(f"Images will be saved to {self.save_dir}")
        if self.enable_detection:
            rospy.loginfo("Object detection is enabled")
        else:
            rospy.loginfo("Object detection is disabled")

    def load_model(self):
        """Load the YOLO model"""
        try:
            rospy.loginfo("Loading standard YOLOv8n model...")

            try:
                import ultralytics
            except ImportError:
                rospy.logwarn("Ultralytics not found, attempting to install...")
                os.system("pip install ultralytics")

            try:
                self.model = YOLO('yolov8n.pt')
                return
            except Exception as e:
                rospy.logwarn(f"Failed to load standard YOLOv8n model directly: {e}")

            try:
                import torch
                rospy.loginfo(f"PyTorch version: {torch.__version__}")

                model_path = os.path.expanduser("~/yolov8n.pt")
                if not os.path.exists(model_path):
                    os.system(f"wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O {model_path}")

                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                    return
                else:
                    self.enable_detection = False
                    return
            except Exception as e:
                self.enable_detection = False
                return

        except Exception as e:
            self.enable_detection = False

    @staticmethod
    def crop_and_zoom(image, zoom_factor=2):
        h, w, _ = image.shape
        new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2
        cropped = image[y1:y1 + new_h, x1:x1 + new_w]
        zoomed_image = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return zoomed_image
    
    def log_detection_results(self,image, results):
        if len(results[0].boxes) == 0:
            return
        detection_count = sum(1 for conf in results[0].boxes.conf.cpu().numpy() if conf > 0.3)

        if detection_count > 0 and self.frame_count % 90 == 0:
            annotated_img = self.process_detections(image, results)
            filename = os.path.join(self.save_dir, f"detection_{self.frame_count}.jpg")
            cv2.imwrite(filename, annotated_img)
            rospy.loginfo(f"Saved detection frame {self.frame_count} to {filename}")
    
    def callback(self, data):
        try:
            cv_image = CameraDetection.crop_and_zoom(self.bridge.imgmsg_to_cv2(data, "bgr8"))

            self.latest_image = cv_image
            self.image_received = True
            self.frame_count += 1

            if self.enable_detection and self.model is not None:
                try:
                    self.detection_results = self.model(cv_image, conf=0.25, verbose=False)
                    self.log_detection_results(cv_image, self.detection_results)
                except Exception as e:
                    rospy.logerr(f"Error in object detection: {e}")

            if self.frame_count % 100 == 0:
                filename = os.path.join(self.save_dir, f"frame_{self.frame_count}.jpg")
                cv2.imwrite(filename, cv_image)
                rospy.loginfo(f"Saved frame {self.frame_count} to {filename}")

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {str(e)}")
        except Exception as e:
            rospy.logerr(f"Error in callback: {str(e)}")

    def display_loop(self):
        window_title = f"{self.camera_name} - {'Detection' if self.enable_detection else 'Camera'}"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 1280, 800)

        if self.camera_name == "head_camera":
            cv2.moveWindow(window_title, 50, 50)
        elif self.camera_name == "right_hand_camera":
            cv2.moveWindow(window_title, 50, 900)

        rate = rospy.Rate(30)

        while self.running and not rospy.is_shutdown():
            self.draw(window_title)
            rate.sleep()

        cv2.destroyAllWindows()

    def draw(self, window_title):
        if self.image_received and self.latest_image is not None:
            try:
                current_time = time.time()
                elapsed = current_time - self.last_time
                self.last_time = current_time
                self.fps = 1.0 / elapsed if elapsed > 0 else 0

                display_image = self.latest_image.copy()

                if self.enable_detection and self.detection_results is not None:
                    display_image = self.process_detections(display_image, self.detection_results)

                mode_text = "Detection" if self.enable_detection else "Camera"
                cv2.putText(
                    display_image,
                    f"{self.camera_name} - {mode_text} - FPS: {self.fps:.1f}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
                )

                cv2.imshow(window_title, display_image)

                key = cv2.waitKey(1)
                if key == 27:  
                    self.running = False
                    rospy.signal_shutdown("User pressed ESC")
                elif key == ord('s'): 
                    filename = os.path.join(self.save_dir, f"manual_save_{int(time.time())}.jpg")
                    cv2.imwrite(filename, display_image)
                    rospy.loginfo(f"Manually saved frame to {filename}")

            except Exception as e:
                rospy.logerr(f"Error displaying image: {str(e)}")

    def process_detections(self, image, results):
        try:
            result = results[0]

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            class_names = result.names
            annotated_img = image.copy()
            detection_count = sum(1 for conf in confs if conf > 0.3)
            cv2.putText( annotated_img, f"Detections: {detection_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            for box, conf, class_id in zip(boxes, confs, class_ids):
                if conf > 0.10: 
                    x1, y1, x2, y2 = box.astype(int)

                    class_name = class_names[class_id]

                    if conf > 0.7:
                        colour = (0, 255, 0)
                    elif conf > 0.5:
                        colour = (0, 255, 255) 
                    else:
                        colour = (0, 0, 255) 

                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), colour, 2)

                    text_size = cv2.getTextSize(f"{class_name}: {conf:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated_img, (x1, y1 - 25), (x1 + text_size[0] + 10, y1), colour, -1 )

                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(annotated_img, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    if self.frame_count % 30 == 0:
                        rospy.loginfo(f"Detected {class_name} with confidence {conf:.2f}")
                    if class_name in ['can', 'cup', 'bottle'] and conf > 0.2:
                        if self.frame_count % 30 == 0:
                            rospy.loginfo(f"Detected {class_name} with confidence {conf:.2f}")

                            
                            pose = self.pixel_to_pose(x1, y1, x2, y2)

                            if pose:
                                self.pose_pub.publish(pose)
                                
            if detection_count == 0:
                cv2.putText(
                    annotated_img,  "No objects detected",
                    (int(annotated_img.shape[1]/2) - 150, int(annotated_img.shape[0]/2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
                )

            return annotated_img
        except Exception as e:
            rospy.logerr(f"Error processing detections: {str(e)}")
            return image
    
    def pixel_to_pose(self, x1, y1, x2, y2):
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = 'base' 

        if center_x < 200:
            pose.pose.position.y = 0.2
        elif center_x < 400:
            pose.pose.position.y = 0.0
        else:
            pose.pose.position.y = -0.2

        pose.pose.position.x = 1.0
        pose.pose.position.z = 0.1
        
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 1.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 0.0

        return pose
    def shutdown(self):
        """Shutdown the node"""
        self.running = False
        if self.display_thread.is_alive():
            self.display_thread.join(1.0)
        cv2.destroyAllWindows()

def main():
    rospy.init_node('camera_detection', anonymous=True)

    camera_name = rospy.get_param('~camera', None)
    enable_detection = rospy.get_param('~detection', True)

    if camera_name is None:
        parser = argparse.ArgumentParser(description='Camera Detection')
        parser.add_argument('-c', '--camera', type=str, default="head_camera",
                            choices=["head_camera", "right_hand_camera"],
                            help='Camera to use')
        parser.add_argument('-d', '--detection', action='store_true',
                            help='Enable object detection')

        argv = rospy.myargv()
        args = parser.parse_args(argv[1:])
        camera_name = args.camera
        enable_detection = args.detection

    detector = CameraDetection(camera_name, enable_detection)

    rospy.loginfo(f"Camera detection for {camera_name} running. Press ESC to exit, 's' to save a frame.")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")

    detector.shutdown()

if __name__ == '__main__':
    main()
