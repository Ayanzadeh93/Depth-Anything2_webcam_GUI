import sys
import cv2
import threading
import time
import numpy as np
from PIL import Image
from transformers import pipeline
import torch
from queue import Queue
import logging
from datetime import datetime
import os
import json
from PyQt6.QtCore import QThread, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, 
    QMainWindow, 
    QWidget, 
    QVBoxLayout,
    QHBoxLayout, 
    QPushButton, 
    QLabel, 
    QComboBox,
    QSlider, 
    QStatusBar, 
    QMessageBox,
    QGroupBox,
    QGridLayout,
    QProgressBar,
    QFileDialog
)
from PyQt6.QtGui import QImage, QPixmap, QPalette, QColor, QIcon

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
# Set the environment variable to enable CPU fallback for unsupported MPS operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import os

# Enable GPU acceleration with Metal Performance Shaders (MPS)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

class DepthEstimationThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, float)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_running = False
        self.frame_queue = Queue(maxsize=2)
        
        # Check and set up GPU device
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
            logger.info("Using Apple Metal GPU (MPS)")
        else:
            self.device = torch.device("cpu")
            logger.info("GPU not available, falling back to CPU")
            
        self.recording = False
        self.video_writer = None
        self.depth_threshold = 128
        self.selected_colormap = cv2.COLORMAP_TURBO
        self.last_frame_time = time.time()
        self.cap = None
        self.model_loaded = False
        
        # Initialize the model
        self.load_model()
    def setup_camera(self):
            """Initialize and configure the camera"""
            try:
                logger.info("Setting up camera...")
                if self.cap is not None and self.cap.isOpened():
                    self.cap.release()
                
                self.cap = cv2.VideoCapture(0)
                
                # Check if camera opened successfully
                if not self.cap.isOpened():
                    raise Exception("Could not open camera")
                    
                # Configure camera settings for optimal performance
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # Verify camera settings
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                logger.info(f"Camera initialized - FPS: {actual_fps}, Resolution: {actual_width}x{actual_height}")
                
                # Test camera by reading one frame
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    raise Exception("Could not read frame from camera")
                    
                return True
                
            except Exception as e:
                error_msg = f"Camera setup failed: {str(e)}"
                logger.error(error_msg)
                self.error_occurred.emit(error_msg)
                return False
    def load_model(self):
        try:
            logger.info(f"Loading depth estimation model on {self.device}...")
            
            # Initialize model with GPU support
            self.pipe = pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=self.device
            )
            
            self.model_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            self.model_loaded = False
            error_msg = f"Failed to initialize model: {str(e)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

    def process_frame(self, frame):
        """Process a single frame with GPU acceleration"""
        try:
            frame_resized = cv2.resize(frame, (320, 240))
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            # Run inference on GPU
            with torch.inference_mode():
                depth_map = self.pipe(pil_img)["depth"]
                
            # Move depth map to CPU for OpenCV processing
            depth_map = depth_map.cpu().numpy() if hasattr(depth_map, 'cpu') else depth_map
            
            depth_normalized = cv2.normalize(
                np.array(depth_map), 
                None, 
                0, 
                255, 
                cv2.NORM_MINMAX
            ).astype(np.uint8)
            
            depth_normalized[depth_normalized < self.depth_threshold] = 0
            depth_colored = cv2.applyColorMap(depth_normalized, self.selected_colormap)
            
            return frame_resized, depth_colored
            
        except Exception as e:
            raise Exception(f"Frame processing error: {str(e)}")

    def run(self):
        if not self.model_loaded:
            self.error_occurred.emit("Model not loaded. Please restart the application.")
            return
            
        if not self.setup_camera():
            self.error_occurred.emit("Failed to setup camera. Please check camera connections.")
            return

        self.is_running = True
        frame_count = 0
        
        try:
            while self.is_running:
                try:
                    current_time = time.time()
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        continue

                    # Process frame with GPU acceleration
                    frame_resized, depth_colored = self.process_frame(frame)
                    
                    # Calculate FPS
                    fps = 1.0 / (time.time() - current_time)
                    
                    if self.recording and self.video_writer:
                        combined_frame = np.hstack((frame_resized, depth_colored))
                        self.video_writer.write(combined_frame)
                    
                    self.frame_ready.emit(frame_resized, depth_colored, fps)
                    frame_count += 1
                    
                except torch.cuda.OutOfMemoryError:
                    logger.error("GPU out of memory, clearing cache...")
                    torch.mps.empty_cache()
                    continue
                    
                except Exception as e:
                    error_msg = f"Frame processing error: {str(e)}"
                    logger.error(error_msg)
                    self.error_occurred.emit(error_msg)
                    continue

        finally:
            self.cleanup()
            logger.info(f"Processed {frame_count} frames")

    def cleanup(self):
        """Clean up GPU resources"""
        logger.info("Cleaning up resources...")
        if hasattr(self, 'pipe'):
            del self.pipe
        torch.mps.empty_cache()
        if self.video_writer:
            self.video_writer.release()
        if self.cap:
            self.cap.release()
        self.is_running = False
        logger.info("Cleanup completed")

    def stop(self):
        """Stop the thread safely"""
        logger.info("Stopping depth estimation thread...")
        self.is_running = False
        self.wait()
        self.cleanup()
class DepthVisualizerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.recording = False
        self.init_ui()
        self.depth_thread = DepthEstimationThread()
        self.depth_thread.frame_ready.connect(self.update_frames)
        self.depth_thread.error_occurred.connect(self.show_error)

    def init_ui(self):
        self.setWindowTitle('Depth Visualization')
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
                border: 2px solid #3498db;
                border-radius: 5px;
                padding: 2px;
                background-color: #1e1e1e;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QComboBox {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
            QStatusBar {
                color: white;
            }
            QGroupBox {
                color: white;
                border: 1px solid #3498db;
                border-radius: 5px;
                margin-top: 1ex;
                padding: 10px;
            }
            QGroupBox::title {
                color: white;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
        """)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create display area
        display_layout = QHBoxLayout()
        self.camera_label = QLabel()
        self.depth_label = QLabel()
        self.camera_label.setMinimumSize(320, 240)
        self.depth_label.setMinimumSize(320, 240)
        display_layout.addWidget(self.camera_label)
        display_layout.addWidget(self.depth_label)
        layout.addLayout(display_layout)

        # Create control panel with groups
        control_layout = QGridLayout()
        
        # Camera Control Group
        camera_group = QGroupBox("Camera Control")
        camera_layout = QHBoxLayout()
        
        self.start_stop_btn = QPushButton('Start')
        self.start_stop_btn.clicked.connect(self.toggle_processing)
        
        self.record_btn = QPushButton('Record')
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        
        camera_layout.addWidget(self.start_stop_btn)
        camera_layout.addWidget(self.record_btn)
        camera_group.setLayout(camera_layout)
        
        # Visualization Control Group
        viz_group = QGroupBox("Visualization Control")
        viz_layout = QGridLayout()
        
        viz_layout.addWidget(QLabel("Colormap:"), 0, 0)
        self.colormap_combo = QComboBox()
        colormaps = ['TURBO', 'JET', 'VIRIDIS', 'PLASMA', 'MAGMA']
        self.colormap_combo.addItems(colormaps)
        self.colormap_combo.currentTextChanged.connect(self.change_colormap)
        viz_layout.addWidget(self.colormap_combo, 0, 1)
        
        viz_layout.addWidget(QLabel("Depth Threshold:"), 1, 0)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(128)
        self.threshold_slider.valueChanged.connect(self.change_threshold)
        viz_layout.addWidget(self.threshold_slider, 1, 1)
        
        self.threshold_value = QLabel("128")
        viz_layout.addWidget(self.threshold_value, 1, 2)
        
        viz_group.setLayout(viz_layout)
        
        # Stats Group
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout()
        
        self.fps_label = QLabel("FPS: 0.0")
        stats_layout.addWidget(self.fps_label, 0, 0)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        stats_layout.addWidget(self.progress_bar, 1, 0)
        
        stats_group.setLayout(stats_layout)
        
        # Add groups to control layout
        control_layout.addWidget(camera_group, 0, 0)
        control_layout.addWidget(viz_group, 0, 1)
        control_layout.addWidget(stats_group, 0, 2)
        
        # Add control layout to main layout
        layout.addLayout(control_layout)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Ready')

        # Set window size and position
        self.setMinimumSize(800, 400)
        self.center_window()

    def center_window(self):
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )

    def toggle_recording(self):
        if not self.recording:
            filename = f"depth_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Video", filename, "Video Files (*.mp4)"
            )
            if file_path:
                self.recording = True
                self.record_btn.setText("Stop Recording")
                self.record_btn.setStyleSheet("background-color: #e74c3c;")
                self.depth_thread.set_recording(True, file_path)
        else:
            self.recording = False
            self.record_btn.setText("Record")
            self.record_btn.setStyleSheet("")
            self.depth_thread.set_recording(False)

    def change_threshold(self, value):
        self.threshold_value.setText(str(value))
        self.depth_thread.set_depth_threshold(value)

    def change_colormap(self, colormap_name):
        self.depth_thread.set_colormap(colormap_name)

    def update_frames(self, frame, depth, fps):
        h, w = frame.shape[:2]
        q_img = QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(q_img))

        h, w = depth.shape[:2]
        q_img = QImage(depth.data, w, h, depth.strides[0], QImage.Format.Format_RGB888)
        self.depth_label.setPixmap(QPixmap.fromImage(q_img))
        
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.progress_bar.setValue(int(min(fps, 30) / 30 * 100))

    def show_error(self, message):
        QMessageBox.critical(self, 'Error', message)
        self.status_bar.showMessage('Error occurred')
        if self.start_stop_btn.text() == 'Stop':
            self.toggle_processing()

    def toggle_processing(self):
        if self.start_stop_btn.text() == 'Start':
            self.depth_thread.start()
            self.start_stop_btn.setText('Stop')
            self.record_btn.setEnabled(True)
            self.status_bar.showMessage('Processing...')
        else:
            self.depth_thread.stop()
            self.start_stop_btn.setText('Start')
            self.record_btn.setEnabled(False)
            self.status_bar.showMessage('Stopped')

    def closeEvent(self, event):
        self.depth_thread.stop()
        event.accept()

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('depth_estimation.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Create Qt application
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        # Set dark theme
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        app.setPalette(dark_palette)

        # Create and show main window
        window = DepthVisualizerGUI()
        window.show()

        # Set up exception handling
        sys._excepthook = sys.excepthook
        def exception_hook(exctype, value, traceback):
            logger.error("Uncaught exception", exc_info=(exctype, value, traceback))
            sys._excepthook(exctype, value, traceback)
        sys.excepthook = exception_hook

        # Start application
        sys.exit(app.exec())

    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
