import sys
import cv2
import numpy as np
import socket
import ipaddress
import requests
from datetime import datetime
import json
import os
import re
import time
import struct
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                               QPushButton, QWidget, QLabel, QComboBox, QGroupBox, QFileDialog,
                               QStyle, QListWidget, QLineEdit, QHBoxLayout, QListWidgetItem, 
                               QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy, QTextEdit, QMessageBox)
from PySide6.QtCore import QDir, QTimer, Qt, Signal, QThread, QRectF, QDateTime, QStandardPaths, Slot
from PySide6.QtGui import QImage, QPixmap, QFont, QPainter, QPainterPath, QColor
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pygame
from queue import Queue, Empty
import torch
from ultralytics import YOLO
import atexit


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def rounded_pixmap(pixmap, radius=20):
    rounded = QPixmap(pixmap.size())
    rounded.fill(Qt.transparent)

    painter = QPainter(rounded)
    painter.setRenderHint(QPainter.Antialiasing)

    path = QPainterPath()
    rect = QRectF(pixmap.rect())
    path.addRoundedRect(rect, radius, radius)

    painter.setClipPath(path)
    painter.drawPixmap(0, 0, pixmap)
    painter.end()

    return rounded

class LoadingScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 200);")  # Dark with some transparency

        # Set window title and size
        self.setWindowTitle("Dark Transparent Window")
        self.resize(800, 600)
        self.setFixedSize(640, 360)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        # Logo
        logo = QLabel()
        splash_image_path = resource_path("iCOILD.png")
        pixmap = QPixmap(splash_image_path).scaled(640, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        rounded_img = rounded_pixmap(pixmap, radius = 25)
        logo.setPixmap(rounded_img)
        logo.setAlignment(Qt.AlignCenter)


        layout.addWidget(logo)

        self.setLayout(layout)
        self.setStyleSheet("background-color: #222; border-radius: 20px;")

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller bundle"""
    if getattr(sys, 'frozen', False):  # Check if running from PyInstaller bundle
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        return os.path.join(os.path.abspath("."), relative_path)
    

class UdpDetectionThread(QThread):
    change_pixmap = Signal(np.ndarray)
    image_saved = Signal(str)
    log_message_signal = Signal(str)
    play_sound_callback = None 
    detection_signal = Signal(list)

    def __init__(self, save_folder=None, get_tree_name=None, get_frond_number=None):
        super().__init__()
        self.running = True
        self.save_folder = save_folder
        self.get_tree_name = get_tree_name
        self.get_frond_number = get_frond_number
        self.frame_queue = Queue()

        # Load model once
        model_path = "best.pt"
        self.model = YOLO(model_path)
        self.model.to("cuda")

        self.last_saved_time = 0
        self.latest_frame = None
        self._new_frame_available = False

        if self.save_folder:
            os.makedirs(self.save_folder, exist_ok=True)

    def update_frame(self, frame):
        # Called from MainWindow to push new frame to the thread
        print("[DETECTION] Received frame for detection")
        if not self.running:
            return
        self.frame_queue.put(frame)
        self.latest_frame = frame
        self._new_frame_available = True

    def run(self):
        self.running = True
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except Empty:
                continue
            if self._new_frame_available and self.latest_frame is not None:
                frame = self.latest_frame.copy()
                self._new_frame_available = False

                # Resize and detect
                frame_resized = cv2.resize(frame, (800, 800))
                results = self.model(frame_resized)

                detections = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0].item()
                        if conf > 0.6:
                            detections.append({
                                "class": "Cocolisap",
                                "confidence": conf,
                                "bounding_box": [x1, y1, x2, y2]
                            })
                            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame_resized, f"{conf:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if detections and time.time() - self.last_saved_time > 2:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    tree_name = self.get_tree_name() or "Unknown_Tree"
                    frond_number = self.get_frond_number() or "F0"
                    filename = os.path.join(self.save_folder, f"{tree_name}_{frond_number}_{timestamp}.jpg")
                    cv2.imwrite(filename, frame_resized)

                    self.log_message_signal.emit(f"Saved: {filename}")
                    self.image_saved.emit(filename)
                    self.last_saved_time = time.time()

                    self.save_detection_details({
                        "image": os.path.basename(filename),
                        "timestamp": timestamp,
                        "detections": detections
                    })
                    self.detection_signal.emit(detections)

                    if self.play_sound_callback:
                        self.play_sound_callback()
                        
                self.change_pixmap.emit(frame_resized)

            else:
                self.msleep(10)  # Small sleep to avoid 100% CPU

    def save_detection_details(self, details):
        json_file = os.path.join(self.save_folder, "detection_data.json")
        data = []

        if os.path.exists(json_file):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                pass

        data.append(details)

        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)

    def stop(self):
        self.running = False
        self.wait()

        
class DetectionWorker(QThread):
    progress = Signal(str)
    finished = Signal(str)
    frame_ready = Signal(QImage)  # 👈 Signal to emit the processed frame
    play_sound_callback = None 
    image_saved = Signal(str)

    def __init__(self, video_path, model, confidence_threshold, output_folder, get_tree_name=None, get_frond_number=None):
        super().__init__()
        self.video_path = video_path
        self.model = model
        self.get_tree_name = get_tree_name
        self.get_frond_number = get_frond_number
        self.confidence_threshold = confidence_threshold
        self.output_folder = output_folder

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished.emit("❌ Cannot open video.")
            return

        os.makedirs(self.output_folder, exist_ok=True)
        frame_count = 0
        saved_count = 0
        last_saved_time = time.time()
        save_interval = 2  # in seconds (change this value as needed)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = time.time()
            results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
            boxes = results[0].boxes
            if boxes and boxes.conf is not None and any(boxes.conf > self.confidence_threshold):
                annotated = results[0].plot()

                # ✅ Convert to QImage and emit to main thread
                rgb_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.frame_ready.emit(q_image)

                # Save frame
                if current_time - last_saved_time >= save_interval:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    tree_name = self.get_tree_name() or "Unknown_Tree"
                    frond_number = self.get_frond_number() or "F0"
                    filename = os.path.join(self.output_folder, f"{tree_name}_{frond_number}_{timestamp}.jpg")
                    cv2.imwrite(os.path.join(self.output_folder, filename), annotated)

                    self.image_saved.emit(filename)

                    last_saved_time = current_time
                    saved_count += 1
                    if self.play_sound_callback:
                        self.play_sound_callback()

        cap.release()
        self.finished.emit(f"✅ Detection complete. Processed: {frame_count}, Saved: {saved_count}")

# --- SETTINGS ---
LISTEN_IP = "0.0.0.0"  # listen on all interfaces
LISTEN_PORT = 5005     # must match ESP32
PACKET_SIZE = 1024

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_audio()
        self.thread = None
        self.setWindowTitle("iCOILD")
        self.setGeometry(100, 100, 960, 480)  # Wider window

        model_path = "best.pt"
        self.model = YOLO(model_path)
        
        # Create main container with horizontal layout
        self.main_widget = QWidget()
        self.main_layout = QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)
        
        # Create three equal columns
        self.left_column = QVBoxLayout()
        self.middle_column = QVBoxLayout()
        self.middle_column.setContentsMargins(10, 10, 10, 10)
        self.right_column = QVBoxLayout()
        
        # Set stretch factors to make columns equal width
        self.main_layout.addLayout(self.left_column, 1)
        self.main_layout.addLayout(self.middle_column, 1)
        self.main_layout.addLayout(self.right_column, 1)
        
        # Add video to left column
        self.welcome_banner = QLabel("Welcome to iCOILD APP")
        self.welcome_banner.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # Left align + vertical center
        self.welcome_banner.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
            }
        """)
        self.left_column.addWidget(self.welcome_banner)

        self.video_label = QLabel()
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(450, 253)  # Optional: set a fixed size
        self.video_label.hide()
        self.left_column.addWidget(self.video_label, alignment=Qt.AlignHCenter)

        self.status_label = QLabel("No video selected.")
        self.status_label.setAlignment(Qt.AlignCenter)
        font = self.status_label.font()
        font.setPointSize(16)  # or use setPixelSize(32) for finer control
        self.status_label.setFont(font)
        self.left_column.addWidget(self.status_label)

        self.status_messages = [
            "🔍 Running detection...",
            "🔍 Analyzing Frames...",
            "Please don't close the window...",
            "🔍 Analyzing Frames...",
            "Please don't close the window...",
            "🔍 Running detection...",
        ]

        self.message_index = 0

        # QTimer (not displayed)
        self.status_timer = QTimer(self)
        self.status_timer.setInterval(2000)  # 1 second

        # UDP socket setup
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((LISTEN_IP, LISTEN_PORT))
        self.sock.setblocking(False)  # non-blocking mode

        self.total_packets_expected = 0
        self.received_packets = []
        self.timer = QTimer()
        self.timer.timeout.connect(self.receive_data)
        self.timer.start(1)  # check for packets continuously

        self.source_combo = QComboBox()
        self.source_combo.addItems(["Desktop Camera", "ESP32-CAM"])
        self.source_combo.setFixedHeight(35)
        self.left_column.addWidget(self.source_combo)
        self.source_combo.setStyleSheet("""
            QComboBox {
                padding-left:10px;
                margin-left: 40px;
                margin-right: 40px;
            }
        """)
        self.source_combo.hide()

        #self.scan_button = QPushButton("Scan iCOILD Network")
        #self.scan_button.clicked.connect(self.rescan_icoild_network)
        #self.scan_button.setFixedHeight(35)
        #self.left_column.addWidget(self.scan_button)
        #self.scan_button.setStyleSheet("""
         #   QPushButton{
          #      margin-left: 40px;
           #     margin-right: 40px;
            #}
        #""")

        self.connect_button = QPushButton("Connect")
        self.connect_button.hide()
        self.connect_button.clicked.connect(self.connect_detection)
        self.connect_button.setFixedHeight(35)
        self.left_column.addWidget(self.connect_button)
        self.connect_button.setStyleSheet("""
            QPushButton{
                margin-left: 40px;
                margin-right: 40px;
            }
        """)

        self.browse_button = QPushButton("Upload Video")
        self.detect_button = QPushButton("Start Detection")

        self.browse_button.setFixedHeight(35)
        self.browse_button.setStyleSheet("""
            QPushButton{
                margin-left: 40px;
                margin-right: 40px;
            }
        """)
        self.detect_button.setFixedHeight(35)
        self.detect_button.setStyleSheet("""
            QPushButton{
                margin-left: 40px;
                margin-right: 40px;
            }
        """)
        self.status_label.setStyleSheet("""
             QLabel {
                padding-left: 40px;
            }
        """)

         # Layout
        self.left_column.addWidget(self.browse_button)
        self.left_column.addWidget(self.detect_button)

        # Connect buttons to functions
        self.browse_button.clicked.connect(self.browse_video)
        self.detect_button.clicked.connect(self.start_detection)

        # System Message Box (bottom portion)
        self.log_banner = QLabel("Log System Message")
        self.log_banner.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # Left align + vertical center
        self.log_banner.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 12px;
                font-weight: bold;
                padding: 5px,0, 0, 0;
                border-radius: 5px;
                margin-left: 35px;
            }
        """)
        self.left_column.addWidget(self.log_banner)

        self.system_message_box = QTextEdit()
        self.system_message_box.setReadOnly(True)
        self.system_message_box.setFixedHeight(250)  # Fixed height in pixels
        self.system_message_box.setMinimumWidth(200)  # Fixed height
        self.system_message_box.setStyleSheet("""
             QTextEdit {
            color: #000000;
            background-color: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
            font-family: 'Courier New';
            font-size: 12px;
            margin-left: 35px;
            margin-right: 35px;
            margin-bottom: 35px;
            }
        """)
        self.left_column.addWidget(self.system_message_box, 1)
            
        # Add controls to middle column 
        # Folder Selection Group Box
        self.folder_group = QGroupBox("Output Folder")
        self.folder_layout = QVBoxLayout()

        # Select Folder Button
        self.btn_select_folder = QPushButton("Select Output Folder")
        self.btn_select_folder.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
        self.btn_select_folder.clicked.connect(self.select_output_folder)
        self.btn_select_folder.setFixedHeight(45)
        
        # Selected Path Display
        self.lbl_folder_path = QLabel("No folder selected")
        self.lbl_folder_path.setWordWrap(True)
        self.lbl_folder_path.setStyleSheet("""
            QLabel {
                color: #555;
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                padding: 5px;
                border-radius: 3px;
            }
        """)
    
    # Add to layout
        self.folder_layout.addWidget(self.btn_select_folder)
        self.folder_layout.addWidget(self.lbl_folder_path)
        self.folder_group.setLayout(self.folder_layout)
        
        # Add to middle column (will add Tree/Frond sections later)
        self.middle_column.addWidget(self.folder_group)
        self.middle_column.addStretch()
        
        # Class variable to store path
        self.output_folder_path = ""

        # Tree Management Group
        self.tree_group = QGroupBox("Tree Management")
        self.tree_layout = QVBoxLayout()
        
        # Tree Input
        self.tree_input_layout = QHBoxLayout()
        self.tree_name_input = QLineEdit()
        self.tree_name_input.setPlaceholderText("Enter tree name")
        self.btn_add_tree = QPushButton("Add")
        self.btn_remove_tree = QPushButton("Remove")

        # Tree List
        self.tree_list = QListWidget()
        self.tree_list.setFixedHeight(250)
        self.tree_list.setSelectionMode(QListWidget.SingleSelection)

        self.tree_input_layout.addWidget(self.tree_name_input)
        self.tree_input_layout.addWidget(self.btn_add_tree)
        self.tree_input_layout.addWidget(self.btn_remove_tree)
        
        self.tree_layout.addLayout(self.tree_input_layout)
        self.tree_layout.addWidget(self.tree_list)
        self.tree_group.setLayout(self.tree_layout)

        # Frond Management Group
        self.frond_group = QGroupBox("Frond Management")
        self.frond_layout = QVBoxLayout()
        
        # Frond Controls
        self.frond_control_layout = QHBoxLayout()
        self.btn_add_frond = QPushButton("Add Frond")
        self.btn_remove_frond = QPushButton("Remove")
        self.lbl_next_frond = QLabel("Next: Frond 1")
        
        self.frond_control_layout.addWidget(self.btn_add_frond)
        self.frond_control_layout.addWidget(self.btn_remove_frond)
        self.frond_control_layout.addWidget(self.lbl_next_frond)
        self.frond_control_layout.addStretch()
        
        # Frond List
        self.frond_list = QListWidget()
        self.frond_list.setFixedHeight(250)
        self.frond_list.setSelectionMode(QListWidget.SingleSelection)
        
        # Add to layout
        self.frond_layout.addLayout(self.frond_control_layout)
        self.frond_layout.addWidget(self.frond_list)
        self.frond_group.setLayout(self.frond_layout)
        
        # Add groups to middle column (above the existing stretch)
        self.middle_column.insertWidget(1, self.tree_group)
        self.middle_column.insertWidget(2, self.frond_group)
        
        # Initialize counters
        self.frond_counter = 1
        self.current_tree = ""
        self.current_frond = ""
        
        # Connect signals
        self.btn_add_tree.clicked.connect(self.add_tree)
        self.btn_remove_tree.clicked.connect(self.remove_tree)
        self.btn_add_frond.clicked.connect(self.add_frond)
        self.btn_remove_frond.clicked.connect(self.remove_frond)
        self.tree_list.itemSelectionChanged.connect(self.tree_selected)
        self.frond_list.itemSelectionChanged.connect(self.frond_selected)
                
        # Add placeholder for right column
        #self.right_column.addWidget(QLabel("Object Detection Results"))
        #self.detection_results = QLabel("Detected objects will appear here")
        #self.detection_results.setAlignment(Qt.AlignTop)
        #self.right_column.addWidget(self.detection_results)

        #Image Display Group

        self.current_folder = ""

        self.image_group = QGroupBox("Image Preview")
        self.image_layout = QVBoxLayout()
        self.image_details = QHBoxLayout()
        self.image_details.setSpacing(20)
        self.image_details.setContentsMargins(0, 10, 0, 0)

        self.image_label = QLabel("No Image Selected")
        self.image_label.setFixedSize(300, 300)  # Fix image preview size
        self.image_label.setStyleSheet("border: 1px solid black;")  # Add border for visibility
        self.image_label.setAlignment(Qt.AlignCenter)  # Center the image inside the label
        self.image_label.setScaledContents(True)  # Ensure it scales inside the label

        # Labels to display detection details
        self.detection_name_label = QLabel("File Name: None")
        self.detection_name_label.setContentsMargins(20, 0, 0, 0)
        self.confidence_label = QLabel("Detection Strength: N/A")
        self.confidence_label.setContentsMargins(20, 0, 0, 0)
        self.object_count_label = QLabel("Cocolisap Detected: 0")
        self.object_count_label.setContentsMargins(20, 0, 0, 0)

        self.detection_name_label.hide()
        self.confidence_label.hide()
        self.object_count_label.hide()

        # Add the labels to the details layout
        self.image_details.addWidget(self.detection_name_label, alignment=Qt.AlignTop)
        self.image_details.addWidget(self.confidence_label, alignment=Qt.AlignTop)
        self.image_details.addWidget(self.object_count_label, alignment=Qt.AlignTop)

        self.image_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        self.image_group.setLayout(self.image_layout)
        self.image_layout.addLayout(self.image_details, stretch=0)

        self.right_column.addWidget(self.image_group)
        
        # Add stretch to push content to top

        self.file_group = QGroupBox("File Browser")
        self.file_layout = QVBoxLayout()

        self.table_layout = QHBoxLayout()

        self.table_files_left = QTableWidget()
        self.table_files_left.setColumnCount(2)
        self.table_files_left.setHorizontalHeaderLabels(["File Name", "Size (KB)"])
        self.table_files_left.setSortingEnabled(True)
        self.table_files_left.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_files_left.setFixedHeight(350)
        self.table_files_left.cellClicked.connect(self.display_image)
        
        self.table_files_right = QTableWidget()
        self.table_files_right.setColumnCount(2)
        self.table_files_right.setHorizontalHeaderLabels(["Tree Name", "Status"])
        self.table_files_right.setSortingEnabled(True)
        self.table_files_right.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_files_right.setFixedSize(200, 350)

        self.table_layout.addWidget(self.table_files_left)
        self.table_layout.addWidget(self.table_files_right)

        self.file_layout.addLayout(self.table_layout)  # Add horizontal table layout
        self.file_group.setLayout(self.file_layout)
        self.file_group.setFixedHeight(390)

        # Add file group to the main layout
        self.right_column.addWidget(self.file_group)

        # Video thread

        self.detection_thread = None

        self.esp32_url = 5005
        # Initialize NetworkScanner
        #self.scanner = NetworkScanner()
        #self.scanner.ip_found.connect(self.connect_to_stream)
        #self.scanner.log_message_signal.connect(self.log_system_message)

        # Start scanning automatically when the app starts
        #self.start_scan()

        self.load_data()

        self.detection_file = resource_path("detection_data.json")

        self.update_tree_status_table()

        # Ensure save on exit
        self.destroyed.connect(self.save_data)
        self.tree_list.itemSelectionChanged.connect(self.save_data)
        self.frond_list.itemSelectionChanged.connect(self.save_data)

        

    #def start_scan(#self):
     #   """Start scanning for ESP32 devices automatically"""
      #  self.scanner.start()  # This triggers the scanning process
       # print("Starting network scan...")
       # self.log_system_message(f"Scanning ESP32 network...")
    
    #def connect_to_stream(self, ip):
       # print(f"Received IP for streaming: {ip}")
        # Set the ESP32 URL dynamically based on the discovered IP
      #  self.esp32_url = f"http://{ip}:81/stream"
      #  print(f"Connected to ESP32 at: {self.esp32_url}")
        #self.log_system_message(f"Connected to ESP32 at: {self.esp32_url}")

    def connect_detection(self):
        self.log_system_message(f"Detection start")
        if not self.output_folder_path:
            self.log_system_message("Please select an output folder first to save images.")
            return  # Prevent starting the video stream if no folder is selected
        self.detection_thread = UdpDetectionThread(
            save_folder=self.output_folder_path,
            get_tree_name=self.get_current_tree_name,
            get_frond_number=self.get_current_frond_number
        )
        self.detection_thread.image_saved.connect(self.handle_image_saved)
        self.detection_thread.play_sound_callback = self.play_detected_sound  
        self.detection_thread.log_message_signal.connect(self.log_system_message)
        self.detection_thread.detection_signal.connect(self.update_image_details)
        self.detection_thread.start()
        
    def toggle_connection(self):
        """Switch between Desktop Camera and ESP32-CAM"""
        if hasattr(self, "video_thread") and self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()  # Ensure thread stops before restarting

        selected_source = self.source_combo.currentText()

        if selected_source == "ESP32-CAM":
            # Check if we already have the ESP32 URL set
            if not hasattr(self, "esp32_url") or not self.esp32_url:
                print("No ESP32 stream URL available. Please scan first.")
                self.log_system_message(f"No ESP32 stream URL available. Please scan first.")
                return  # Don't continue if ESP32 URL is not set

            self.start_video_stream(self.esp32_url)
        else:
            self.start_video_stream(0)  # 0 for the default desktop camera
            self.log_system_message("Default desktop camera")

    #def rescan_icoild_network(self):
        #self.start_scan()
            
    def start_video_stream(self, source):
        # Check if a folder is selected before starting the video stream
        if not self.output_folder_path:
            self.log_system_message("Please select an output folder first to save images.")
            return  # Prevent starting the video stream if no folder is selected
        if self.video_thread:
            self.video_thread.stop()
            #source,
        self.video_thread = UdpDetectionThread(udp_port=5005 ,save_folder=self.output_folder_path, get_tree_name=self.get_current_tree_name,
    get_frond_number=self.get_current_frond_number)
        self.video_thread.change_pixmap.connect(self.update_image)
        self.video_thread.image_saved.connect(self.handle_image_saved)
        self.video_thread.play_sound_callback = self.play_detected_sound  
        self.video_thread.log_message_signal.connect(self.log_system_message)
        self.video_thread.detection_signal.connect(self.update_image_details)

        self.video_thread.start()

    def get_current_tree_name(self):
        item = self.tree_list.currentItem()
        return item.text().replace(" ", "_") if item else "Unknown_Tree"

    def get_current_frond_number(self):
        item = self.frond_list.currentItem()
        return item.text().replace("Frond ", "F") if item else "F0"

    def handle_image_saved(self, folder_path):
        self.load_files(folder_path)  # Update the file list
        self.update_tree_status_table()  # Update the tree status

    #Handle update message
    def log_system_message(self, message):
        timestamp = QDateTime.currentDateTime().toString("[hh:mm:ss]")
        self.system_message_box.append(f"{timestamp} {message}")
        # Auto-scroll to bottom
        self.system_message_box.verticalScrollBar().setValue(
            self.system_message_box.verticalScrollBar().maximum()
        )

    def select_output_folder(self):
        """Open folder dialog and store selection"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            QDir.homePath(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if folder:
            self.output_folder_path = folder
            # Display abbreviated path if too long
            display_path = folder if len(folder) < 40 else f"...{folder[-37:]}"
            self.lbl_folder_path.setText(display_path)
            self.lbl_folder_path.setToolTip(folder)  # Show full path on hover
            
            # Enable/disable dependent features
            self.update_folder_dependent_controls(True)
            
            # Log system message
            self.log_system_message(f"Output folder set to: {folder}")
            self.save_data()
            self.load_files(folder)
            self.current_folder = folder
            self.update_tree_status_table()
        else:
            QMessageBox.warning(self, "Folder Required", "Please select an output folder.")

    def update_folder_dependent_controls(self, enabled):
        """Enable/disable controls that require valid folder"""
        # You'll connect these later for Tree/Frond features
        pass

    def add_tree(self):
        tree_name = self.tree_name_input.text().strip()
        if tree_name and tree_name not in self.tree_data:
            self.tree_list.addItem(tree_name)
            self.tree_data[tree_name] = []
            self.tree_name_input.clear()
            self.save_data()
            self.log_system_message(f"Tree added: {tree_name}")

    def remove_tree(self):
        selected_item = self.tree_list.currentItem()
        if selected_item:
            tree_name = selected_item.text()
            del self.tree_data[tree_name]
            self.tree_list.takeItem(self.tree_list.row(selected_item))
            self.frond_list.clear()
            self.current_tree = ""
            self.save_data()
            self.log_system_message(f"Tree removed: {tree_name}")

    def tree_selected(self):
        selected_item = self.tree_list.currentItem()
        if selected_item:
            self.current_tree = selected_item.text()
            self.load_fronds()
            self.log_system_message(f"SELECTED TREE: {self.current_tree}")
    
    def frond_selected(self):
        selected_item = self.frond_list.currentItem()
        if selected_item:
            self.current_frond = selected_item.text()
            self.log_system_message(f"SELECTED FROND: {self.current_frond}")

    def add_frond(self):
        if not self.current_tree:
            return
        frond_name = f"Frond {len(self.tree_data[self.current_tree]) + 1}"
        self.tree_data[self.current_tree].append(frond_name)
        self.frond_list.addItem(frond_name)
        self.lbl_next_frond.setText(f"Next: Frond {len(self.tree_data[self.current_tree]) + 1}")
        self.save_data()
        self.log_system_message(f"Frond added: {frond_name}")

    def remove_frond(self):
        selected_item = self.frond_list.currentItem()
        if selected_item and self.current_tree:
            frond_name = selected_item.text()
            self.tree_data[self.current_tree].remove(frond_name)
            self.frond_list.takeItem(self.frond_list.row(selected_item))
            self.lbl_next_frond.setText(f"Next: Frond {len(self.tree_data[self.current_tree]) + 1}")
            self.save_data()
            self.log_system_message(f"Frond removed: {frond_name}")

    def load_fronds(self):
        """Load fronds for the selected tree"""
        self.frond_list.clear()
        if self.current_tree in self.tree_data:
            self.frond_list.addItems(self.tree_data[self.current_tree])
            self.lbl_next_frond.setText(f"Next: Frond {len(self.tree_data[self.current_tree]) + 1}")

    def save_data(self):
        """Save tree and frond data to a JSON file"""
        tree_data_path = resource_path("tree_data.json")
        with open(tree_data_path, "w") as file:
            json.dump(self.tree_data, file)

    def load_data(self):
        """Load tree and frond data from a JSON file"""
        tree_data_path = resource_path("tree_data.json")
        try:
            with open(tree_data_path, "r") as file:
                self.tree_data = json.load(file)
                self.tree_list.addItems(self.tree_data.keys())
        except (FileNotFoundError, json.JSONDecodeError):
            self.tree_data = {}
    
    def log_system_message(self, message):
        """Add timestamped messages to the system message box
        
        Args:
            message (str): The message to display
        """
        try:
            # Get current timestamp
            timestamp = QDateTime.currentDateTime().toString("[hh:mm:ss]")
            
            # Format the message with timestamp
            formatted_message = f"{timestamp} {message}"
            
            # Append to the QTextEdit
            self.system_message_box.append(formatted_message)
            
            # Auto-scroll to bottom
            scrollbar = self.system_message_box.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
            # Optional: Print to console for debugging
            print(formatted_message)
            
        except Exception as e:
            print(f"Error logging message: {e}")

    def load_files(self, folder_path):
        """Load files from selected folder into the table"""
        self.table_files_left.setRowCount(0)  # Clear table

        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path) // 1024  # Convert to KB
                row_position = self.table_files_left.rowCount()
                self.table_files_left.insertRow(row_position)
                self.table_files_left.setItem(row_position, 0, QTableWidgetItem(file))
                self.table_files_left.setItem(row_position, 1, QTableWidgetItem(str(file_size)))

    def display_image(self, row, column):
        """Show image in the QLabel when a file is clicked"""
        file_name = self.table_files_left.item(row, 0).text()
        file_path = os.path.join(self.current_folder, file_name)

        if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                self.image_label.setPixmap(pixmap.scaled(300, 300))  # Resize to fit label
            else:
                self.image_label.setText("Invalid Image")
        else:
            self.image_label.setText("Not an Image")

        # Now load the detection details
        detection_data_path = "detection_data.json"
        json_file = os.path.join(self.output_folder_path, detection_data_path)

        if os.path.exists(json_file):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                # Find the entry for the selected image
                selected_image_details = None
                for entry in data:
                    if entry["image"] and os.path.basename(entry["image"]) == os.path.basename(file_path):
                        selected_image_details = entry
                        break

                if selected_image_details:
                    # Update the details section with the detection data
                    self.update_image_details(selected_image_details["image"],selected_image_details["detections"])
                else:
                    self.clear_image_details()

            except json.JSONDecodeError:
                print("Error loading detection data.")
        else:
            print(f"No detection data found at {json_file}")
 
    def clear_image_details(self):
        """Clear the image details in case no detection data is found."""
        self.detection_name_label.setText("Filename: None")
        self.confidence_label.setText("Detection Strength: N/A")
        self.object_count_label.setText("Cocolisap Detected: 0")
        self.image_label.clear()


    def update_tree_status_table(self):
        """Update the second table with tree names and their status based on file names."""
        if not self.current_folder:
            return

        self.table_files_right.setRowCount(0)  # Clear the table

        # Get all image filenames in the selected folder
        files = os.listdir(self.current_folder)

        # Dictionary to track unique frond counts for each tree
        tree_frond_count = {}

        # Adjusted regex pattern to match filenames
        pattern = re.compile(r"^([\w-]+)_F(\d+)_\d{8}_\d{6}(?:_\d+)?\.jpg$")

        for file in files:
            file = file.strip()  # remove whitespace
            if file.lower().endswith(".jpg"):  # optionally normalize extension
                match = pattern.match(file)
                if match:
                    tree_name = match.group(1)
                    frond_number = match.group(2)

                    if tree_name not in tree_frond_count:
                        tree_frond_count[tree_name] = set()

                    tree_frond_count[tree_name].add(frond_number)
                else:
                    print(f"Skipping file (no match): {file}")  # Debugging
            else:
                print(f"Skipping non-JPG file: {file}")

        # Populate the second table
        for index in range(self.tree_list.count()):
            tree_item = self.tree_list.item(index)
            tree_name_with_spaces = tree_item.text().strip()
            tree_name_key = tree_name_with_spaces.replace(" ", "_")  # Convert to match filenames

            # Determine status based on unique frond count
            unique_fronds = tree_frond_count.get(tree_name_key, set())
            frond_count = len(unique_fronds)

            if frond_count == 0:
                status = "⚪ None"
            elif frond_count < 10:
                status = "🟢 LOW"
            elif frond_count == 10:
                status = "🟠 MODERATE"
            else:
                status = "🔴 HIGH"

            # Add row to the second table
            row = self.table_files_right.rowCount()
            self.table_files_right.insertRow(row)
            self.table_files_right.setItem(row, 0, QTableWidgetItem(tree_name_with_spaces))  # Show original name
            self.table_files_right.setItem(row, 1, QTableWidgetItem(status))

    def update_image_details(self,image_name, detected_objects):
        """Update the details section with detection info."""
        print(f"Updating details: {len(detected_objects)} objects detected")
        if detected_objects:
            self.detection_name_label.setText(f"Filename: {image_name}")
            #self.detection_name_label.setText(f"Object: {detected_objects[0]['class']}")
            self.confidence_label.setText(f"Detection Strength: {detected_objects[0]['confidence']* 100:.2f}%")
            self.object_count_label.setText(f"Cocolisap Detected: {len(detected_objects)}")
        else:
            self.detection_name_label.setText("Filename: None")
            self.confidence_label.setText("Detection Strenght: N/A")
            self.object_count_label.setText("Cocolisap Detected: 0")


    def init_audio(self):
        pygame.mixer.init()
        audio_path = resource_path("cocolisap_notif.wav")
        try:
            self.detected_sound = pygame.mixer.Sound(audio_path)
        except pygame.error as e:
            print(f"Audio load failed: {e}")
            self.log_system_message(f"Audio load failed: {e}")
            self.detected_sound = None

    def play_detected_sound(self):
        if self.detected_sound:
            self.detected_sound.play()


    @Slot(np.ndarray)
    def update_image(self, frame):
        # Resize frame to match QLabel
        label_width = self.video_label.width()
        label_height = self.video_label.height()
        display_frame = cv2.resize(frame, (label_width, label_height), interpolation=cv2.INTER_LINEAR)

        # Convert to RGB for Qt
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimage = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Set it to the label
        self.video_label.setPixmap(QPixmap.fromImage(qimage))

    def receive_data(self):
        try:
            while True:
                data, _ = self.sock.recvfrom(PACKET_SIZE + 10)

                # First packet contains the number of total packets
                if len(data) <= 4:
                    self.total_packets_expected = int.from_bytes(data, byteorder='little')
                    self.received_packets = []
                else:
                    self.received_packets.append(data)

                # When all packets are received, rebuild the image
                if self.total_packets_expected > 0 and len(self.received_packets) >= self.total_packets_expected:
                    full_data = b''.join(self.received_packets)
                    np_arr = np.frombuffer(full_data, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    # Resize frame to match QLabel
                    label_width = self.video_label.width()
                    label_height = self.video_label.height()
                    frame = cv2.resize(frame, (label_width, label_height), interpolation=cv2.INTER_LINEAR)

                    if frame is not None:
                        # Send frame to detection thread
                        if self.detection_thread and self.detection_thread.isRunning():
                            self.detection_thread.update_frame(frame)

                        # Display in video label
                        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_image.shape
                        qimg = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
                        self.video_label.setPixmap(QPixmap.fromImage(qimg))

                    # Reset for next frame
                    self.total_packets_expected = 0
                    self.received_packets = []

        except BlockingIOError:
            # No data available — normal in non-blocking mode
            pass

    def browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            self.video_path = file_path
            self.status_label.setText(f"Selected: {os.path.basename(file_path)}")

    def start_detection(self):
        if not self.video_path:
            self.status_label.setText("⚠️ Please select a video first.")
            return

        if not hasattr(self, 'output_folder_path') or not self.output_folder_path:
            self.status_label.setText("⚠️ Please select an output folder.")
            return

        confidence_threshold = 0.7
        self.message_index = 0

        self.status_timer.timeout.connect(self.update_status_message)
        self.status_timer.start()

        # Create and start the detection thread
        self.worker = DetectionWorker(
            self.video_path,
            self.model,  # Model is already on GPU
            confidence_threshold,
            self.output_folder_path,
            get_tree_name=self.get_current_tree_name,
            get_frond_number=self.get_current_frond_number
        )

        self.worker.play_sound_callback = self.play_detected_sound  
        self.worker.progress.connect(self.status_label.setText)
        self.worker.image_saved.connect(self.handle_image_saved)
        self.worker.finished.connect(self.status_label.setText)
        self.worker.frame_ready.connect(self.update_video_label)
        self.worker.start()
    
    def update_video_label(self, q_image):
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def detect_cocolisap_from_video(self, video_path, model, confidence_threshold, output_folder):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Cannot open video.")
            return

        os.makedirs(output_folder, exist_ok=True)

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            results = model.predict(frame, conf=confidence_threshold, verbose=False)
            boxes = results[0].boxes
            if boxes and boxes.conf is not None and any(boxes.conf > confidence_threshold):
                annotated = results[0].plot()
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                cv2.imwrite(os.path.join(output_folder, f"detected_{frame_count}_{timestamp}.jpg"), annotated)
                saved_count += 1

        cap.release()
        print(f"Done: Processed {frame_count}, Saved {saved_count} frames.")

    def update_status_message(self):
        if self.message_index < len(self.status_messages):
            self.status_label.setText(self.status_messages[self.message_index])
            self.message_index += 1
        else:
            self.status_timer.stop()

    def closeEvent(self, event):
        if self.detection_thread:
            self.detection_thread.stop()
        event.accept()

    def goodbye():
        print("The app is exiting — atexit triggered")

    atexit.register(goodbye)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    splash = LoadingScreen()
    splash.show()

    main_window = MainWindow()
    QTimer.singleShot(3000, splash.close)
    QTimer.singleShot(3000, main_window.show)

    sys.exit(app.exec())
