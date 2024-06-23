import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QProgressBar
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QBrush, QPalette
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

class AnalysisThread(QThread):
    finished = pyqtSignal(list)
    progress = pyqtSignal(int)  # Yeni sinyal ekleme

    def __init__(self, video_path):
        QThread.__init__(self)
        self.video_path = video_path

    def run(self):
        video_frames = read_video(self.video_path)
        num_frames = len(video_frames)
        tracker = Tracker('models_yolov5/weights/best.pt')
        tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
        
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

        for frame_num, player_tracks in enumerate(tracks['players']):
            for player_id, track in player_tracks.items():
                team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

            # İlerleme bilgisini güncelle
            self.progress.emit(int((frame_num + 1) / num_frames * 90))

        player_assigner = PlayerBallAssigner()
        team_ball_control = []
        for frame_num, player_track in enumerate(tracks['players']):
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'] if assigned_player != -1 else None)

        output_video_frames = tracker.draw_annotations(video_frames, tracks, np.array(team_ball_control))
        self.progress.emit(100)
        self.finished.emit(output_video_frames)

class FootballAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.loadDemoVideo("input_videos/0bfacc_1.mp4")  # Demo video yolu

    def initUI(self):
        self.setWindowTitle('Futbol Analiz Uygulaması')
        self.setGeometry(100, 100, 1024, 680)
        
        # Set the background image
        palette = self.palette()
        brush = QBrush(QPixmap('input_videos/test_photo.jpg').scaled(self.size(), Qt.IgnoreAspectRatio))
        palette.setBrush(QPalette.Window, brush)
        self.setPalette(palette)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Video Widget
        self.videoWidget = QVideoWidget()
        self.videoWidget.setMinimumSize(800, 450)
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.videoWidget)
        self.videoWidget.setAspectRatioMode(Qt.IgnoreAspectRatio)  # Videonun boyut oranını yok say
        layout.addWidget(self.videoWidget)
        layout.addStretch()

        # Buttons with updated style
        self.btn_load = QPushButton('Video Yükle')
        self.btn_load.setStyleSheet("background-color: green; color: white; font-weight: bold;")
        self.btn_load.clicked.connect(self.openFileDialog)
        layout.addWidget(self.btn_load)

        self.btn_analyze = QPushButton('Videoyu Analiz Et')
        self.btn_analyze.setStyleSheet("background-color: blue; color: white; font-weight: bold;")
        self.btn_analyze.clicked.connect(self.analyzeVideo)
        self.btn_analyze.setEnabled(False)
        layout.addWidget(self.btn_analyze)

        self.btn_play = QPushButton('Analiz Edilmiş Videoyu Oynat')
        self.btn_play.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        self.btn_play.clicked.connect(self.playAnalyzedVideo)
        self.btn_play.setEnabled(False)
        layout.addWidget(self.btn_play)

        self.btn_pause = QPushButton('Durdur/Başlat')
        self.btn_pause.setStyleSheet("background-color: orange; color: white; font-weight: bold;")
        self.btn_pause.clicked.connect(self.pauseOrResumeVideo)
        layout.addWidget(self.btn_pause)

        # Progress Bar
        self.progress = QProgressBar(self)
        self.progress.setMaximum(100)  # Maksimum değeri 100 olarak ayarla
        self.progress.setValue(0)  # Başlangıçta %0 olarak ayarla
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }

            QProgressBar::chunk {
                background-color: #05B8CC;
                width: 20px;
            }
        """)

        layout.addWidget(self.progress)

        self.label = QLabel('Videoyu yükleyin ve analizi başlatın.')
        self.label.setStyleSheet("color: white; font-weight: bold;")
        layout.addWidget(self.label)


    def loadDemoVideo(self, video_path):
        self.video_path = video_path
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
        self.player.pause()  # Videonun oynamaması için başlangıçta durdur
        self.label.setText('Demo video yüklendi: ' + video_path)
        self.btn_analyze.setEnabled(True)
    def openFileDialog(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Video Seç", "", "Video Files (*.mp4 *.avi)")
        if fileName:
            self.video_path = fileName
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(fileName)))
            self.player.play()  # Videoyu otomatik olarak başlat
            self.label.setText('Video yüklendi: ' + fileName)
            self.btn_analyze.setEnabled(True)

    def analyzeVideo(self):
        self.player.stop()
        self.progress.setRange(0, 100)  # % olarak ayarla
        self.progress.setValue(0)
        self.analysis_thread = AnalysisThread(self.video_path)
        self.analysis_thread.finished.connect(self.onAnalysisComplete)
        self.analysis_thread.progress.connect(self.progress.setValue)  # Sinyal bağlantısı
        self.analysis_thread.start()

    def onAnalysisComplete(self, output_video_frames):
        save_video(output_video_frames, 'output_videos/output_video.avi')
        self.analyzed_video_path = 'output_videos/output_video.avi'
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.analyzed_video_path)))
        self.player.pause()
        self.btn_play.setEnabled(True)
        self.label.setText('Analiz tamamlandı, "Analiz Edilmiş Videoyu Oynat" butonuna basarak videoyu başlatın.')

    def playAnalyzedVideo(self):
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.analyzed_video_path)))
        self.player.play()

    def pauseOrResumeVideo(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()
    def mediaStateChanged(self, state):
        if state == QMediaPlayer.LoadedState:
            self.player.play()
        elif state == QMediaPlayer.StoppedState:
            self.label.setText('Video durdu.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FootballAnalysisApp()
    ex.show()
    sys.exit(app.exec_())