from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
import cv2


class Ui_imageWindow(object):

    def setupUi(self, imageWindow, h, w):

        imageWindow.setObjectName("imageWindow")
        imageWindow.resize(w, h)
        imageWindow.setWindowTitle("Image Window")
        self.centralwidget = QtWidgets.QWidget(imageWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pixmapImage_label = QtWidgets.QLabel(self.centralwidget)
        self.pixmapImage_label.setGeometry(QtCore.QRect(0, 0, w, h))
        self.pixmapImage_label.setText("")
        self.pixmapImage_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter)
        self.pixmapImage_label.setObjectName("pixmapImage_label")
        QtCore.QMetaObject.connectSlotsByName(imageWindow)

    def setImage(self, image, h, w):
        print(h, w)
        self.image = self._resize(image.copy(), [w, h])
        try:
            print(f"image shape - {self.image.shape}")
            if len(self.image.shape) == 2:
                _, lastW = self.image.shape
                lastCh = 1
                imageFormat = QtGui.QImage.Format.Format_Grayscale8
            else:
                _, lastW, lastCh = self.image.shape
                imageFormat = QtGui.QImage.Format.Format_BGR888
            bytesPline = lastCh * lastW
            imageInWindow = QtGui.QImage(
                self.image.data, self.image.shape[1], self.image.shape[0], bytesPline, imageFormat)
            pixmapImage = QtGui.QPixmap.fromImage(imageInWindow)
            self.pixmapImage_label.setPixmap(pixmapImage)
            self.pixmapImage_label.mousePressEvent = self.get_event

        except Exception as e:
            print(f"Image can't be displayed, because {e}")

    def _resize(self, image=None, resolution=None):
        width = resolution[0]
        height = resolution[1]
        h, w = image.shape[:2]
        print([height, width], [h, w])
        dimensions = []
        if w > h:
            if w > width:
                ratio = w/width
                w /= ratio
                dimensions = [int(w), int(h/ratio)]
            else:
                ratio = width/w
                w *= ratio
                dimensions = [int(w), int(h*ratio)]
        else:
            if h > height:
                ratio = h/height
                h /= ratio
                dimensions = [int(w/ratio), int(h)]
            else:
                ratio = height/h
                h *= ratio
                dimensions = [int(w*ratio), int(h)]
            print(dimensions)
        return cv2.resize(image.copy(), dimensions)

    def get_event(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            file_path = QtWidgets.QFileDialog.getSaveFileName(
                directory="./", filter="jpg Files (*.jpg)")[0]
            cv2.imwrite(file_path + '.jpg', self.image)


if __name__ == "__main__":

    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_imageWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
