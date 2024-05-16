from textwrap import indent
from PyQt6 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
from imageWindow import Ui_imageWindow


class QLabelClickable(QtWidgets.QLabel):
    
    clicked = QtCore.pyqtSignal(str)
    
    def __init__(self, parent=None):
        super(QLabelClickable, self).__init__(parent)

    def mousePressEvent(self, event):
        self.mouseEvent = "click"
    
    def mouseReleaseEvent(self, event):
        if self.mouseEvent == "click":
            QtCore.QTimer.singleShot(QtWidgets.QApplication.instance().doubleClickInterval(),
                              self.performSingleClickAction)
        else:
            self.clicked.emit(self.mouseEvent)
    
    def mouseDoubleClickEvent(self, event):
        self.mouseEvent = "double click"
    
    def performSingleClickAction(self):
        if self.mouseEvent == "click":
            self.clicked.emit(self.mouseEvent)


class CustomDialog(QtWidgets.QDialog):
    
    def __init__(self):
    
        super().__init__()

        self.setWindowTitle("Warning")

        QBtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel

        self.buttonBox = QtWidgets.QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QtWidgets.QVBoxLayout()
        message = QtWidgets.QLabel("If you load\nall unsaved data will be lost")
        message.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)


class Ui_MainWindow(object):    
    
    action_dict = {}
    image_previous_name = ''
    
    
    def setupUi(self, MainWindow):        
        self.MainWindow = MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1400, 500)
        MainWindow.setWindowOpacity(1.0)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.setupButtonScrollArea()
        self.setupButtonsConnections()
        self.setupActionsScrollArea()
        self.setupElementsScrollArea()
        self.setupImageLabelArea()
        MainWindow.setCentralWidget(self.centralwidget)
        MainWindow.setWindowTitle("Image Processing with OpenCV")

        # Button for Choose image
        
        self.pushButton = QtWidgets.QPushButton(MainWindow)
        self.pushButton.setGeometry(QtCore.QRect(710, 465, 100, 25))
        self.pushButton.setObjectName("chooseFile")
        self.pushButton.setText("Choose image")
        self.pushButton.clicked.connect(self.chooseFile)
        
        # Button for Generate and Save .py filtering script
        
        self.pushButton = QtWidgets.QPushButton(MainWindow)
        self.pushButton.setGeometry(QtCore.QRect(820, 465, 150, 25))
        self.pushButton.setObjectName("saveFilteringScript")
        self.pushButton.setText("Save filtering script")
        self.pushButton.clicked.connect(self.saveFiltering)
        
        # Button for Load .py filtering script
        
        self.pushButton = QtWidgets.QPushButton(MainWindow)
        self.pushButton.setGeometry(QtCore.QRect(980, 465, 150, 25))
        self.pushButton.setObjectName("loadFilteringScript")
        self.pushButton.setText("Load filtering script")
        self.pushButton.clicked.connect(self.loadFiltering)


    def setupButtonScrollArea(self):        
        self.buttons_scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.buttons_scrollArea.setGeometry(QtCore.QRect(10, 10, 180, 480))
        self.buttons_scrollArea.setMaximumSize(QtCore.QSize(200, 10000))
        self.buttons_scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.buttons_scrollArea.setWidgetResizable(True)
        self.buttons_scrollArea.setObjectName("buttons_scrollArea")
        self.buttons_scrollAreaWidgetContents = QtWidgets.QWidget()
        self.buttons_scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 178, 478))
        self.buttons_scrollAreaWidgetContents.setObjectName("buttons_scrollAreaWidgetContents")
        self.buttons_verticalLayout_main = QtWidgets.QVBoxLayout(self.buttons_scrollAreaWidgetContents)
        self.buttons_verticalLayout_main.setObjectName("verticalLayout_3")
        self.buttons_verticalLayout = QtWidgets.QVBoxLayout()
        self.buttons_verticalLayout.setObjectName("buttons_verticalLayout")
        self.button_cvtColor = QtWidgets.QPushButton(self.buttons_scrollAreaWidgetContents)
        self.button_cvtColor.setMinimumSize(QtCore.QSize(140, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_cvtColor.setFont(font)
        self.button_cvtColor.setObjectName("button_cvtColor")
        self.buttons_verticalLayout.addWidget(self.button_cvtColor)
        self.button_gaussianBlur = QtWidgets.QPushButton(self.buttons_scrollAreaWidgetContents)
        self.button_gaussianBlur.setMinimumSize(QtCore.QSize(140, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_gaussianBlur.setFont(font)
        self.button_gaussianBlur.setObjectName("button_gaussianBlur")
        self.buttons_verticalLayout.addWidget(self.button_gaussianBlur)
        self.button_medianBlur = QtWidgets.QPushButton(self.buttons_scrollAreaWidgetContents)
        self.button_medianBlur.setMinimumSize(QtCore.QSize(140, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_medianBlur.setFont(font)
        self.button_medianBlur.setObjectName("button_medianBlur")
        self.buttons_verticalLayout.addWidget(self.button_medianBlur)
        self.button_threshold = QtWidgets.QPushButton(self.buttons_scrollAreaWidgetContents)
        self.button_threshold.setMinimumSize(QtCore.QSize(140, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_threshold.setFont(font)
        self.button_threshold.setObjectName("button_threshold")
        self.buttons_verticalLayout.addWidget(self.button_threshold)
        self.button_adaptiveThreshold = QtWidgets.QPushButton(self.buttons_scrollAreaWidgetContents)
        self.button_adaptiveThreshold.setMinimumSize(QtCore.QSize(140, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_adaptiveThreshold.setFont(font)
        self.button_adaptiveThreshold.setObjectName("button_adaptiveThreshold")
        self.buttons_verticalLayout.addWidget(self.button_adaptiveThreshold)
        self.button_filter2D = QtWidgets.QPushButton(self.buttons_scrollAreaWidgetContents)
        self.button_filter2D.setMinimumSize(QtCore.QSize(140, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_filter2D.setFont(font)
        self.button_filter2D.setObjectName("button_filter2D")
        self.buttons_verticalLayout.addWidget(self.button_filter2D)
        self.button_bilateralFilter = QtWidgets.QPushButton(self.buttons_scrollAreaWidgetContents)
        self.button_bilateralFilter.setMinimumSize(QtCore.QSize(140, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_bilateralFilter.setFont(font)
        self.button_bilateralFilter.setObjectName("button_bilateralFilter")
        self.buttons_verticalLayout.addWidget(self.button_bilateralFilter)
        self.button_createCLAHE = QtWidgets.QPushButton(self.buttons_scrollAreaWidgetContents)
        self.button_createCLAHE.setMinimumSize(QtCore.QSize(140, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_createCLAHE.setFont(font)
        self.button_createCLAHE.setObjectName("button_createCLAHE")
        self.buttons_verticalLayout.addWidget(self.button_createCLAHE)
        self.button_morphologyEx = QtWidgets.QPushButton(self.buttons_scrollAreaWidgetContents)
        self.button_morphologyEx.setMinimumSize(QtCore.QSize(140, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_morphologyEx.setFont(font)
        self.button_morphologyEx.setObjectName("button_morphologyEx")
        self.buttons_verticalLayout.addWidget(self.button_morphologyEx)
        self.button_erode = QtWidgets.QPushButton(self.buttons_scrollAreaWidgetContents)
        self.button_erode.setMinimumSize(QtCore.QSize(140, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_erode.setFont(font)
        self.button_erode.setObjectName("button_erode")
        self.buttons_verticalLayout.addWidget(self.button_erode)
        self.button_dilate = QtWidgets.QPushButton(self.buttons_scrollAreaWidgetContents)
        self.button_dilate.setMinimumSize(QtCore.QSize(140, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_dilate.setFont(font)
        self.button_dilate.setObjectName("button_dilate")
        self.buttons_verticalLayout.addWidget(self.button_dilate)
        
        self.button_cvtColor.setText("Add cvtColor")
        self.button_gaussianBlur.setText("Add GaussianBlur")
        self.button_medianBlur.setText("Add MedianBlur")
        self.button_threshold.setText("Add Threshold")
        self.button_adaptiveThreshold.setText("Add AdaptiveThreshold")
        self.button_filter2D.setText("Add Filter2D")
        self.button_bilateralFilter.setText("Add BilateralFilter")
        self.button_createCLAHE.setText("Add CreateCLAHE")
        self.button_morphologyEx.setText("Add MorphologyEx")
        self.button_erode.setText("Add Erode")
        self.button_dilate.setText("Add Dilate")
        
        self.buttons_verticalLayout_main.addLayout(self.buttons_verticalLayout)
        self.buttons_scrollArea.setWidget(self.buttons_scrollAreaWidgetContents)


    def setupButtonsConnections(self):        
        self.button_cvtColor.clicked.connect(self.cvtColor_add_action)
        self.button_gaussianBlur.clicked.connect(self.gaussianBlur_add_action)
        self.button_medianBlur.clicked.connect(self.medianBlur_add_action)
        self.button_threshold.clicked.connect(self.threshold_add_action)
        self.button_adaptiveThreshold.clicked.connect(self.adaptiveThreshold_add_action)
        self.button_filter2D.clicked.connect(self.filter2D_add_action)
        self.button_bilateralFilter.clicked.connect(self.bilateralFilter_add_action)
        self.button_createCLAHE.clicked.connect(self.createCLAHE_add_action)
        self.button_morphologyEx.clicked.connect(self.morphologyEx_add_action)
        self.button_erode.clicked.connect(self.erode_add_action)
        self.button_dilate.clicked.connect(self.dilate_add_action)
    
    
    def chooseFile(self):        
        file_path = QtWidgets.QFileDialog.getOpenFileName()[0]
        _file_path = file_path.split('/')
        self.image = cv2.imread(file_path)
        
        if self.image_previous_name != _file_path[-1]:
            self.image_previous_name = _file_path[-1]
            if hasattr(self, "ui_ImageWindow"):
                self.ui_ImageWindow.close()
                self.clickImageLabel()
        
        self.h, self.w, self.ch = self.image.shape
        self.bytes_per_line = self.ch * self.w
        qImage = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], 
                            self.bytes_per_line, QtGui.QImage.Format.Format_BGR888) 
        pixmapImage = QtGui.QPixmap.fromImage(qImage)
        pixmapImage = pixmapImage.scaled(self.Image_label.width(), self.Image_label.height(), aspectRatioMode = QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.Image_label.setPixmap(pixmapImage)
        if hasattr(self, "list_of_filters"):
            self.drawImage()
    
    
    def saveFiltering(self):
        
        file_path = QtWidgets.QFileDialog.getSaveFileName(directory="ATK_ingot\\Ingot_processing\\filters", filter="Python Files (*.py)")[0]        
        scriptText =   "import cv2\nimport numpy as np\nimport math\n\n\ndef filtering(image):\n\n"
        if hasattr(self, "list_of_filters"):
            for filter in self.list_of_filters:
                if "cv2.createCLAHE" in filter:
                    scriptText += f"\tclahe = {filter}\n"
                    scriptText += f"\timage = clahe.apply(image)\n" 
                else:
                    scriptText += f"\timage = {filter}\n"
        scriptText += "\n\treturn image"
                
        with open(file_path, 'w') as f:
            f.write(scriptText)
    
    
    def loadFiltering(self):
        
        loadedListOfFilters = []
        file_path = QtWidgets.QFileDialog.getOpenFileName(directory="ATK_ingot\\Ingot_processing\\filters", filter="Python Files (*.py)")[0]
        with open(file_path, 'r') as f:
            scriptText = f.readlines()[7:-2]
        for item in scriptText:
            if "clahe.apply" in item:
                continue
            if "clahe = " in item or "image = " in item:
                loadedListOfFilters.append(item.split(" = ")[1].strip())
        
        if hasattr(self, "list_of_filters"):
            dialog = CustomDialog()
            if dialog.exec():
                temp_action_dict = self.action_dict.copy()
                for action in temp_action_dict.keys():
                    self.DELETEAction(f"{action}")
                
                self.list_of_filters = loadedListOfFilters.copy()
                self.loadFilteringAction()
            else:
                return 
        else:
            self.list_of_filters = loadedListOfFilters.copy()
            self.loadFilteringAction()
    
    
    def loadFilteringAction(self):
    
        filterActions = {
            "cvtColor": self.cvtColor_add_action,
            "GaussianBlur": self.gaussianBlur_add_action,
            "medianBlur": self.medianBlur_add_action,
            "threshold": self.threshold_add_action,
            "adaptiveThreshold": self.adaptiveThreshold_add_action,
            "filter2D": self.filter2D_add_action,
            "bilateralFilter": self.bilateralFilter_add_action,
            "createCLAHE": self.createCLAHE_add_action,
            "morphologyEx": self.morphologyEx_add_action,
            "erode": self.erode_add_action,
            "dilate": self.dilate_add_action
        }
        filterValues = {
            "cvtColor":             self.cvtColor_change_values,
            "GaussianBlur":         self.gaussianBlur_change_values,
            "medianBlur":           self.medianBlur_change_values,
            "threshold":            self.threshold_change_values,
            "adaptiveThreshold":    self.adaptiveThreshold_change_values,
            "filter2D":             self.filter2D_change_values,
            "bilateralFilter":      self.bilateralFilter_change_values,
            "createCLAHE":          self.createCLAHE_change_values,
            "morphologyEx":         self.morphologyEx_change_values,
            "erode":                self.erode_change_values,
            "dilate":               self.dilate_change_values
        }
        
        count = 0
        for filter in self.list_of_filters:
            temp = filter.split(".")[1].split("(")[0]
            print(temp)
            filterActions[temp].__call__() 
            filterValues[temp].__call__(temp, filter, count)
            count += 1 
        

    def cvtColor_add_action(self):
        
        index = len(self.action_dict)
        exec(f'self.cvtColor_{index}_groupBox = QtWidgets.QGroupBox(self.Elements_scrollAreaWidgetContents)')
        exec(f'self.cvtColor_{index}_layoutWidget = QtWidgets.QWidget(self.cvtColor_{index}_groupBox)')
        exec(f'self.cvtColor_{index}_formLayout_colorSpace = QtWidgets.QFormLayout(self.cvtColor_{index}_layoutWidget)')
        exec(f'self.cvtColor_{index}_label_colorSpace = QtWidgets.QLabel(self.cvtColor_{index}_layoutWidget)')
        exec(f'self.cvtColor_{index}_comboBox_colorSpace = QtWidgets.QComboBox(self.cvtColor_{index}_layoutWidget)')  
        self.action_dict.update({f'cvtColor_{index}': 
            [index,
            {
                'groupBox':                            eval(f"self.cvtColor_{index}_groupBox"),
                'layoutWidget':                        eval(f"self.cvtColor_{index}_layoutWidget"),
                'formLayout_colorSpace':               eval(f"self.cvtColor_{index}_formLayout_colorSpace"),
                'label_colorSpace':                    eval(f"self.cvtColor_{index}_label_colorSpace"),
                'comboBox_colorSpace':                 eval(f"self.cvtColor_{index}_comboBox_colorSpace")
            },
            {
                'colorSpace': None
            }]})
        self.action_dict[f"cvtColor_{index}"][1]["groupBox"].setMinimumSize(QtCore.QSize(260, 80))
        self.action_dict[f"cvtColor_{index}"][1]["groupBox"].setMaximumSize(QtCore.QSize(260, 80))
        self.action_dict[f"cvtColor_{index}"][1]["groupBox"].setCheckable(True)
        self.action_dict[f"cvtColor_{index}"][1]["groupBox"].setChecked(True)
        self.action_dict[f"cvtColor_{index}"][1]["groupBox"].setObjectName(f"cvtColor_{index}_groupBox")
        self.action_dict[f"cvtColor_{index}"][1]["layoutWidget"].setGeometry(QtCore.QRect(10, 20, 241, 40))
        self.action_dict[f"cvtColor_{index}"][1]["layoutWidget"].setObjectName(f"cvtColor_{index}_layoutWidget")
        self.action_dict[f"cvtColor_{index}"][1]["formLayout_colorSpace"].setContentsMargins(10, 10, 10, 10)
        self.action_dict[f"cvtColor_{index}"][1]["formLayout_colorSpace"].setObjectName(f"cvtColor_{index}_formLayout_colorSpace")
        self.action_dict[f"cvtColor_{index}"][1]["label_colorSpace"].setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f"cvtColor_{index}"][1]["label_colorSpace"].setObjectName(f"cvtColor_{index}_label_colorSpace")
        self.action_dict[f"cvtColor_{index}"][1]["formLayout_colorSpace"].setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.action_dict[f"cvtColor_{index}"][1]["label_colorSpace"])
        self.action_dict[f"cvtColor_{index}"][1]["comboBox_colorSpace"].setEditable(False)
        self.action_dict[f"cvtColor_{index}"][1]["comboBox_colorSpace"].setMaxVisibleItems(4)
        self.action_dict[f"cvtColor_{index}"][1]["comboBox_colorSpace"].setMaxCount(4)
        self.action_dict[f"cvtColor_{index}"][1]["comboBox_colorSpace"].setObjectName(f"cvtColor_{index}_comboBox_colorSpace")
        self.action_dict[f"cvtColor_{index}"][1]["comboBox_colorSpace"].addItem("BGR2GRAY")
        self.action_dict[f"cvtColor_{index}"][1]["comboBox_colorSpace"].addItem("RGB2GRAY")
        self.action_dict[f"cvtColor_{index}"][1]["comboBox_colorSpace"].addItem("GRAY2BGR")
        self.action_dict[f"cvtColor_{index}"][1]["comboBox_colorSpace"].addItem("GRAY2RGB")
        self.action_dict[f"cvtColor_{index}"][1]["groupBox"].setTitle(f"cvtColor: {index} - On")
        self.action_dict[f"cvtColor_{index}"][1]["label_colorSpace"].setText("Color Space")
        self.action_dict[f"cvtColor_{index}"][1]["formLayout_colorSpace"].setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.action_dict[f"cvtColor_{index}"][1]["comboBox_colorSpace"])
        self.Elements_scrollArea_verticalLayout_main.addWidget(self.action_dict[f"cvtColor_{index}"][1]["groupBox"])
        self.add_action(index, 'cvtColor')
        self.action_dict[f"cvtColor_{index}"].append(f'cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)')
        
        self.action_dict[f"cvtColor_{index}"][1]["comboBox_colorSpace"].currentIndexChanged.connect(lambda: self.input_parameters(index))


    def gaussianBlur_add_action(self):        
        index = len(self.action_dict)
        exec(f'self.GaussianBlur_{index}_groupBox = QtWidgets.QGroupBox(self.Elements_scrollAreaWidgetContents)')
        exec(f"self.GaussianBlur_{index}_layoutWidget = QtWidgets.QWidget(self.GaussianBlur_{index}_groupBox)")      
        exec(f"self.GaussianBlur_{index}_verticalLayout_all = QtWidgets.QVBoxLayout(self.GaussianBlur_{index}_layoutWidget)")
        exec(f"self.GaussianBlur_{index}_horizontalLayout_size = QtWidgets.QHBoxLayout()")
        exec(f"self.GaussianBlur_{index}_label_size = QtWidgets.QLabel(self.GaussianBlur_{index}_layoutWidget)")
        exec(f"self.GaussianBlur_{index}_horizontalLayout_sizeBoth = QtWidgets.QHBoxLayout()")
        exec(f"self.GaussianBlur_{index}_lineEdit_size_0 = QtWidgets.QLineEdit(self.GaussianBlur_{index}_layoutWidget)")
        exec(f"self.GaussianBlur_{index}_lineEdit_size_1 = QtWidgets.QLineEdit(self.GaussianBlur_{index}_layoutWidget)")
        exec(f"self.GaussianBlur_{index}_formLayout_borderType = QtWidgets.QFormLayout()")
        exec(f"self.GaussianBlur_{index}_label_borderType = QtWidgets.QLabel(self.GaussianBlur_{index}_layoutWidget)")
        exec(f"self.GaussianBlur_{index}_comboBox_borderType = QtWidgets.QComboBox(self.GaussianBlur_{index}_layoutWidget)")
  
        self.action_dict.update({f'GaussianBlur_{index}': 
            [index,
            {
                'groupBox':                            eval(f"self.GaussianBlur_{index}_groupBox"),
                'layoutWidget':                        eval(f"self.GaussianBlur_{index}_layoutWidget"),
                'verticalLayout_all':                  eval(f"self.GaussianBlur_{index}_verticalLayout_all"),
                'horizontalLayout_size':               eval(f"self.GaussianBlur_{index}_horizontalLayout_size"),
                'label_size':                          eval(f"self.GaussianBlur_{index}_label_size"),
                'horizontalLayout_sizeBoth':           eval(f"self.GaussianBlur_{index}_horizontalLayout_sizeBoth"),
                'lineEdit_size_0':                     eval(f"self.GaussianBlur_{index}_lineEdit_size_0"),
                'lineEdit_size_1':                     eval(f"self.GaussianBlur_{index}_lineEdit_size_1"),
                'formLayout_borderType':               eval(f"self.GaussianBlur_{index}_formLayout_borderType"),
                'label_borderType':                    eval(f"self.GaussianBlur_{index}_label_borderType"),
                'comboBox_borderType':                 eval(f"self.GaussianBlur_{index}_comboBox_borderType")
            },
            {
                'size_0': None,
                'size_1': None,
                'borderType': None
            }]})
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.action_dict[f'GaussianBlur_{index}'][1]['groupBox'].sizePolicy().hasHeightForWidth())
        self.action_dict[f'GaussianBlur_{index}'][1]['groupBox'].setSizePolicy(sizePolicy)
        self.action_dict[f'GaussianBlur_{index}'][1]['groupBox'].setMinimumSize(QtCore.QSize(260, 120))
        self.action_dict[f'GaussianBlur_{index}'][1]['groupBox'].setMaximumSize(QtCore.QSize(260, 120))
        self.action_dict[f'GaussianBlur_{index}'][1]['groupBox'].setCheckable(True)
        self.action_dict[f'GaussianBlur_{index}'][1]['groupBox'].setObjectName(f"GaussianBlur_{index}_groupBox")
        self.action_dict[f'GaussianBlur_{index}'][1]['layoutWidget'].setGeometry(QtCore.QRect(10, 20, 241, 90))
        self.action_dict[f'GaussianBlur_{index}'][1]['layoutWidget'].setObjectName(f"GaussianBlur_{index}_layoutWidget")
        self.action_dict[f'GaussianBlur_{index}'][1]['verticalLayout_all'].setContentsMargins(0, 0, 0, 0)
        self.action_dict[f'GaussianBlur_{index}'][1]['verticalLayout_all'].setObjectName(f"GaussianBlur_{index}_verticalLayout_all")
        self.action_dict[f'GaussianBlur_{index}'][1]['horizontalLayout_size'].setContentsMargins(10, 10, 10, 10)
        self.action_dict[f'GaussianBlur_{index}'][1]['horizontalLayout_size'].setSpacing(2)
        self.action_dict[f'GaussianBlur_{index}'][1]['horizontalLayout_size'].setObjectName(f"GaussianBlur_{index}_horizontalLayout_size")
        self.action_dict[f'GaussianBlur_{index}'][1]['label_size'].setMinimumSize(QtCore.QSize(50, 0))
        self.action_dict[f'GaussianBlur_{index}'][1]['label_size'].setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f'GaussianBlur_{index}'][1]['label_size'].setObjectName(f"GaussianBlur_{index}_label_size")
        self.action_dict[f'GaussianBlur_{index}'][1]['horizontalLayout_size'].addWidget(self.action_dict[f'GaussianBlur_{index}'][1]['label_size'])
        self.action_dict[f'GaussianBlur_{index}'][1]['horizontalLayout_sizeBoth'] = QtWidgets.QHBoxLayout()
        self.action_dict[f'GaussianBlur_{index}'][1]['horizontalLayout_sizeBoth'].setObjectName(f"GaussianBlur_{index}_horizontalLayout_sizeBoth")
        self.action_dict[f'GaussianBlur_{index}'][1]['lineEdit_size_0'].setObjectName(f"GaussianBlur_{index}_lineEdit_size_0")
        self.action_dict[f'GaussianBlur_{index}'][1]['lineEdit_size_0'].setText('3')
        self.action_dict[f'GaussianBlur_{index}'][1]['horizontalLayout_sizeBoth'].addWidget(self.action_dict[f'GaussianBlur_{index}'][1]['lineEdit_size_0'])
        self.action_dict[f'GaussianBlur_{index}'][1]['lineEdit_size_1'].setObjectName(f"GaussianBlur_{index}_lineEdit_size_1")
        self.action_dict[f'GaussianBlur_{index}'][1]['lineEdit_size_1'].setText('3')
        self.action_dict[f'GaussianBlur_{index}'][1]['horizontalLayout_sizeBoth'].addWidget(self.action_dict[f'GaussianBlur_{index}'][1]['lineEdit_size_1'])
        self.action_dict[f'GaussianBlur_{index}'][1]['horizontalLayout_size'].addLayout(self.action_dict[f'GaussianBlur_{index}'][1]['horizontalLayout_sizeBoth'])
        self.action_dict[f'GaussianBlur_{index}'][1]['verticalLayout_all'].addLayout(self.action_dict[f'GaussianBlur_{index}'][1]['horizontalLayout_size'])
        self.action_dict[f'GaussianBlur_{index}'][1]['formLayout_borderType'].setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.action_dict[f'GaussianBlur_{index}'][1]['formLayout_borderType'].setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.action_dict[f'GaussianBlur_{index}'][1]['formLayout_borderType'].setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f'GaussianBlur_{index}'][1]['formLayout_borderType'].setFormAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f'GaussianBlur_{index}'][1]['formLayout_borderType'].setContentsMargins(10, 10, 10, 10)
        self.action_dict[f'GaussianBlur_{index}'][1]['formLayout_borderType'].setObjectName(f"GaussianBlur_{index}_formLayout_borderType")
        self.action_dict[f'GaussianBlur_{index}'][1]['label_borderType'].setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f'GaussianBlur_{index}'][1]['label_borderType'].setObjectName(f"GaussianBlur_{index}_label_borderType")
        self.action_dict[f'GaussianBlur_{index}'][1]['formLayout_borderType'].setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.action_dict[f'GaussianBlur_{index}'][1]['label_borderType'])
        self.action_dict[f'GaussianBlur_{index}'][1]['comboBox_borderType'].setEditable(False)
        self.action_dict[f'GaussianBlur_{index}'][1]['comboBox_borderType'].setMaxVisibleItems(10)
        self.action_dict[f'GaussianBlur_{index}'][1]['comboBox_borderType'].setMaxCount(7)
        self.action_dict[f'GaussianBlur_{index}'][1]['comboBox_borderType'].setObjectName(f"GaussianBlur_{index}_comboBox_borderType")
        self.action_dict[f'GaussianBlur_{index}'][1]['comboBox_borderType'].addItem("BORDER_DEFAULT")
        self.action_dict[f'GaussianBlur_{index}'][1]['comboBox_borderType'].addItem("BORDER_CONSTANT")
        self.action_dict[f'GaussianBlur_{index}'][1]['comboBox_borderType'].addItem("BORDER_REPLICATE")
        self.action_dict[f'GaussianBlur_{index}'][1]['comboBox_borderType'].addItem("BORDER_REFLECT")
        self.action_dict[f'GaussianBlur_{index}'][1]['comboBox_borderType'].addItem("BORDER_WRAP")
        self.action_dict[f'GaussianBlur_{index}'][1]['comboBox_borderType'].addItem("BORDER_TRANSPARENT")
        self.action_dict[f'GaussianBlur_{index}'][1]['comboBox_borderType'].addItem("BORDER_ISOLATED")
        self.action_dict[f'GaussianBlur_{index}'][1]['formLayout_borderType'].setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.action_dict[f'GaussianBlur_{index}'][1]['comboBox_borderType'])
        self.action_dict[f'GaussianBlur_{index}'][1]['verticalLayout_all'].addLayout(self.action_dict[f'GaussianBlur_{index}'][1]['formLayout_borderType'])
        self.action_dict[f'GaussianBlur_{index}'][1]['groupBox'].setTitle(f"GaussianBlur: {index} - On")
        self.action_dict[f'GaussianBlur_{index}'][1]['label_size'].setText("Size")
        self.action_dict[f'GaussianBlur_{index}'][1]['label_borderType'].setText("Border Type")
        self.Elements_scrollArea_verticalLayout_main.addWidget(self.action_dict[f'GaussianBlur_{index}'][1]['groupBox'])
        
        self.add_action(index, 'GaussianBlur')
        self.action_dict[f"GaussianBlur_{index}"].append(f'cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)')
        
        # self.GaussianBlur_test_lineEdit_size_0 = QtWidgets.QLineEdit(eval(f"self.GaussianBlur_{index}_layoutWidget"))
        # self.GaussianBlur_test_lineEdit_size_0.te
        self.action_dict[f'GaussianBlur_{index}'][1]['comboBox_borderType'].currentIndexChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f'GaussianBlur_{index}'][1]['lineEdit_size_0'].textChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f'GaussianBlur_{index}'][1]['lineEdit_size_1'].textChanged.connect(lambda: self.input_parameters(index))


    def medianBlur_add_action(self):        
        index = len(self.action_dict)
        exec(f"self.MedianBlur_{index}_groupBox = QtWidgets.QGroupBox(self.Elements_scrollAreaWidgetContents)")
        exec(f"self.MedianBlur_{index}_layoutWidget = QtWidgets.QWidget(self.MedianBlur_{index}_groupBox)")
        exec(f"self.MedianBlur_{index}_horizontalLayout_all = QtWidgets.QHBoxLayout(self.MedianBlur_{index}_layoutWidget)")
        exec(f"self.MedianBlur_{index}_label_size = QtWidgets.QLabel(self.MedianBlur_{index}_layoutWidget)")
        exec(f"self.MedianBlur_{index}_horizontalLayout_sizeBoth = QtWidgets.QHBoxLayout()")
        exec(f"self.MedianBlur_{index}_lineEdit_size_0 = QtWidgets.QLineEdit(self.MedianBlur_{index}_layoutWidget)")
        exec(f"self.MedianBlur_{index}_lineEdit_size_1 = QtWidgets.QLineEdit(self.MedianBlur_{index}_layoutWidget)")
        
        self.action_dict.update({f'MedianBlur_{index}': 
            [index,
            {
                'groupBox':                            eval(f"self.MedianBlur_{index}_groupBox"),
                'layoutWidget':                        eval(f"self.MedianBlur_{index}_layoutWidget"),
                'horizontalLayout_all':                eval(f"self.MedianBlur_{index}_horizontalLayout_all"),
                'label_size':                          eval(f"self.MedianBlur_{index}_label_size"),
                'horizontalLayout_sizeBoth':           eval(f"self.MedianBlur_{index}_horizontalLayout_sizeBoth"),
                'lineEdit_size_0':                     eval(f"self.MedianBlur_{index}_lineEdit_size_0"),
                'lineEdit_size_1':                     eval(f"self.MedianBlur_{index}_lineEdit_size_1")
            },
            {
                'size_0': None,
                'size_1': None
            }            
            ]})
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.action_dict[f'MedianBlur_{index}'][1]['groupBox'].sizePolicy().hasHeightForWidth())        
        self.action_dict[f'MedianBlur_{index}'][1]['groupBox'].setSizePolicy(sizePolicy)
        self.action_dict[f'MedianBlur_{index}'][1]['groupBox'].setMinimumSize(QtCore.QSize(260, 80))
        self.action_dict[f'MedianBlur_{index}'][1]['groupBox'].setMaximumSize(QtCore.QSize(260, 80))
        self.action_dict[f'MedianBlur_{index}'][1]['groupBox'].setCheckable(True)
        self.action_dict[f'MedianBlur_{index}'][1]['groupBox'].setObjectName(f"MedianBlur_{index}_groupBox")
        self.action_dict[f'MedianBlur_{index}'][1]['layoutWidget'].setGeometry(QtCore.QRect(10, 20, 239, 42))
        self.action_dict[f'MedianBlur_{index}'][1]['layoutWidget'].setObjectName(f"MedianBlur_{index}_layoutWidget")
        self.action_dict[f'MedianBlur_{index}'][1]['horizontalLayout_all'].setContentsMargins(10, 10, 10, 10)
        self.action_dict[f'MedianBlur_{index}'][1]['horizontalLayout_all'].setSpacing(2)
        self.action_dict[f'MedianBlur_{index}'][1]['horizontalLayout_all'].setObjectName(f"MedianBlur_{index}_horizontalLayout_all")
        self.action_dict[f'MedianBlur_{index}'][1]['label_size'].setMinimumSize(QtCore.QSize(50, 0))
        self.action_dict[f'MedianBlur_{index}'][1]['label_size'].setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f'MedianBlur_{index}'][1]['label_size'].setObjectName(f"MedianBlur_{index}_label_size")
        self.action_dict[f'MedianBlur_{index}'][1]['horizontalLayout_all'].addWidget(self.action_dict[f'MedianBlur_{index}'][1]['label_size'])
        self.action_dict[f'MedianBlur_{index}'][1]['horizontalLayout_sizeBoth'].setObjectName(f"MedianBlur_ID_horizontalLayout_sizeBoth")
        self.action_dict[f'MedianBlur_{index}'][1]['lineEdit_size_0'].setObjectName(f"MedianBlur_{index}_lineEdit_size_0")
        self.action_dict[f'MedianBlur_{index}'][1]['lineEdit_size_0'].setText('3')
        self.action_dict[f'MedianBlur_{index}'][1]['horizontalLayout_sizeBoth'].addWidget(self.action_dict[f'MedianBlur_{index}'][1]['lineEdit_size_0'])
        self.action_dict[f'MedianBlur_{index}'][1]['lineEdit_size_1'].setObjectName(f"MedianBlur_{index}_lineEdit_size_1")
        self.action_dict[f'MedianBlur_{index}'][1]['lineEdit_size_1'].setText('3')
        self.action_dict[f'MedianBlur_{index}'][1]['horizontalLayout_sizeBoth'].addWidget(self.action_dict[f'MedianBlur_{index}'][1]['lineEdit_size_1'])
        self.action_dict[f'MedianBlur_{index}'][1]['horizontalLayout_all'].addLayout(self.action_dict[f'MedianBlur_{index}'][1]['horizontalLayout_sizeBoth'])
        self.action_dict[f'MedianBlur_{index}'][1]['groupBox'].setTitle(f"MedianBlur: {index} - On")
        self.action_dict[f'MedianBlur_{index}'][1]['label_size'].setText("Size")
        self.Elements_scrollArea_verticalLayout_main.addWidget(self.action_dict[f'MedianBlur_{index}'][1]['groupBox'])
        
        self.add_action(index, 'MedianBlur')
        self.action_dict[f"MedianBlur_{index}"].append(f'cv2.medianBlur(image, (3, 3))')
        
        self.action_dict[f'MedianBlur_{index}'][1]['lineEdit_size_0'].textChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f'MedianBlur_{index}'][1]['lineEdit_size_1'].textChanged.connect(lambda: self.input_parameters(index))
 

    def threshold_add_action(self):
        index = len(self.action_dict)
        exec(f'self.Threshold_{index}_groupBox = QtWidgets.QGroupBox(self.Elements_scrollAreaWidgetContents)')
        exec(f'self.Threshold_{index}_layoutWidget = QtWidgets.QWidget(self.Threshold_{index}_groupBox)')
        exec(f'self.Threshold_{index}_verticalLayout_all = QtWidgets.QVBoxLayout(self.Threshold_{index}_layoutWidget)')
        exec(f'self.Threshold_{index}_verticalLayout_minValue = QtWidgets.QVBoxLayout()')
        exec(f'self.Threshold_{index}_label_minValue = QtWidgets.QLabel(self.Threshold_{index}_layoutWidget)')
        exec(f'self.Threshold_{index}_horizontalLayout_minValue = QtWidgets.QHBoxLayout()')
        exec(f'self.Threshold_{index}_horizontalSlider_minValue = QtWidgets.QSlider(self.Threshold_{index}_layoutWidget)')
        exec(f'self.Threshold_{index}_spinBox_minValue = QtWidgets.QSpinBox(self.Threshold_{index}_layoutWidget)')
        exec(f'self.Threshold_{index}_verticalLayout_maxValue = QtWidgets.QVBoxLayout()')
        exec(f'self.Threshold_{index}_label_maxValue = QtWidgets.QLabel(self.Threshold_{index}_layoutWidget)')
        exec(f'self.Threshold_{index}_horizontalLayout_maxValue = QtWidgets.QHBoxLayout()')
        exec(f'self.Threshold_{index}_horizontalSlider_maxValue = QtWidgets.QSlider(self.Threshold_{index}_layoutWidget)')
        exec(f'self.Threshold_{index}_spinBox_maxValue = QtWidgets.QSpinBox(self.Threshold_{index}_layoutWidget)')
        exec(f'self.Threshold_{index}_formLayout_thresholdType = QtWidgets.QFormLayout()')
        exec(f'self.Threshold_{index}_label_thresholdType = QtWidgets.QLabel(self.Threshold_{index}_layoutWidget)')
        exec(f'self.Threshold_{index}_comboBox_thresholdType = QtWidgets.QComboBox(self.Threshold_{index}_layoutWidget)')

        self.action_dict.update({f'Threshold_{index}': 
            [index,
            {
                'groupBox': eval(f"self.Threshold_{index}_groupBox"),
                'layoutWidget': eval(f"self.Threshold_{index}_layoutWidget"),
                'verticalLayout_all': eval(f"self.Threshold_{index}_verticalLayout_all"),
                'verticalLayout_minValue': eval(f"self.Threshold_{index}_verticalLayout_minValue"),
                'label_minValue': eval(f"self.Threshold_{index}_label_minValue"),
                'horizontalLayout_minValue': eval(f"self.Threshold_{index}_horizontalLayout_minValue"),
                'horizontalSlider_minValue': eval(f"self.Threshold_{index}_horizontalSlider_minValue"),
                'spinBox_minValue': eval(f"self.Threshold_{index}_spinBox_minValue"),
                'verticalLayout_maxValue': eval(f"self.Threshold_{index}_verticalLayout_maxValue"),
                'label_maxValue': eval(f"self.Threshold_{index}_label_maxValue"),
                'horizontalLayout_maxValue': eval(f"self.Threshold_{index}_horizontalLayout_maxValue"),
                'horizontalSlider_maxValue': eval(f"self.Threshold_{index}_horizontalSlider_maxValue"),
                'spinBox_maxValue': eval(f"self.Threshold_{index}_spinBox_maxValue"),
                'formLayout_thresholdType': eval(f"self.Threshold_{index}_formLayout_thresholdType"),
                'label_thresholdType': eval(f"self.Threshold_{index}_label_thresholdType"),
                'comboBox_thresholdType': eval(f"self.Threshold_{index}_comboBox_thresholdType")
            },
            {
                'minValue': None,
                'maxValue': None,
                'thresholdType': None
            }
            ]})

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.action_dict[f"Threshold_{index}"][1]["groupBox"].sizePolicy().hasHeightForWidth())
        self.action_dict[f"Threshold_{index}"][1]["groupBox"].setSizePolicy(sizePolicy)
        self.action_dict[f"Threshold_{index}"][1]["groupBox"].setMinimumSize(QtCore.QSize(260, 200))
        self.action_dict[f"Threshold_{index}"][1]["groupBox"].setMaximumSize(QtCore.QSize(260, 200))
        self.action_dict[f"Threshold_{index}"][1]["groupBox"].setCheckable(True)
        self.action_dict[f"Threshold_{index}"][1]["groupBox"].setObjectName(f"Threshold_{index}_groupBox")
        self.action_dict[f"Threshold_{index}"][1]["layoutWidget"].setGeometry(QtCore.QRect(10, 20, 242, 163))
        self.action_dict[f"Threshold_{index}"][1]["layoutWidget"].setObjectName("layoutWidget3")
        self.action_dict[f"Threshold_{index}"][1]["verticalLayout_all"].setContentsMargins(0, 0, 0, 0)
        self.action_dict[f"Threshold_{index}"][1]["verticalLayout_all"].setObjectName(f"Threshold_{index}_verticalLayout_all")
        self.action_dict[f"Threshold_{index}"][1]["verticalLayout_minValue"].setObjectName(f"Threshold_{index}_verticalLayout_minValue")
        font = QtGui.QFont()
        font.setPointSize(8)
        self.action_dict[f"Threshold_{index}"][1]["label_minValue"].setFont(font)
        self.action_dict[f"Threshold_{index}"][1]["label_minValue"].setObjectName(f"Threshold_{index}_label_minValue")
        self.action_dict[f"Threshold_{index}"][1]["verticalLayout_minValue"].addWidget(self.action_dict[f"Threshold_{index}"][1]["label_minValue"])
        self.action_dict[f"Threshold_{index}"][1]["horizontalLayout_minValue"].setObjectName(f"Threshold_{index}_horizontalLayout_minValue")
        self.action_dict[f"Threshold_{index}"][1]["horizontalSlider_minValue"].setMinimumSize(QtCore.QSize(180, 22))
        self.action_dict[f"Threshold_{index}"][1]["horizontalSlider_minValue"].setMaximum(255)
        self.action_dict[f"Threshold_{index}"][1]["horizontalSlider_minValue"].setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.action_dict[f"Threshold_{index}"][1]["horizontalSlider_minValue"].setObjectName(f"Threshold_{index}_horizontalSlider_minValue")
        self.action_dict[f"Threshold_{index}"][1]["horizontalLayout_minValue"].addWidget(self.action_dict[f"Threshold_{index}"][1]["horizontalSlider_minValue"])
        self.action_dict[f"Threshold_{index}"][1]["spinBox_minValue"].setMinimumSize(QtCore.QSize(50, 22))
        self.action_dict[f"Threshold_{index}"][1]["spinBox_minValue"].setMaximum(255)
        self.action_dict[f"Threshold_{index}"][1]["spinBox_minValue"].setObjectName(f"Threshold_{index}_spinBox_minValue")
        self.action_dict[f"Threshold_{index}"][1]["horizontalLayout_minValue"].addWidget(self.action_dict[f"Threshold_{index}"][1]["spinBox_minValue"])
        self.action_dict[f"Threshold_{index}"][1]["verticalLayout_minValue"].addLayout(self.action_dict[f"Threshold_{index}"][1]["horizontalLayout_minValue"])
        self.action_dict[f"Threshold_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"Threshold_{index}"][1]["verticalLayout_minValue"])
        self.action_dict[f"Threshold_{index}"][1]["verticalLayout_maxValue"].setObjectName(f"Threshold_{index}_verticalLayout_maxValue")
        font = QtGui.QFont()
        font.setPointSize(8)
        self.action_dict[f"Threshold_{index}"][1]["label_maxValue"].setFont(font)
        self.action_dict[f"Threshold_{index}"][1]["label_maxValue"].setObjectName(f"Threshold_{index}_label_maxValue")
        self.action_dict[f"Threshold_{index}"][1]["verticalLayout_maxValue"].addWidget(self.action_dict[f"Threshold_{index}"][1]["label_maxValue"])
        self.action_dict[f"Threshold_{index}"][1]["horizontalLayout_maxValue"].setObjectName(f"Threshold_{index}_horizontalLayout_maxValue")
        self.action_dict[f"Threshold_{index}"][1]["horizontalSlider_maxValue"].setMinimumSize(QtCore.QSize(180, 22))
        self.action_dict[f"Threshold_{index}"][1]["horizontalSlider_maxValue"].setMaximum(255)
        self.action_dict[f"Threshold_{index}"][1]["horizontalSlider_maxValue"].setProperty("value", 255)
        self.action_dict[f"Threshold_{index}"][1]["horizontalSlider_maxValue"].setSliderPosition(255)
        self.action_dict[f"Threshold_{index}"][1]["horizontalSlider_maxValue"].setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.action_dict[f"Threshold_{index}"][1]["horizontalSlider_maxValue"].setObjectName(f"Threshold_{index}_horizontalSlider_maxValue")
        self.action_dict[f"Threshold_{index}"][1]["horizontalLayout_maxValue"].addWidget(self.action_dict[f"Threshold_{index}"][1]["horizontalSlider_maxValue"])
        self.action_dict[f"Threshold_{index}"][1]["spinBox_maxValue"].setMinimumSize(QtCore.QSize(50, 22))
        self.action_dict[f"Threshold_{index}"][1]["spinBox_maxValue"].setMaximum(255)
        self.action_dict[f"Threshold_{index}"][1]["spinBox_maxValue"].setProperty("value", 255)
        self.action_dict[f"Threshold_{index}"][1]["spinBox_maxValue"].setObjectName(f"Threshold_{index}_spinBox_maxValue")
        self.action_dict[f"Threshold_{index}"][1]["horizontalLayout_maxValue"].addWidget(self.action_dict[f"Threshold_{index}"][1]["spinBox_maxValue"])
        self.action_dict[f"Threshold_{index}"][1]["verticalLayout_maxValue"].addLayout(self.action_dict[f"Threshold_{index}"][1]["horizontalLayout_maxValue"])
        self.action_dict[f"Threshold_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"Threshold_{index}"][1]["verticalLayout_maxValue"])
        self.action_dict[f"Threshold_{index}"][1]["formLayout_thresholdType"].setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.action_dict[f"Threshold_{index}"][1]["formLayout_thresholdType"].setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.action_dict[f"Threshold_{index}"][1]["formLayout_thresholdType"].setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f"Threshold_{index}"][1]["formLayout_thresholdType"].setFormAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f"Threshold_{index}"][1]["formLayout_thresholdType"].setContentsMargins(10, 10, 10, 10)
        self.action_dict[f"Threshold_{index}"][1]["formLayout_thresholdType"].setObjectName(f"Threshold_{index}_formLayout_thresholdType")
        self.action_dict[f"Threshold_{index}"][1]["label_thresholdType"].setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f"Threshold_{index}"][1]["label_thresholdType"].setObjectName(f"Threshold_{index}_label_thresholdType")
        self.action_dict[f"Threshold_{index}"][1]["formLayout_thresholdType"].setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.action_dict[f"Threshold_{index}"][1]["label_thresholdType"])
        self.action_dict[f"Threshold_{index}"][1]["comboBox_thresholdType"].setEditable(False)
        self.action_dict[f"Threshold_{index}"][1]["comboBox_thresholdType"].setMaxVisibleItems(10)
        self.action_dict[f"Threshold_{index}"][1]["comboBox_thresholdType"].setMaxCount(7)
        self.action_dict[f"Threshold_{index}"][1]["comboBox_thresholdType"].setObjectName(f"Threshold_{index}_comboBox_thresholdType")
        self.action_dict[f"Threshold_{index}"][1]["comboBox_thresholdType"].addItem("THRESH_BINARY")
        self.action_dict[f"Threshold_{index}"][1]["comboBox_thresholdType"].addItem("THRESH_BINARY_INV")
        self.action_dict[f"Threshold_{index}"][1]["comboBox_thresholdType"].addItem("THRESH_TRUNC")
        self.action_dict[f"Threshold_{index}"][1]["comboBox_thresholdType"].addItem("THRESH_TOZERO")
        self.action_dict[f"Threshold_{index}"][1]["comboBox_thresholdType"].addItem("THRESH_TOZERO_INV")
        self.action_dict[f"Threshold_{index}"][1]["formLayout_thresholdType"].setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.action_dict[f"Threshold_{index}"][1]["comboBox_thresholdType"])
        self.action_dict[f"Threshold_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"Threshold_{index}"][1]["formLayout_thresholdType"])       
        self.action_dict[f"Threshold_{index}"][1]["groupBox"].setTitle(f"Threshold: {index} - On")
        self.action_dict[f"Threshold_{index}"][1]["label_minValue"].setText("Min Value")
        self.action_dict[f"Threshold_{index}"][1]["label_maxValue"].setText("Max Value")
        self.action_dict[f"Threshold_{index}"][1]["label_thresholdType"].setText("Threshold Type")
                
        self.action_dict[f"Threshold_{index}"][1]["horizontalSlider_minValue"].valueChanged.connect(lambda: self.action_dict[f"Threshold_{index}"][1]["spinBox_minValue"].
                                                                                                    setValue(self.action_dict[f"Threshold_{index}"][1]["horizontalSlider_minValue"].value()))
        self.action_dict[f"Threshold_{index}"][1]["horizontalSlider_minValue"].valueChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f"Threshold_{index}"][1]["spinBox_minValue"].valueChanged.connect(lambda: self.action_dict[f"Threshold_{index}"][1]["horizontalSlider_minValue"].
                                                                                                    setValue(self.action_dict[f"Threshold_{index}"][1]["spinBox_minValue"].value()))
        self.action_dict[f"Threshold_{index}"][1]["spinBox_minValue"].valueChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f"Threshold_{index}"][1]["horizontalSlider_maxValue"].valueChanged.connect(lambda: self.action_dict[f"Threshold_{index}"][1]["spinBox_maxValue"].
                                                                                                    setValue(self.action_dict[f"Threshold_{index}"][1]["horizontalSlider_maxValue"].value()))
        self.action_dict[f"Threshold_{index}"][1]["horizontalSlider_maxValue"].valueChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f"Threshold_{index}"][1]["spinBox_maxValue"].valueChanged.connect(lambda: self.action_dict[f"Threshold_{index}"][1]["horizontalSlider_maxValue"].
                                                                                                    setValue(self.action_dict[f"Threshold_{index}"][1]["spinBox_maxValue"].value()))
        self.action_dict[f"Threshold_{index}"][1]["spinBox_maxValue"].valueChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f"Threshold_{index}"][1]["comboBox_thresholdType"].currentIndexChanged.connect(lambda: self.input_parameters(index))
        
        self.add_action(index, 'Threshold')
        self.action_dict[f"Threshold_{index}"].append(f'cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]')
        
        self.Elements_scrollArea_verticalLayout_main.addWidget(self.action_dict[f"Threshold_{index}"][1]["groupBox"])


    def adaptiveThreshold_add_action(self):
        index = len(self.action_dict)
        exec(f"self.AdaptiveThreshold_{index}_groupBox = QtWidgets.QGroupBox(self.Elements_scrollAreaWidgetContents)")
        exec(f"self.AdaptiveThreshold_{index}_layoutWidget = QtWidgets.QWidget(self.AdaptiveThreshold_{index}_groupBox)")
        exec(f"self.AdaptiveThreshold_{index}_verticalLayout_all = QtWidgets.QVBoxLayout(self.AdaptiveThreshold_{index}_layoutWidget)")
        exec(f"self.AdaptiveThreshold_{index}_verticalLayout_maxValue = QtWidgets.QVBoxLayout()")
        exec(f"self.AdaptiveThreshold_{index}_label_maxValue = QtWidgets.QLabel(self.AdaptiveThreshold_{index}_layoutWidget)")
        exec(f"self.AdaptiveThreshold_{index}_horizontalLayout_maxValue = QtWidgets.QHBoxLayout()")
        exec(f"self.AdaptiveThreshold_{index}_horizontalSlider_maxValue = QtWidgets.QSlider(self.AdaptiveThreshold_{index}_layoutWidget)")
        exec(f"self.AdaptiveThreshold_{index}_spinBox_maxValue = QtWidgets.QSpinBox(self.AdaptiveThreshold_{index}_layoutWidget)")
        exec(f"self.AdaptiveThreshold_{index}_formLayout_adaptiveThresholdType = QtWidgets.QFormLayout()")
        exec(f"self.AdaptiveThreshold_{index}_comboBox_adaptiveThresholdType = QtWidgets.QComboBox(self.AdaptiveThreshold_{index}_layoutWidget)")
        exec(f"self.AdaptiveThreshold_{index}_label_adaptiveThresholdType = QtWidgets.QLabel(self.AdaptiveThreshold_{index}_layoutWidget)")
        exec(f"self.AdaptiveThreshold_{index}_formLayout_thresholdType = QtWidgets.QFormLayout()")
        exec(f"self.AdaptiveThreshold_{index}_label_thresholdType = QtWidgets.QLabel(self.AdaptiveThreshold_{index}_layoutWidget)")
        exec(f"self.AdaptiveThreshold_{index}_comboBox_thresholdType = QtWidgets.QComboBox(self.AdaptiveThreshold_{index}_layoutWidget)")
        exec(f"self.AdaptiveThreshold_{index}_verticalLayout_blockSize = QtWidgets.QVBoxLayout()")
        exec(f"self.AdaptiveThreshold_{index}_label_blockSize = QtWidgets.QLabel(self.AdaptiveThreshold_{index}_layoutWidget)")
        exec(f"self.AdaptiveThreshold_{index}_horizontalLayout_blockSize = QtWidgets.QHBoxLayout()")
        exec(f"self.AdaptiveThreshold_{index}_horizontalSlider_blockSize = QtWidgets.QSlider(self.AdaptiveThreshold_{index}_layoutWidget)")
        exec(f"self.AdaptiveThreshold_{index}_spinBox_blockSize = QtWidgets.QSpinBox(self.AdaptiveThreshold_{index}_layoutWidget)")
        exec(f"self.AdaptiveThreshold_{index}_verticalLayout_cConstantValue = QtWidgets.QVBoxLayout()")
        exec(f"self.AdaptiveThreshold_{index}_label_cConstantValue = QtWidgets.QLabel(self.AdaptiveThreshold_{index}_layoutWidget)")
        exec(f"self.AdaptiveThreshold_{index}_horizontalLayout_cConstantValue = QtWidgets.QHBoxLayout()")
        exec(f"self.AdaptiveThreshold_{index}_horizontalSlider_cConstantValue = QtWidgets.QSlider(self.AdaptiveThreshold_{index}_layoutWidget)")
        exec(f"self.AdaptiveThreshold_{index}_spinBox_cConstantValue = QtWidgets.QSpinBox(self.AdaptiveThreshold_{index}_layoutWidget)")

        self.action_dict.update({f'AdaptiveThreshold_{index}': 
            [index,
            {   
                'groupBox': eval(f"self.AdaptiveThreshold_{index}_groupBox"),
                'layoutWidget': eval(f"self.AdaptiveThreshold_{index}_layoutWidget"),
                'verticalLayout_all': eval(f"self.AdaptiveThreshold_{index}_verticalLayout_all"),
                'verticalLayout_maxValue': eval(f"self.AdaptiveThreshold_{index}_verticalLayout_maxValue"),
                'label_maxValue': eval(f"self.AdaptiveThreshold_{index}_label_maxValue"),
                'horizontalLayout_maxValue': eval(f"self.AdaptiveThreshold_{index}_horizontalLayout_maxValue"),
                'horizontalSlider_maxValue': eval(f"self.AdaptiveThreshold_{index}_horizontalSlider_maxValue"),
                'spinBox_maxValue': eval(f"self.AdaptiveThreshold_{index}_spinBox_maxValue"),
                'formLayout_adaptiveThresholdType': eval(f"self.AdaptiveThreshold_{index}_formLayout_adaptiveThresholdType"),
                'comboBox_adaptiveThresholdType': eval(f"self.AdaptiveThreshold_{index}_comboBox_adaptiveThresholdType"),
                'label_adaptiveThresholdType': eval(f"self.AdaptiveThreshold_{index}_label_adaptiveThresholdType"),
                'formLayout_thresholdType': eval(f"self.AdaptiveThreshold_{index}_formLayout_thresholdType"),
                'label_thresholdType': eval(f"self.AdaptiveThreshold_{index}_label_thresholdType"),
                'comboBox_thresholdType': eval(f"self.AdaptiveThreshold_{index}_comboBox_thresholdType"),
                'verticalLayout_blockSize': eval(f"self.AdaptiveThreshold_{index}_verticalLayout_blockSize"),
                'label_blockSize': eval(f"self.AdaptiveThreshold_{index}_label_blockSize"),
                'horizontalLayout_blockSize': eval(f"self.AdaptiveThreshold_{index}_horizontalLayout_blockSize"),
                'horizontalSlider_blockSize': eval(f"self.AdaptiveThreshold_{index}_horizontalSlider_blockSize"),
                'spinBox_blockSize': eval(f"self.AdaptiveThreshold_{index}_spinBox_blockSize"),
                'verticalLayout_cConstantValue': eval(f"self.AdaptiveThreshold_{index}_verticalLayout_cConstantValue"),
                'label_cConstantValue': eval(f"self.AdaptiveThreshold_{index}_label_cConstantValue"),
                'horizontalLayout_cConstantValue': eval(f"self.AdaptiveThreshold_{index}_horizontalLayout_cConstantValue"),
                'horizontalSlider_cConstantValue': eval(f"self.AdaptiveThreshold_{index}_horizontalSlider_cConstantValue"),
                'spinBox_cConstantValue': eval(f"self.AdaptiveThreshold_{index}_spinBox_cConstantValue")
            },
            {
                'maxValue': None,
                'adaptiveThresholdType': None,
                'thresholdType': None,
                'blockSize': None,
                'cConstantValue': None
            }]})
        
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.action_dict[f"AdaptiveThreshold_{index}"][1]["groupBox"].sizePolicy().hasHeightForWidth())
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["groupBox"].setSizePolicy(sizePolicy)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["groupBox"].setMinimumSize(QtCore.QSize(260, 280))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["groupBox"].setMaximumSize(QtCore.QSize(260, 280))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["groupBox"].setCheckable(True)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["groupBox"].setObjectName(f"AdaptiveThreshold_{index}_groupBox")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["layoutWidget"].setGeometry(QtCore.QRect(10, 20, 244, 247))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["layoutWidget"].setObjectName("layoutWidget_5")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["verticalLayout_all"].setContentsMargins(0, 0, 0, 0)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["verticalLayout_all"].setObjectName(f"AdaptiveThreshold_{index}_verticalLayout_all")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["verticalLayout_maxValue"].setObjectName(f"AdaptiveThreshold_{index}_verticalLayout_maxValue")
        font = QtGui.QFont()
        font.setPointSize(8)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["label_maxValue"].setFont(font)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["label_maxValue"].setObjectName(f"AdaptiveThreshold_{index}_label_maxValue")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["verticalLayout_maxValue"].addWidget(self.action_dict[f"AdaptiveThreshold_{index}"][1]["label_maxValue"])
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalLayout_maxValue"].setObjectName(f"AdaptiveThreshold_{index}_horizontalLayout_maxValue")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_maxValue"].setMinimumSize(QtCore.QSize(180, 22))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_maxValue"].setMaximum(255)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_maxValue"].setProperty("value", 255)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_maxValue"].setSliderPosition(255)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_maxValue"].setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_maxValue"].setObjectName(f"AdaptiveThreshold_{index}_horizontalSlider_maxValue")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalLayout_maxValue"].addWidget(self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_maxValue"])
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_maxValue"].setMinimumSize(QtCore.QSize(50, 22))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_maxValue"].setMaximum(255)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_maxValue"].setProperty("value", 255)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_maxValue"].setObjectName(f"AdaptiveThreshold_{index}_spinBox_maxValue")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalLayout_maxValue"].addWidget(self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_maxValue"])
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["verticalLayout_maxValue"].addLayout(self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalLayout_maxValue"])
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"AdaptiveThreshold_{index}"][1]["verticalLayout_maxValue"])
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["formLayout_adaptiveThresholdType"].setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["formLayout_adaptiveThresholdType"].setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["formLayout_adaptiveThresholdType"].setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["formLayout_adaptiveThresholdType"].setFormAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["formLayout_adaptiveThresholdType"].setContentsMargins(10, 10, 10, 10)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["formLayout_adaptiveThresholdType"].setObjectName(f"AdaptiveThreshold_{index}_formLayout_adaptiveThresholdType")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["comboBox_adaptiveThresholdType"].setEditable(False)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["comboBox_adaptiveThresholdType"].setMaxVisibleItems(10)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["comboBox_adaptiveThresholdType"].setMaxCount(7)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["comboBox_adaptiveThresholdType"].setObjectName(f"AdaptiveThreshold_{index}_comboBox_adaptiveThresholdType")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["comboBox_adaptiveThresholdType"].addItem("ADAPTIVE_THRESH_GAUSSIAN_C")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["comboBox_adaptiveThresholdType"].addItem("ADAPTIVE_THRESH_MEAN_C")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["formLayout_adaptiveThresholdType"].setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.action_dict[f"AdaptiveThreshold_{index}"][1]["comboBox_adaptiveThresholdType"])
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["label_adaptiveThresholdType"].setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["label_adaptiveThresholdType"].setObjectName(f"AdaptiveThreshold_{index}_label_adaptiveThresholdType")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["formLayout_adaptiveThresholdType"].setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.action_dict[f"AdaptiveThreshold_{index}"][1]["label_adaptiveThresholdType"])
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"AdaptiveThreshold_{index}"][1]["formLayout_adaptiveThresholdType"])
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["formLayout_thresholdType"].setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["formLayout_thresholdType"].setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["formLayout_thresholdType"].setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["formLayout_thresholdType"].setFormAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["formLayout_thresholdType"].setContentsMargins(10, 10, 10, 10)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["formLayout_thresholdType"].setObjectName(f"AdaptiveThreshold_{index}_formLayout_thresholdType")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["label_thresholdType"].setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["label_thresholdType"].setObjectName(f"AdaptiveThreshold_{index}_label_thresholdType")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["formLayout_thresholdType"].setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.action_dict[f"AdaptiveThreshold_{index}"][1]["label_thresholdType"])
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["comboBox_thresholdType"].setEditable(False)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["comboBox_thresholdType"].setMaxVisibleItems(10)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["comboBox_thresholdType"].setMaxCount(7)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["comboBox_thresholdType"].setObjectName(f"AdaptiveThreshold_{index}_comboBox_thresholdType")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["comboBox_thresholdType"].addItem("THRESH_BINARY")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["comboBox_thresholdType"].addItem("THRESH_BINARY_INV")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["comboBox_thresholdType"].addItem("THRESH_TRUNC")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["comboBox_thresholdType"].addItem("THRESH_TOZERO")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["comboBox_thresholdType"].addItem("THRESH_TOZERO_INV")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["formLayout_thresholdType"].setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.action_dict[f"AdaptiveThreshold_{index}"][1]["comboBox_thresholdType"])
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"AdaptiveThreshold_{index}"][1]["formLayout_thresholdType"])
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["verticalLayout_blockSize"].setObjectName(f"AdaptiveThreshold_{index}_verticalLayout_blockSize")
        font = QtGui.QFont()
        font.setPointSize(8)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["label_blockSize"].setFont(font)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["label_blockSize"].setObjectName(f"AdaptiveThreshold_{index}_label_blockSize")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["verticalLayout_blockSize"].addWidget(self.action_dict[f"AdaptiveThreshold_{index}"][1]["label_blockSize"])
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalLayout_blockSize"].setObjectName(f"AdaptiveThreshold_{index}_horizontalLayout_blockSize")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_blockSize"].setMinimumSize(QtCore.QSize(180, 22))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_blockSize"].setMinimum(3)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_blockSize"].setMaximum(199)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_blockSize"].setSingleStep(2)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_blockSize"].setProperty("value", 3)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_blockSize"].setSliderPosition(3)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_blockSize"].setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_blockSize"].setObjectName(f"AdaptiveThreshold_{index}_horizontalSlider_blockSize")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalLayout_blockSize"].addWidget(self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_blockSize"])
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_blockSize"].setMinimumSize(QtCore.QSize(50, 22))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_blockSize"].setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_blockSize"].setSpecialValueText("")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_blockSize"].setKeyboardTracking(True)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_blockSize"].setMinimum(3)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_blockSize"].setMaximum(199)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_blockSize"].setSingleStep(2)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_blockSize"].setProperty("value", 3)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_blockSize"].setObjectName(f"AdaptiveThreshold_{index}_spinBox_blockSize")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalLayout_blockSize"].addWidget(self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_blockSize"])
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["verticalLayout_blockSize"].addLayout(self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalLayout_blockSize"])
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"AdaptiveThreshold_{index}"][1]["verticalLayout_blockSize"])
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["verticalLayout_cConstantValue"].setObjectName(f"AdaptiveThreshold_{index}_verticalLayout_cConstantValue")
        font = QtGui.QFont()
        font.setPointSize(8)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["label_cConstantValue"].setFont(font)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["label_cConstantValue"].setObjectName(f"AdaptiveThreshold_{index}_label_cConstantValue")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["verticalLayout_cConstantValue"].addWidget(self.action_dict[f"AdaptiveThreshold_{index}"][1]["label_cConstantValue"])
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalLayout_cConstantValue"].setObjectName(f"AdaptiveThreshold_{index}_horizontalLayout_cConstantValue")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_cConstantValue"].setMinimumSize(QtCore.QSize(180, 22))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_cConstantValue"].setMinimum(0)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_cConstantValue"].setMaximum(200)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_cConstantValue"].setSingleStep(1)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_cConstantValue"].setProperty("value", 0)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_cConstantValue"].setSliderPosition(0)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_cConstantValue"].setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_cConstantValue"].setObjectName(f"AdaptiveThreshold_{index}_horizontalSlider_cConstantValue")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalLayout_cConstantValue"].addWidget(self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_cConstantValue"])
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_cConstantValue"].setMinimumSize(QtCore.QSize(50, 22))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_cConstantValue"].setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_cConstantValue"].setSpecialValueText("")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_cConstantValue"].setKeyboardTracking(True)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_cConstantValue"].setMinimum(0)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_cConstantValue"].setMaximum(200)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_cConstantValue"].setSingleStep(1)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_cConstantValue"].setProperty("value", 0)
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_cConstantValue"].setObjectName(f"AdaptiveThreshold_{index}_spinBox_cConstantValue")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalLayout_cConstantValue"].addWidget(self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_cConstantValue"])
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["verticalLayout_cConstantValue"].addLayout(self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalLayout_cConstantValue"])
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"AdaptiveThreshold_{index}"][1]["verticalLayout_cConstantValue"])
        
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["groupBox"].setTitle(f"Adaptive Threshold: {index} - On")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["label_maxValue"].setText("Max Value")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["label_adaptiveThresholdType"].setText("Adaptive Threshold Type")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["label_thresholdType"].setText("Threshold Type")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["label_blockSize"].setText("Block Size Value")
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["label_cConstantValue"].setText("C constant Value")
        
        self.Elements_scrollArea_verticalLayout_main.addWidget(self.action_dict[f"AdaptiveThreshold_{index}"][1]["groupBox"])
        
        self.add_action(index, 'AdaptiveThreshold')
        self.action_dict[f"AdaptiveThreshold_{index}"].append(f'cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)')
        
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_maxValue"].valueChanged.connect(lambda: self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_maxValue"].
                                                                                                    setValue(self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_maxValue"].value()))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_maxValue"].valueChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_maxValue"].valueChanged.connect(lambda: self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_maxValue"].
                                                                                                    setValue(self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_maxValue"].value()))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_maxValue"].valueChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["comboBox_adaptiveThresholdType"].currentIndexChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["comboBox_thresholdType"].currentIndexChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_blockSize"].valueChanged.connect(lambda: self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_blockSize"].
                                                                                                    setValue(self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_blockSize"].value()))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_blockSize"].valueChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_blockSize"].valueChanged.connect(lambda: self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_blockSize"].
                                                                                                    setValue(self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_blockSize"].value()))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_blockSize"].valueChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_cConstantValue"].valueChanged.connect(lambda: self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_cConstantValue"].
                                                                                                    setValue(self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_cConstantValue"].value()))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_cConstantValue"].valueChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_cConstantValue"].valueChanged.connect(lambda: self.action_dict[f"AdaptiveThreshold_{index}"][1]["horizontalSlider_cConstantValue"].
                                                                                                    setValue(self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_cConstantValue"].value()))
        self.action_dict[f"AdaptiveThreshold_{index}"][1]["spinBox_cConstantValue"].valueChanged.connect(lambda: self.input_parameters(index))


    def filter2D_add_action(self):        
        index = len(self.action_dict)
        exec(f"self.Filter2D_{index}_groupBox = QtWidgets.QGroupBox(self.Elements_scrollAreaWidgetContents)")
        exec(f"self.Filter2D_{index}_layoutWidget = QtWidgets.QWidget(self.Filter2D_{index}_groupBox)")
        exec(f"self.Filter2D_{index}_verticalLayout_all = QtWidgets.QVBoxLayout(self.Filter2D_{index}_layoutWidget)")
        exec(f"self.Filter2D_{index}_horizontalLayout_size = QtWidgets.QHBoxLayout()")
        exec(f"self.Filter2D_{index}_label_size = QtWidgets.QLabel(self.Filter2D_{index}_layoutWidget)")
        exec(f"self.Filter2D_{index}_horizontalLayout_sizeBoth = QtWidgets.QHBoxLayout()")
        exec(f"self.Filter2D_{index}_lineEdit_size_0 = QtWidgets.QLineEdit(self.Filter2D_{index}_layoutWidget)")
        exec(f"self.Filter2D_{index}_lineEdit_size_1 = QtWidgets.QLineEdit(self.Filter2D_{index}_layoutWidget)")
        exec(f"self.Filter2D_{index}_horizontalLayout_kernel = QtWidgets.QHBoxLayout()")
        exec(f"self.Filter2D_{index}_checkBox_sameValue = QtWidgets.QCheckBox(self.Filter2D_{index}_layoutWidget)")
        exec(f"self.Filter2D_{index}_horizontalLayout_sameValue = QtWidgets.QHBoxLayout()")
        exec(f"self.Filter2D_{index}_lineEdit_sameValue = QtWidgets.QLineEdit(self.Filter2D_{index}_layoutWidget)")
        exec(f"self.Filter2D_{index}_tableWidget_kernel = QtWidgets.QTableWidget(self.Filter2D_{index}_layoutWidget)")

        self.action_dict.update({f'Filter2D_{index}': 
            [index,
            {
                'groupBox': eval(f"self.Filter2D_{index}_groupBox"),
                'layoutWidget': eval(f"self.Filter2D_{index}_layoutWidget"),
                'verticalLayout_all': eval(f"self.Filter2D_{index}_verticalLayout_all"),
                'horizontalLayout_size': eval(f"self.Filter2D_{index}_horizontalLayout_size"),
                'label_size': eval(f"self.Filter2D_{index}_label_size"),
                'horizontalLayout_sizeBoth': eval(f"self.Filter2D_{index}_horizontalLayout_sizeBoth"),
                'lineEdit_size_0': eval(f"self.Filter2D_{index}_lineEdit_size_0"),
                'lineEdit_size_1': eval(f"self.Filter2D_{index}_lineEdit_size_1"),
                'horizontalLayout_kernel': eval(f"self.Filter2D_{index}_horizontalLayout_kernel"),
                'checkBox_sameValue': eval(f"self.Filter2D_{index}_checkBox_sameValue"),
                'horizontalLayout_sameValue': eval(f"self.Filter2D_{index}_horizontalLayout_sameValue"),
                'lineEdit_sameValue': eval(f"self.Filter2D_{index}_lineEdit_sameValue"),
                'tableWidget_kernel': eval(f"self.Filter2D_{index}_tableWidget_kernel")
            },
            {
                'size_0': None,
                'size_1': None,
                'sameValue_state': None,
                'sameValue': None,
                'kernel': None
            }]})
        
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.action_dict[f"Filter2D_{index}"][1]["groupBox"].sizePolicy().hasHeightForWidth())
        self.action_dict[f"Filter2D_{index}"][1]["groupBox"].setSizePolicy(sizePolicy)
        self.action_dict[f"Filter2D_{index}"][1]["groupBox"].setMinimumSize(QtCore.QSize(260, 370))
        self.action_dict[f"Filter2D_{index}"][1]["groupBox"].setMaximumSize(QtCore.QSize(260, 370))
        self.action_dict[f"Filter2D_{index}"][1]["groupBox"].setCheckable(True)
        self.action_dict[f"Filter2D_{index}"][1]["groupBox"].setObjectName(f"Filter2D_{index}_groupBox")
        self.action_dict[f"Filter2D_{index}"][1]["layoutWidget"].setGeometry(QtCore.QRect(10, 20, 242, 342))
        self.action_dict[f"Filter2D_{index}"][1]["layoutWidget"].setObjectName("layoutWidget_7")
        self.action_dict[f"Filter2D_{index}"][1]["verticalLayout_all"].setContentsMargins(0, 0, 0, 0)
        self.action_dict[f"Filter2D_{index}"][1]["verticalLayout_all"].setObjectName(f"Filter2D_{index}_verticalLayout_all")
        self.action_dict[f"Filter2D_{index}"][1]["horizontalLayout_size"].setContentsMargins(10, 10, 10, 10)
        self.action_dict[f"Filter2D_{index}"][1]["horizontalLayout_size"].setSpacing(2)
        self.action_dict[f"Filter2D_{index}"][1]["horizontalLayout_size"].setObjectName(f"Filter2D_{index}_horizontalLayout_size")
        self.action_dict[f"Filter2D_{index}"][1]["label_size"].setMinimumSize(QtCore.QSize(50, 0))
        self.action_dict[f"Filter2D_{index}"][1]["label_size"].setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f"Filter2D_{index}"][1]["label_size"].setObjectName(f"Filter2D_{index}_label_size")
        self.action_dict[f"Filter2D_{index}"][1]["horizontalLayout_size"].addWidget(self.action_dict[f"Filter2D_{index}"][1]["label_size"])
        self.action_dict[f"Filter2D_{index}"][1]["horizontalLayout_sizeBoth"].setObjectName(f"Filter2D_{index}_horizontalLayout_sizeBoth")
        self.action_dict[f"Filter2D_{index}"][1]["lineEdit_size_0"].setObjectName(f"Filter2D_{index}_lineEdit_size_0")
        self.action_dict[f"Filter2D_{index}"][1]["lineEdit_size_0"].setText('3')
        self.action_dict[f"Filter2D_{index}"][1]["horizontalLayout_sizeBoth"].addWidget(self.action_dict[f"Filter2D_{index}"][1]["lineEdit_size_0"])
        self.action_dict[f"Filter2D_{index}"][1]["lineEdit_size_1"].setObjectName(f"Filter2D_{index}_lineEdit_size_1")
        self.action_dict[f"Filter2D_{index}"][1]["lineEdit_size_1"].setText('3')
        self.action_dict[f"Filter2D_{index}"][1]["horizontalLayout_sizeBoth"].addWidget(self.action_dict[f"Filter2D_{index}"][1]["lineEdit_size_1"])
        self.action_dict[f"Filter2D_{index}"][1]["horizontalLayout_size"].addLayout(self.action_dict[f"Filter2D_{index}"][1]["horizontalLayout_sizeBoth"])
        self.action_dict[f"Filter2D_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"Filter2D_{index}"][1]["horizontalLayout_size"])
        self.action_dict[f"Filter2D_{index}"][1]["horizontalLayout_kernel"].setContentsMargins(10, 10, 10, 10)
        self.action_dict[f"Filter2D_{index}"][1]["horizontalLayout_kernel"].setSpacing(2)
        self.action_dict[f"Filter2D_{index}"][1]["horizontalLayout_kernel"].setObjectName(f"Filter2D_{index}_horizontalLayout_kernel")
        self.action_dict[f"Filter2D_{index}"][1]["checkBox_sameValue"].setObjectName(f"Filter2D_{index}_checkBox_sameValue")
        self.action_dict[f"Filter2D_{index}"][1]["horizontalLayout_kernel"].addWidget(self.action_dict[f"Filter2D_{index}"][1]["checkBox_sameValue"])
        self.action_dict[f"Filter2D_{index}"][1]["horizontalLayout_sameValue"].setObjectName(f"Filter2D_{index}_horizontalLayout_sameValue")
        self.action_dict[f"Filter2D_{index}"][1]["lineEdit_sameValue"].setObjectName(f"Filter2D_{index}_lineEdit_sameValue")
        self.action_dict[f"Filter2D_{index}"][1]["horizontalLayout_sameValue"].addWidget(self.action_dict[f"Filter2D_{index}"][1]["lineEdit_sameValue"])
        self.action_dict[f"Filter2D_{index}"][1]["horizontalLayout_kernel"].addLayout(self.action_dict[f"Filter2D_{index}"][1]["horizontalLayout_sameValue"])
        self.action_dict[f"Filter2D_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"Filter2D_{index}"][1]["horizontalLayout_kernel"])
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setMinimumSize(QtCore.QSize(240, 240))
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setMaximumSize(QtCore.QSize(240, 240))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setFont(font)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setAutoFillBackground(False)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setStyleSheet("")
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setLineWidth(1)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setMidLineWidth(1)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setAutoScroll(True)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setAutoScrollMargin(16)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.AnyKeyPressed|QtWidgets.QAbstractItemView.EditTrigger.CurrentChanged|QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked|QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setAlternatingRowColors(False)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setTextElideMode(QtCore.Qt.TextElideMode.ElideMiddle)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setShowGrid(True)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setCornerButtonEnabled(True)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setRowCount(3)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setColumnCount(3)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setObjectName(f"Filter2D_{index}_tableWidget_kernel")
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].horizontalHeader().setVisible(False)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].horizontalHeader().setCascadingSectionResizes(False)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].horizontalHeader().setDefaultSectionSize(40)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].horizontalHeader().setMinimumSectionSize(40)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].horizontalHeader().setStretchLastSection(False)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].verticalHeader().setVisible(False)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].verticalHeader().setCascadingSectionResizes(False)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].verticalHeader().setDefaultSectionSize(40)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].verticalHeader().setMinimumSectionSize(40)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].verticalHeader().setSortIndicatorShown(False)
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].verticalHeader().setStretchLastSection(False)
        self.action_dict[f"Filter2D_{index}"][1]["verticalLayout_all"].addWidget(self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"])
        self.Elements_scrollArea_verticalLayout_main.addWidget(self.action_dict[f"Filter2D_{index}"][1]["groupBox"])
        
        self.action_dict[f"Filter2D_{index}"][1]["groupBox"].setTitle(f"Filter2D: {index} - On")
        self.action_dict[f"Filter2D_{index}"][1]["label_size"].setText("Size")
        self.action_dict[f"Filter2D_{index}"][1]["checkBox_sameValue"].setText("Same Value")
        self.action_dict[f"Filter2D_{index}"][1]["tableWidget_kernel"].setSortingEnabled(False)
        
        # self.Filter2D_test_checkBox_sameValue = QtWidgets.QTableWidget(self.Filter2D_test_layoutWidget)
        # self.Filter2D_test_checkBox_sameValue.setDisabled
        self.action_dict[f"Filter2D_{index}"][2]['size_0'] = self.action_dict[f"Filter2D_{index}"][1]['lineEdit_size_0'].text()
        self.action_dict[f"Filter2D_{index}"][2]['size_1'] = self.action_dict[f"Filter2D_{index}"][1]['lineEdit_size_1'].text()
        self.add_action(index, 'Filter2D')
        self.kernel(f'Filter2D_{index}')
        kernel = self.action_dict[f'Filter2D_{index}'][2]['kernel']
        self.action_dict[f"Filter2D_{index}"].append(f'cv2.filter2D(image, -1, {kernel})')
        
        self.action_dict[f'Filter2D_{index}'][1]['lineEdit_size_0'].textChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f'Filter2D_{index}'][1]['lineEdit_size_1'].textChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f'Filter2D_{index}'][1]['checkBox_sameValue'].clicked.connect(lambda: self.input_parameters(index))
        self.action_dict[f'Filter2D_{index}'][1]['lineEdit_sameValue'].textChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f'Filter2D_{index}'][1]['tableWidget_kernel'].cellChanged.connect(lambda: self.input_parameters(index))
        

    def bilateralFilter_add_action(self):        
        index = len(self.action_dict)
        exec(f"self.BilateralFilter_{index}_groupBox = QtWidgets.QGroupBox(self.Elements_scrollAreaWidgetContents)")
        exec(f"self.BilateralFilter_{index}_layoutWidget = QtWidgets.QWidget(self.BilateralFilter_{index}_groupBox)")
        exec(f"self.BilateralFilter_{index}_verticalLayout_all = QtWidgets.QVBoxLayout(self.BilateralFilter_{index}_layoutWidget)")
        exec(f"self.BilateralFilter_{index}_verticalLayout_dSize = QtWidgets.QVBoxLayout()")
        exec(f"self.BilateralFilter_{index}_label_dSize = QtWidgets.QLabel(self.BilateralFilter_{index}_layoutWidget)")
        exec(f"self.BilateralFilter_{index}_horizontalLayout_dSize = QtWidgets.QHBoxLayout()")
        exec(f"self.BilateralFilter_{index}_horizontalSlider_dSize = QtWidgets.QSlider(self.BilateralFilter_{index}_layoutWidget)")
        exec(f"self.BilateralFilter_{index}_spinBox_dSize = QtWidgets.QSpinBox(self.BilateralFilter_{index}_layoutWidget)")
        exec(f"self.BilateralFilter_{index}_verticalLayout_sigmaFirst = QtWidgets.QVBoxLayout()")
        exec(f"self.BilateralFilter_{index}_label_sigmaFirst = QtWidgets.QLabel(self.BilateralFilter_{index}_layoutWidget)")
        exec(f"self.BilateralFilter_{index}_horizontalLayout_sigmaFirst = QtWidgets.QHBoxLayout()")
        exec(f"self.BilateralFilter_{index}_horizontalSlider_sigmaFirst = QtWidgets.QSlider(self.BilateralFilter_{index}_layoutWidget)")
        exec(f"self.BilateralFilter_{index}_spinBox_sigmaFirst = QtWidgets.QSpinBox(self.BilateralFilter_{index}_layoutWidget)")
        exec(f"self.BilateralFilter_{index}_verticalLayout_sigmaSecond = QtWidgets.QVBoxLayout()")
        exec(f"self.BilateralFilter_{index}_label_sigmaSecond = QtWidgets.QLabel(self.BilateralFilter_{index}_layoutWidget)")
        exec(f"self.BilateralFilter_{index}_horizontalLayout_sigmaSecond = QtWidgets.QHBoxLayout()")
        exec(f"self.BilateralFilter_{index}_horizontalSlider_sigmaSecond = QtWidgets.QSlider(self.BilateralFilter_{index}_layoutWidget)")
        exec(f"self.BilateralFilter_{index}_spinBox_sigmaSecond = QtWidgets.QSpinBox(self.BilateralFilter_{index}_layoutWidget)")

        self.action_dict.update({f'BilateralFilter_{index}': 
            [index,
            {
                'groupBox': eval(f"self.BilateralFilter_{index}_groupBox"),
                'layoutWidget': eval(f"self.BilateralFilter_{index}_layoutWidget"),
                'verticalLayout_all': eval(f"self.BilateralFilter_{index}_verticalLayout_all"),
                'verticalLayout_dSize': eval(f"self.BilateralFilter_{index}_verticalLayout_dSize"),
                'label_dSize': eval(f"self.BilateralFilter_{index}_label_dSize"),
                'horizontalLayout_dSize': eval(f"self.BilateralFilter_{index}_horizontalLayout_dSize"),
                'horizontalSlider_dSize': eval(f"self.BilateralFilter_{index}_horizontalSlider_dSize"),
                'spinBox_dSize': eval(f"self.BilateralFilter_{index}_spinBox_dSize"),
                'verticalLayout_sigmaFirst': eval(f"self.BilateralFilter_{index}_verticalLayout_sigmaFirst"),
                'label_sigmaFirst': eval(f"self.BilateralFilter_{index}_label_sigmaFirst"),
                'horizontalLayout_sigmaFirst': eval(f"self.BilateralFilter_{index}_horizontalLayout_sigmaFirst"),
                'horizontalSlider_sigmaFirst': eval(f"self.BilateralFilter_{index}_horizontalSlider_sigmaFirst"),
                'spinBox_sigmaFirst': eval(f"self.BilateralFilter_{index}_spinBox_sigmaFirst"),
                'verticalLayout_sigmaSecond': eval(f"self.BilateralFilter_{index}_verticalLayout_sigmaSecond"),
                'label_sigmaSecond': eval(f"self.BilateralFilter_{index}_label_sigmaSecond"),
                'horizontalLayout_sigmaSecond': eval(f"self.BilateralFilter_{index}_horizontalLayout_sigmaSecond"),
                'horizontalSlider_sigmaSecond': eval(f"self.BilateralFilter_{index}_horizontalSlider_sigmaSecond"),
                'spinBox_sigmaSecond': eval(f"self.BilateralFilter_{index}_spinBox_sigmaSecond")
            },
            {
                'dSize': None,
                'sigmaFirst': None,
                'sigmaSecond': None
            }
            ]})
        
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.action_dict[f"BilateralFilter_{index}"][1]["groupBox"].sizePolicy().hasHeightForWidth())
        self.action_dict[f"BilateralFilter_{index}"][1]["groupBox"].setSizePolicy(sizePolicy)
        self.action_dict[f"BilateralFilter_{index}"][1]["groupBox"].setMinimumSize(QtCore.QSize(260, 190))
        self.action_dict[f"BilateralFilter_{index}"][1]["groupBox"].setMaximumSize(QtCore.QSize(260, 190))
        self.action_dict[f"BilateralFilter_{index}"][1]["groupBox"].setCheckable(True)
        self.action_dict[f"BilateralFilter_{index}"][1]["groupBox"].setObjectName(f"BilateralFilter_{index}_groupBox")
        self.action_dict[f"BilateralFilter_{index}"][1]["layoutWidget"].setGeometry(QtCore.QRect(10, 20, 244, 161))
        self.action_dict[f"BilateralFilter_{index}"][1]["layoutWidget"].setObjectName("layoutWidget_4")
        self.action_dict[f"BilateralFilter_{index}"][1]["verticalLayout_all"].setContentsMargins(0, 0, 0, 0)
        self.action_dict[f"BilateralFilter_{index}"][1]["verticalLayout_all"].setObjectName(f"BilateralFilter_{index}_verticalLayout_all")
        self.action_dict[f"BilateralFilter_{index}"][1]["verticalLayout_dSize"].setObjectName(f"BilateralFilter_{index}_verticalLayout_dSize")
        font = QtGui.QFont()
        font.setPointSize(8)
        self.action_dict[f"BilateralFilter_{index}"][1]["label_dSize"].setFont(font)
        self.action_dict[f"BilateralFilter_{index}"][1]["label_dSize"].setObjectName(f"BilateralFilter_{index}_label_dSize")
        self.action_dict[f"BilateralFilter_{index}"][1]["verticalLayout_dSize"].addWidget(self.action_dict[f"BilateralFilter_{index}"][1]["label_dSize"])
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalLayout_dSize"].setObjectName(f"BilateralFilter_{index}_horizontalLayout_dSize")
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_dSize"].setMinimumSize(QtCore.QSize(180, 22))
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_dSize"].setMinimum(0)
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_dSize"].setMaximum(50)
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_dSize"].setProperty("value", 0)
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_dSize"].setSliderPosition(0)
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_dSize"].setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_dSize"].setObjectName(f"BilateralFilter_{index}_horizontalSlider_dSize")
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalLayout_dSize"].addWidget(self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_dSize"])
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_dSize"].setMinimumSize(QtCore.QSize(50, 22))
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_dSize"].setMinimum(0)
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_dSize"].setMaximum(50)
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_dSize"].setProperty("value", 0)
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_dSize"].setObjectName(f"BilateralFilter_{index}_spinBox_dSize")
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalLayout_dSize"].addWidget(self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_dSize"])
        self.action_dict[f"BilateralFilter_{index}"][1]["verticalLayout_dSize"].addLayout(self.action_dict[f"BilateralFilter_{index}"][1]["horizontalLayout_dSize"])
        self.action_dict[f"BilateralFilter_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"BilateralFilter_{index}"][1]["verticalLayout_dSize"])
        self.action_dict[f"BilateralFilter_{index}"][1]["verticalLayout_sigmaFirst"].setObjectName(f"BilateralFilter_{index}_verticalLayout_sigmaFirst")
        font = QtGui.QFont()
        font.setPointSize(8)
        self.action_dict[f"BilateralFilter_{index}"][1]["label_sigmaFirst"].setFont(font)
        self.action_dict[f"BilateralFilter_{index}"][1]["label_sigmaFirst"].setObjectName(f"BilateralFilter_{index}_label_sigmaFirst")
        self.action_dict[f"BilateralFilter_{index}"][1]["verticalLayout_sigmaFirst"].addWidget(self.action_dict[f"BilateralFilter_{index}"][1]["label_sigmaFirst"])
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalLayout_sigmaFirst"].setObjectName(f"BilateralFilter_{index}_horizontalLayout_sigmaFirst")
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaFirst"].setMinimumSize(QtCore.QSize(180, 22))
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaFirst"].setMinimum(0)
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaFirst"].setMaximum(300)
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaFirst"].setSingleStep(1)
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaFirst"].setProperty("value", 0)
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaFirst"].setSliderPosition(0)
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaFirst"].setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaFirst"].setObjectName(f"BilateralFilter_{index}_horizontalSlider_sigmaFirst")
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalLayout_sigmaFirst"].addWidget(self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaFirst"])
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaFirst"].setMinimumSize(QtCore.QSize(50, 22))
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaFirst"].setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaFirst"].setSpecialValueText("")
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaFirst"].setKeyboardTracking(True)
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaFirst"].setMinimum(0)
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaFirst"].setMaximum(300)
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaFirst"].setSingleStep(1)
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaFirst"].setProperty("value", 0)
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaFirst"].setObjectName(f"BilateralFilter_{index}_spinBox_sigmaFirst")
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalLayout_sigmaFirst"].addWidget(self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaFirst"])
        self.action_dict[f"BilateralFilter_{index}"][1]["verticalLayout_sigmaFirst"].addLayout(self.action_dict[f"BilateralFilter_{index}"][1]["horizontalLayout_sigmaFirst"])
        self.action_dict[f"BilateralFilter_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"BilateralFilter_{index}"][1]["verticalLayout_sigmaFirst"])
        self.action_dict[f"BilateralFilter_{index}"][1]["verticalLayout_sigmaSecond"].setObjectName(f"BilateralFilter_{index}_verticalLayout_sigmaSecond")
        font = QtGui.QFont()
        font.setPointSize(8)
        self.action_dict[f"BilateralFilter_{index}"][1]["label_sigmaSecond"].setFont(font)
        self.action_dict[f"BilateralFilter_{index}"][1]["label_sigmaSecond"].setObjectName(f"BilateralFilter_{index}_label_sigmaSecond")
        self.action_dict[f"BilateralFilter_{index}"][1]["verticalLayout_sigmaSecond"].addWidget(self.action_dict[f"BilateralFilter_{index}"][1]["label_sigmaSecond"])
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalLayout_sigmaSecond"].setObjectName(f"BilateralFilter_{index}_horizontalLayout_sigmaSecond")
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaSecond"].setMinimumSize(QtCore.QSize(180, 22))
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaSecond"].setMinimum(0)
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaSecond"].setMaximum(300)
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaSecond"].setSingleStep(1)
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaSecond"].setProperty("value", 0)
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaSecond"].setSliderPosition(0)
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaSecond"].setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaSecond"].setObjectName(f"BilateralFilter_{index}_horizontalSlider_sigmaSecond")
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalLayout_sigmaSecond"].addWidget(self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaSecond"])
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaSecond"].setMinimumSize(QtCore.QSize(50, 22))
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaSecond"].setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaSecond"].setSpecialValueText("")
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaSecond"].setKeyboardTracking(True)
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaSecond"].setMinimum(0)
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaSecond"].setMaximum(300)
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaSecond"].setSingleStep(1)
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaSecond"].setProperty("value", 0)
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaSecond"].setObjectName(f"BilateralFilter_{index}_spinBox_sigmaSecond")
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalLayout_sigmaSecond"].addWidget(self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaSecond"])
        self.action_dict[f"BilateralFilter_{index}"][1]["verticalLayout_sigmaSecond"].addLayout(self.action_dict[f"BilateralFilter_{index}"][1]["horizontalLayout_sigmaSecond"])
        self.action_dict[f"BilateralFilter_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"BilateralFilter_{index}"][1]["verticalLayout_sigmaSecond"])
        self.Elements_scrollArea_verticalLayout_main.addWidget(self.action_dict[f"BilateralFilter_{index}"][1]["groupBox"])
        
        self.action_dict[f"BilateralFilter_{index}"][1]["groupBox"].setTitle(f"Bilateral Filter: {index} - On")
        self.action_dict[f"BilateralFilter_{index}"][1]["label_dSize"].setText("Filter Size (d)")
        self.action_dict[f"BilateralFilter_{index}"][1]["label_sigmaFirst"].setText("Sigma First Value")
        self.action_dict[f"BilateralFilter_{index}"][1]["label_sigmaSecond"].setText("Sigma Second Value")
        
        self.add_action(index, 'BilateralFilter')
        self.action_dict[f"BilateralFilter_{index}"].append(f'cv2.bilateralFilter(image, 0, 0, 0)')
        
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_dSize"].valueChanged.connect(lambda: self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_dSize"].
                                                                                                    setValue(self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_dSize"].value()))
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_dSize"].valueChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_dSize"].valueChanged.connect(lambda: self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_dSize"].
                                                                                                    setValue(self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_dSize"].value()))
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_dSize"].valueChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaFirst"].valueChanged.connect(lambda: self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaFirst"].
                                                                                                    setValue(self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaFirst"].value()))
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaFirst"].valueChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaFirst"].valueChanged.connect(lambda: self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaFirst"].
                                                                                                    setValue(self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaFirst"].value()))
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaFirst"].valueChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaSecond"].valueChanged.connect(lambda: self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaSecond"].
                                                                                                    setValue(self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaSecond"].value()))
        self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaSecond"].valueChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaSecond"].valueChanged.connect(lambda: self.action_dict[f"BilateralFilter_{index}"][1]["horizontalSlider_sigmaSecond"].
                                                                                                    setValue(self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaSecond"].value()))
        self.action_dict[f"BilateralFilter_{index}"][1]["spinBox_sigmaSecond"].valueChanged.connect(lambda: self.input_parameters(index))
     

    def createCLAHE_add_action(self):        
        index = len(self.action_dict)
        exec(f"self.CLAHEFilter_{index}_groupBox = QtWidgets.QGroupBox(self.Elements_scrollAreaWidgetContents)")
        exec(f"self.CLAHEFilter_{index}_layoutWidget = QtWidgets.QWidget(self.CLAHEFilter_{index}_groupBox)")
        exec(f"self.CLAHEFilter_{index}_verticalLayout_all = QtWidgets.QVBoxLayout(self.CLAHEFilter_{index}_layoutWidget)")
        exec(f"self.CLAHEFilter_{index}_verticalLayout_clipLimit = QtWidgets.QVBoxLayout()")
        exec(f"self.CLAHEFilter_{index}_label_clipLimit = QtWidgets.QLabel(self.CLAHEFilter_{index}_layoutWidget)")
        exec(f"self.CLAHEFilter_{index}_horizontalLayout_clipLimit = QtWidgets.QHBoxLayout()")
        exec(f"self.CLAHEFilter_{index}_horizontalSlider_clipLimit = QtWidgets.QSlider(self.CLAHEFilter_{index}_layoutWidget)")
        exec(f"self.CLAHEFilter_{index}_doubleSpinBox_clipLimit = QtWidgets.QDoubleSpinBox(self.CLAHEFilter_{index}_layoutWidget)")
        exec(f"self.CLAHEFilter_{index}_horizontalLayout_size = QtWidgets.QHBoxLayout()")
        exec(f"self.CLAHEFilter_{index}_label_Size = QtWidgets.QLabel(self.CLAHEFilter_{index}_layoutWidget)")
        exec(f"self.CLAHEFilter_{index}_horizontalLayout_sizeBoth = QtWidgets.QHBoxLayout()")
        exec(f"self.CLAHEFilter_{index}_lineEdit_size_0 = QtWidgets.QLineEdit(self.CLAHEFilter_{index}_layoutWidget)")
        exec(f"self.CLAHEFilter_{index}_lineEdit_size_1 = QtWidgets.QLineEdit(self.CLAHEFilter_{index}_layoutWidget)")

        self.action_dict.update({f'CLAHEFilter_{index}': 
            [index,
            {
                'groupBox': eval(f"self.CLAHEFilter_{index}_groupBox"),
                'layoutWidget': eval(f"self.CLAHEFilter_{index}_layoutWidget"),
                'verticalLayout_all': eval(f"self.CLAHEFilter_{index}_verticalLayout_all"),
                'verticalLayout_clipLimit': eval(f"self.CLAHEFilter_{index}_verticalLayout_clipLimit"),
                'label_clipLimit': eval(f"self.CLAHEFilter_{index}_label_clipLimit"),
                'horizontalLayout_clipLimit': eval(f"self.CLAHEFilter_{index}_horizontalLayout_clipLimit"),
                'horizontalSlider_clipLimit': eval(f"self.CLAHEFilter_{index}_horizontalSlider_clipLimit"),
                'doubleSpinBox_clipLimit': eval(f"self.CLAHEFilter_{index}_doubleSpinBox_clipLimit"),
                'horizontalLayout_size': eval(f"self.CLAHEFilter_{index}_horizontalLayout_size"),
                'label_Size': eval(f"self.CLAHEFilter_{index}_label_Size"),
                'horizontalLayout_sizeBoth': eval(f"self.CLAHEFilter_{index}_horizontalLayout_sizeBoth"),
                'lineEdit_size_0': eval(f"self.CLAHEFilter_{index}_lineEdit_size_0"),
                'lineEdit_size_1': eval(f"self.CLAHEFilter_{index}_lineEdit_size_1")
            },
            {
                'clipLimit': None,
                'size_0': None,
                'size_1': None
            }]})
                
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.action_dict[f"CLAHEFilter_{index}"][1]["groupBox"].sizePolicy().hasHeightForWidth())
        self.action_dict[f"CLAHEFilter_{index}"][1]["groupBox"].setSizePolicy(sizePolicy)
        self.action_dict[f"CLAHEFilter_{index}"][1]["groupBox"].setMinimumSize(QtCore.QSize(260, 120))
        self.action_dict[f"CLAHEFilter_{index}"][1]["groupBox"].setMaximumSize(QtCore.QSize(260, 120))
        self.action_dict[f"CLAHEFilter_{index}"][1]["groupBox"].setCheckable(True)
        self.action_dict[f"CLAHEFilter_{index}"][1]["groupBox"].setObjectName(f"CLAHEFilter_{index}_groupBox")
        self.action_dict[f"CLAHEFilter_{index}"][1]["layoutWidget"].setGeometry(QtCore.QRect(10, 20, 241, 95))
        self.action_dict[f"CLAHEFilter_{index}"][1]["layoutWidget"].setObjectName("layoutWidget4")
        self.action_dict[f"CLAHEFilter_{index}"][1]["verticalLayout_all"].setContentsMargins(0, 0, 0, 0)
        self.action_dict[f"CLAHEFilter_{index}"][1]["verticalLayout_all"].setObjectName(f"CLAHEFilter_{index}_verticalLayout_all")
        self.action_dict[f"CLAHEFilter_{index}"][1]["verticalLayout_clipLimit"].setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        self.action_dict[f"CLAHEFilter_{index}"][1]["verticalLayout_clipLimit"].setObjectName(f"CLAHEFilter_{index}_verticalLayout_clipLimit")
        self.action_dict[f"CLAHEFilter_{index}"][1]["label_clipLimit"].setMaximumSize(QtCore.QSize(16777215, 22))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.action_dict[f"CLAHEFilter_{index}"][1]["label_clipLimit"].setFont(font)
        self.action_dict[f"CLAHEFilter_{index}"][1]["label_clipLimit"].setObjectName(f"CLAHEFilter_{index}_label_clipLimit")
        self.action_dict[f"CLAHEFilter_{index}"][1]["verticalLayout_clipLimit"].addWidget(self.action_dict[f"CLAHEFilter_{index}"][1]["label_clipLimit"])
        self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalLayout_clipLimit"].setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalLayout_clipLimit"].setObjectName(f"CLAHEFilter_{index}_horizontalLayout_clipLimit")
        self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalSlider_clipLimit"].setMinimumSize(QtCore.QSize(180, 22))
        self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalSlider_clipLimit"].setMinimum(0)
        self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalSlider_clipLimit"].setMaximum(8000)
        self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalSlider_clipLimit"].setProperty("value", 0)
        self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalSlider_clipLimit"].setSliderPosition(0)
        self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalSlider_clipLimit"].setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalSlider_clipLimit"].setObjectName(f"CLAHEFilter_{index}_horizontalSlider_clipLimit")
        self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalLayout_clipLimit"].addWidget(self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalSlider_clipLimit"])
        self.action_dict[f"CLAHEFilter_{index}"][1]["doubleSpinBox_clipLimit"].setDecimals(1)
        self.action_dict[f"CLAHEFilter_{index}"][1]["doubleSpinBox_clipLimit"].setMaximum(800.0)
        self.action_dict[f"CLAHEFilter_{index}"][1]["doubleSpinBox_clipLimit"].setSingleStep(0.1)
        self.action_dict[f"CLAHEFilter_{index}"][1]["doubleSpinBox_clipLimit"].setObjectName(f"CLAHEFilter_{index}_doubleSpinBox_clipLimit")
        self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalLayout_clipLimit"].addWidget(self.action_dict[f"CLAHEFilter_{index}"][1]["doubleSpinBox_clipLimit"])
        self.action_dict[f"CLAHEFilter_{index}"][1]["verticalLayout_clipLimit"].addLayout(self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalLayout_clipLimit"])
        self.action_dict[f"CLAHEFilter_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"CLAHEFilter_{index}"][1]["verticalLayout_clipLimit"])
        self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalLayout_size"].setContentsMargins(10, 10, 10, 10)
        self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalLayout_size"].setSpacing(2)
        self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalLayout_size"].setObjectName(f"CLAHEFilter_{index}_horizontalLayout_size")
        self.action_dict[f"CLAHEFilter_{index}"][1]["label_Size"].setMinimumSize(QtCore.QSize(50, 0))
        self.action_dict[f"CLAHEFilter_{index}"][1]["label_Size"].setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f"CLAHEFilter_{index}"][1]["label_Size"].setObjectName(f"CLAHEFilter_{index}_label_Size")
        self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalLayout_size"].addWidget(self.action_dict[f"CLAHEFilter_{index}"][1]["label_Size"])
        self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalLayout_sizeBoth"].setObjectName(f"CLAHEFilter_{index}_horizontalLayout_sizeBoth")
        self.action_dict[f"CLAHEFilter_{index}"][1]["lineEdit_size_0"].setObjectName(f"CLAHEFilter_{index}_lineEdit_size_0")
        self.action_dict[f"CLAHEFilter_{index}"][1]["lineEdit_size_0"].setText('3')
        self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalLayout_sizeBoth"].addWidget(self.action_dict[f"CLAHEFilter_{index}"][1]["lineEdit_size_0"])
        self.action_dict[f"CLAHEFilter_{index}"][1]["lineEdit_size_1"].setObjectName(f"CLAHEFilter_{index}_lineEdit_size_1")
        self.action_dict[f"CLAHEFilter_{index}"][1]["lineEdit_size_1"].setText('3')
        self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalLayout_sizeBoth"].addWidget(self.action_dict[f"CLAHEFilter_{index}"][1]["lineEdit_size_1"])
        self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalLayout_size"].addLayout(self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalLayout_sizeBoth"])
        self.action_dict[f"CLAHEFilter_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalLayout_size"])
        self.Elements_scrollArea_verticalLayout_main.addWidget(self.action_dict[f"CLAHEFilter_{index}"][1]["groupBox"])
        
        self.add_action(index, 'CLAHEFilter')
        self.action_dict[f"CLAHEFilter_{index}"].append(f'cv2.createCLAHE(clipLimit=0, tileGridSize=(3, 3))')
        
        self.action_dict[f"CLAHEFilter_{index}"][1]["groupBox"].setTitle(f"CLAHE Filter: {index} - On")
        self.action_dict[f"CLAHEFilter_{index}"][1]["label_clipLimit"].setText("Clip Limit")
        self.action_dict[f"CLAHEFilter_{index}"][1]["label_Size"].setText("Size")
        
        self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalSlider_clipLimit"].valueChanged.connect(lambda: self.action_dict[f"CLAHEFilter_{index}"][1]["doubleSpinBox_clipLimit"].
                                                                                                    setValue(self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalSlider_clipLimit"].value()/10))
        self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalSlider_clipLimit"].valueChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f"CLAHEFilter_{index}"][1]["doubleSpinBox_clipLimit"].valueChanged.connect(lambda: self.action_dict[f"CLAHEFilter_{index}"][1]["horizontalSlider_clipLimit"].
                                                                                                    setValue(int(self.action_dict[f"CLAHEFilter_{index}"][1]["doubleSpinBox_clipLimit"].value()*10)))
        self.action_dict[f"CLAHEFilter_{index}"][1]["doubleSpinBox_clipLimit"].valueChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f'CLAHEFilter_{index}'][1]['lineEdit_size_0'].textChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f'CLAHEFilter_{index}'][1]['lineEdit_size_1'].textChanged.connect(lambda: self.input_parameters(index))
      

    def morphologyEx_add_action(self):       
        index = len(self.action_dict)
        exec(f"self.Morphology_{index}_groupBox = QtWidgets.QGroupBox(self.Elements_scrollAreaWidgetContents)")
        exec(f"self.Morphology_{index}_layoutWidget = QtWidgets.QWidget(self.Morphology_{index}_groupBox)")
        exec(f"self.Morphology_{index}_verticalLayout_all = QtWidgets.QVBoxLayout(self.Morphology_{index}_layoutWidget)")
        exec(f"self.Morphology_{index}_formLayout_functionType = QtWidgets.QFormLayout()")
        exec(f"self.Morphology_{index}_label_functionType = QtWidgets.QLabel(self.Morphology_{index}_layoutWidget)")
        exec(f"self.Morphology_{index}_comboBox_functionType = QtWidgets.QComboBox(self.Morphology_{index}_layoutWidget)")
        exec(f"self.Morphology_{index}_horizontalLayout_size = QtWidgets.QHBoxLayout()")
        exec(f"self.Morphology_{index}_label_size = QtWidgets.QLabel(self.Morphology_{index}_layoutWidget)")
        exec(f"self.Morphology_{index}_horizontalLayout_sizeBoth = QtWidgets.QHBoxLayout()")
        exec(f"self.Morphology_{index}_lineEdit_size_0 = QtWidgets.QLineEdit(self.Morphology_{index}_layoutWidget)")
        exec(f"self.Morphology_{index}_lineEdit_size_1 = QtWidgets.QLineEdit(self.Morphology_{index}_layoutWidget)")
        exec(f"self.Morphology_{index}_horizontalLayout_kernel = QtWidgets.QHBoxLayout()")
        exec(f"self.Morphology_{index}_checkBox_sameValue = QtWidgets.QCheckBox(self.Morphology_{index}_layoutWidget)")
        exec(f"self.Morphology_{index}_horizontalLayout_sameValue = QtWidgets.QHBoxLayout()")
        exec(f"self.Morphology_{index}_lineEdit_sameValue = QtWidgets.QLineEdit(self.Morphology_{index}_layoutWidget)")
        exec(f"self.Morphology_{index}_tableWidget_kernel = QtWidgets.QTableWidget(self.Morphology_{index}_layoutWidget)")

        self.action_dict.update({f'Morphology_{index}': 
            [index,
            {
                'groupBox': eval(f"self.Morphology_{index}_groupBox"),
                'layoutWidget': eval(f"self.Morphology_{index}_layoutWidget"),
                'verticalLayout_all': eval(f"self.Morphology_{index}_verticalLayout_all"),
                'formLayout_functionType': eval(f"self.Morphology_{index}_formLayout_functionType"),
                'label_functionType': eval(f"self.Morphology_{index}_label_functionType"),
                'comboBox_functionType': eval(f"self.Morphology_{index}_comboBox_functionType"),
                'horizontalLayout_size': eval(f"self.Morphology_{index}_horizontalLayout_size"),
                'label_size': eval(f"self.Morphology_{index}_label_size"),
                'horizontalLayout_sizeBoth': eval(f"self.Morphology_{index}_horizontalLayout_sizeBoth"),
                'lineEdit_size_0': eval(f"self.Morphology_{index}_lineEdit_size_0"),
                'lineEdit_size_1': eval(f"self.Morphology_{index}_lineEdit_size_1"),
                'horizontalLayout_kernel': eval(f"self.Morphology_{index}_horizontalLayout_kernel"),
                'checkBox_sameValue': eval(f"self.Morphology_{index}_checkBox_sameValue"),
                'horizontalLayout_sameValue': eval(f"self.Morphology_{index}_horizontalLayout_sameValue"),
                'lineEdit_sameValue': eval(f"self.Morphology_{index}_lineEdit_sameValue"),
                'tableWidget_kernel': eval(f"self.Morphology_{index}_tableWidget_kernel")
            },
            {
                'functionType': None,
                'size_0': None,
                'size_1': None,
                'sameValue': None,
                'kernel': None
            }
            ]})
        
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.action_dict[f"Morphology_{index}"][1]["groupBox"].sizePolicy().hasHeightForWidth())
        self.action_dict[f"Morphology_{index}"][1]["groupBox"].setSizePolicy(sizePolicy)
        self.action_dict[f"Morphology_{index}"][1]["groupBox"].setMinimumSize(QtCore.QSize(260, 420))
        self.action_dict[f"Morphology_{index}"][1]["groupBox"].setMaximumSize(QtCore.QSize(260, 420))
        self.action_dict[f"Morphology_{index}"][1]["groupBox"].setCheckable(True)
        self.action_dict[f"Morphology_{index}"][1]["groupBox"].setObjectName(f"Morphology_{index}_groupBox")
        self.action_dict[f"Morphology_{index}"][1]["layoutWidget"].setGeometry(QtCore.QRect(10, 20, 242, 384))
        self.action_dict[f"Morphology_{index}"][1]["layoutWidget"].setObjectName("layoutWidget5")
        self.action_dict[f"Morphology_{index}"][1]["verticalLayout_all"].setContentsMargins(0, 0, 0, 0)
        self.action_dict[f"Morphology_{index}"][1]["verticalLayout_all"].setObjectName(f"Morphology_{index}_verticalLayout_all")
        self.action_dict[f"Morphology_{index}"][1]["formLayout_functionType"].setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.action_dict[f"Morphology_{index}"][1]["formLayout_functionType"].setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.action_dict[f"Morphology_{index}"][1]["formLayout_functionType"].setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f"Morphology_{index}"][1]["formLayout_functionType"].setFormAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f"Morphology_{index}"][1]["formLayout_functionType"].setContentsMargins(10, 10, 10, 10)
        self.action_dict[f"Morphology_{index}"][1]["formLayout_functionType"].setObjectName(f"Morphology_{index}_formLayout_functionType")
        self.action_dict[f"Morphology_{index}"][1]["label_functionType"].setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f"Morphology_{index}"][1]["label_functionType"].setObjectName(f"Morphology_{index}_label_functionType")
        self.action_dict[f"Morphology_{index}"][1]["formLayout_functionType"].setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.action_dict[f"Morphology_{index}"][1]["label_functionType"])
        self.action_dict[f"Morphology_{index}"][1]["comboBox_functionType"].setEditable(False)
        self.action_dict[f"Morphology_{index}"][1]["comboBox_functionType"].setMaxVisibleItems(10)
        self.action_dict[f"Morphology_{index}"][1]["comboBox_functionType"].setMaxCount(5)
        self.action_dict[f"Morphology_{index}"][1]["comboBox_functionType"].setObjectName(f"Morphology_{index}_comboBox_functionType")
        self.action_dict[f"Morphology_{index}"][1]["comboBox_functionType"].addItem("MORPH_OPEN")
        self.action_dict[f"Morphology_{index}"][1]["comboBox_functionType"].addItem("MORPH_CLOSE")
        self.action_dict[f"Morphology_{index}"][1]["comboBox_functionType"].addItem("MORPH_GRADIENT")
        self.action_dict[f"Morphology_{index}"][1]["comboBox_functionType"].addItem("MORPH_TOPHAT")
        self.action_dict[f"Morphology_{index}"][1]["comboBox_functionType"].addItem("MORPH_BLACKHAT")
        self.action_dict[f"Morphology_{index}"][1]["formLayout_functionType"].setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.action_dict[f"Morphology_{index}"][1]["comboBox_functionType"])
        self.action_dict[f"Morphology_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"Morphology_{index}"][1]["formLayout_functionType"])
        self.action_dict[f"Morphology_{index}"][1]["horizontalLayout_size"].setContentsMargins(10, 10, 10, 10)
        self.action_dict[f"Morphology_{index}"][1]["horizontalLayout_size"].setSpacing(2)
        self.action_dict[f"Morphology_{index}"][1]["horizontalLayout_size"].setObjectName(f"Morphology_{index}_horizontalLayout_size")
        self.action_dict[f"Morphology_{index}"][1]["label_size"].setMinimumSize(QtCore.QSize(50, 0))
        self.action_dict[f"Morphology_{index}"][1]["label_size"].setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f"Morphology_{index}"][1]["label_size"].setObjectName(f"Morphology_{index}_label_size")
        self.action_dict[f"Morphology_{index}"][1]["horizontalLayout_size"].addWidget(self.action_dict[f"Morphology_{index}"][1]["label_size"])
        self.action_dict[f"Morphology_{index}"][1]["horizontalLayout_sizeBoth"].setObjectName(f"Morphology_{index}_horizontalLayout_sizeBoth")
        self.action_dict[f"Morphology_{index}"][1]["lineEdit_size_0"].setObjectName(f"Morphology_{index}_lineEdit_size_0")
        self.action_dict[f"Morphology_{index}"][1]["lineEdit_size_0"].setText('3')
        self.action_dict[f"Morphology_{index}"][1]["horizontalLayout_sizeBoth"].addWidget(self.action_dict[f"Morphology_{index}"][1]["lineEdit_size_0"])
        self.action_dict[f"Morphology_{index}"][1]["lineEdit_size_1"].setObjectName(f"Morphology_{index}_lineEdit_size_1")
        self.action_dict[f"Morphology_{index}"][1]["lineEdit_size_1"].setText('3')
        self.action_dict[f"Morphology_{index}"][1]["horizontalLayout_sizeBoth"].addWidget(self.action_dict[f"Morphology_{index}"][1]["lineEdit_size_1"])
        self.action_dict[f"Morphology_{index}"][1]["horizontalLayout_size"].addLayout(self.action_dict[f"Morphology_{index}"][1]["horizontalLayout_sizeBoth"])
        self.action_dict[f"Morphology_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"Morphology_{index}"][1]["horizontalLayout_size"])
        self.action_dict[f"Morphology_{index}"][1]["horizontalLayout_kernel"].setContentsMargins(10, 10, 10, 10)
        self.action_dict[f"Morphology_{index}"][1]["horizontalLayout_kernel"].setSpacing(2)
        self.action_dict[f"Morphology_{index}"][1]["horizontalLayout_kernel"].setObjectName(f"Morphology_{index}_horizontalLayout_kernel")
        self.action_dict[f"Morphology_{index}"][1]["checkBox_sameValue"].setObjectName(f"Morphology_{index}_checkBox_sameValue")
        self.action_dict[f"Morphology_{index}"][1]["horizontalLayout_kernel"].addWidget(self.action_dict[f"Morphology_{index}"][1]["checkBox_sameValue"])
        self.action_dict[f"Morphology_{index}"][1]["horizontalLayout_sameValue"].setObjectName(f"Morphology_{index}_horizontalLayout_sameValue")
        self.action_dict[f"Morphology_{index}"][1]["lineEdit_sameValue"].setObjectName(f"Morphology_{index}_lineEdit_sameValue")
        self.action_dict[f"Morphology_{index}"][1]["horizontalLayout_sameValue"].addWidget(self.action_dict[f"Morphology_{index}"][1]["lineEdit_sameValue"])
        self.action_dict[f"Morphology_{index}"][1]["horizontalLayout_kernel"].addLayout(self.action_dict[f"Morphology_{index}"][1]["horizontalLayout_sameValue"])
        self.action_dict[f"Morphology_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"Morphology_{index}"][1]["horizontalLayout_kernel"])
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setMinimumSize(QtCore.QSize(240, 240))
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setMaximumSize(QtCore.QSize(240, 240))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setFont(font)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setAutoFillBackground(False)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setStyleSheet("")
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setLineWidth(1)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setMidLineWidth(1)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setAutoScroll(True)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setAutoScrollMargin(16)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.AnyKeyPressed|QtWidgets.QAbstractItemView.EditTrigger.CurrentChanged|QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked|QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setAlternatingRowColors(False)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setTextElideMode(QtCore.Qt.TextElideMode.ElideMiddle)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setShowGrid(True)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setCornerButtonEnabled(True)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setRowCount(3)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setColumnCount(3)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setObjectName(f"Morphology_{index}_tableWidget_kernel")
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].horizontalHeader().setVisible(False)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].horizontalHeader().setCascadingSectionResizes(False)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].horizontalHeader().setDefaultSectionSize(40)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].horizontalHeader().setMinimumSectionSize(40)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].horizontalHeader().setStretchLastSection(False)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].verticalHeader().setVisible(False)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].verticalHeader().setCascadingSectionResizes(False)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].verticalHeader().setDefaultSectionSize(40)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].verticalHeader().setMinimumSectionSize(40)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].verticalHeader().setSortIndicatorShown(False)
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].verticalHeader().setStretchLastSection(False)
        self.action_dict[f"Morphology_{index}"][1]["verticalLayout_all"].addWidget(self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"])
        self.Elements_scrollArea_verticalLayout_main.addWidget(self.action_dict[f"Morphology_{index}"][1]["groupBox"])
        
        self.action_dict[f"Morphology_{index}"][2]['size_0'] = self.action_dict[f"Morphology_{index}"][1]['lineEdit_size_0'].text()
        self.action_dict[f"Morphology_{index}"][2]['size_1'] = self.action_dict[f"Morphology_{index}"][1]['lineEdit_size_1'].text()
        self.add_action(index, 'Morphology')
        self.kernel(f'Morphology_{index}')
        kernel = self.action_dict[f'Morphology_{index}'][2]['kernel']
        self.action_dict[f"Morphology_{index}"].append(f'cv2.morphologyEx(image, cv2.MORPH_OPEN, {kernel}, iterations=1)')
        
        self.action_dict[f"Morphology_{index}"][1]["groupBox"].setTitle(f"Morphology: {index} - On")
        self.action_dict[f"Morphology_{index}"][1]["label_functionType"].setText("Function Type")
        self.action_dict[f"Morphology_{index}"][1]["label_size"].setText("Size")
        self.action_dict[f"Morphology_{index}"][1]["checkBox_sameValue"].setText("Same Value")
        self.action_dict[f"Morphology_{index}"][1]["tableWidget_kernel"].setSortingEnabled(False)
        
        self.action_dict[f"Morphology_{index}"][1]["comboBox_functionType"].currentIndexChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f'Morphology_{index}'][1]['lineEdit_size_0'].textChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f'Morphology_{index}'][1]['lineEdit_size_1'].textChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f'Morphology_{index}'][1]['checkBox_sameValue'].clicked.connect(lambda: self.input_parameters(index))
        self.action_dict[f'Morphology_{index}'][1]['lineEdit_sameValue'].textChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f'Morphology_{index}'][1]['tableWidget_kernel'].cellChanged.connect(lambda: self.input_parameters(index))
    

    def erode_add_action(self):        
        index = len(self.action_dict)
        exec(f"self.Erode_{index}_groupBox = QtWidgets.QGroupBox(self.Elements_scrollAreaWidgetContents)")
        exec(f"self.Erode_{index}_layoutWidget = QtWidgets.QWidget(self.Erode_{index}_groupBox)")
        exec(f"self.Erode_{index}_verticalLayout_all = QtWidgets.QVBoxLayout(self.Erode_{index}_layoutWidget)")
        exec(f"self.Erode_{index}_horizontalLayout_size = QtWidgets.QHBoxLayout()")
        exec(f"self.Erode_{index}_label_Size = QtWidgets.QLabel(self.Erode_{index}_layoutWidget)")
        exec(f"self.Erode_{index}_horizontalLayout_sizeBoth = QtWidgets.QHBoxLayout()")
        exec(f"self.Erode_{index}_lineEdit_size_0 = QtWidgets.QLineEdit(self.Erode_{index}_layoutWidget)")
        exec(f"self.Erode_{index}_lineEdit_size_1 = QtWidgets.QLineEdit(self.Erode_{index}_layoutWidget)")
        exec(f"self.Erode_{index}_horizontalLayout_kernel = QtWidgets.QHBoxLayout()")
        exec(f"self.Erode_{index}_checkBox_sameValue = QtWidgets.QCheckBox(self.Erode_{index}_layoutWidget)")
        exec(f"self.Erode_{index}_horizontalLayout_sameValueEdit = QtWidgets.QHBoxLayout()")
        exec(f"self.Erode_{index}_lineEdit_sameValue = QtWidgets.QLineEdit(self.Erode_{index}_layoutWidget)")
        exec(f"self.Erode_{index}_tableWidget_kernel = QtWidgets.QTableWidget(self.Erode_{index}_layoutWidget)")

        self.action_dict.update({f'Erode_{index}': 
            [index,
            {
                'groupBox': eval(f"self.Erode_{index}_groupBox"),
                'layoutWidget': eval(f"self.Erode_{index}_layoutWidget"),
                'verticalLayout_all': eval(f"self.Erode_{index}_verticalLayout_all"),
                'horizontalLayout_size': eval(f"self.Erode_{index}_horizontalLayout_size"),
                'label_Size': eval(f"self.Erode_{index}_label_Size"),
                'horizontalLayout_sizeBoth': eval(f"self.Erode_{index}_horizontalLayout_sizeBoth"),
                'lineEdit_size_0': eval(f"self.Erode_{index}_lineEdit_size_0"),
                'lineEdit_size_1': eval(f"self.Erode_{index}_lineEdit_size_1"),
                'horizontalLayout_kernel': eval(f"self.Erode_{index}_horizontalLayout_kernel"),
                'checkBox_sameValue': eval(f"self.Erode_{index}_checkBox_sameValue"),
                'horizontalLayout_sameValueEdit': eval(f"self.Erode_{index}_horizontalLayout_sameValueEdit"),
                'lineEdit_sameValue': eval(f"self.Erode_{index}_lineEdit_sameValue"),
                'tableWidget_kernel': eval(f"self.Erode_{index}_tableWidget_kernel")
            },
            {
                'size_0': None,
                'size_1': None,
                'sameValue': None,
                'kernel': None                
            }
            ]})
        
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.action_dict[f"Erode_{index}"][1]["groupBox"].sizePolicy().hasHeightForWidth())
        self.action_dict[f"Erode_{index}"][1]["groupBox"].setSizePolicy(sizePolicy)
        self.action_dict[f"Erode_{index}"][1]["groupBox"].setMinimumSize(QtCore.QSize(260, 370))
        self.action_dict[f"Erode_{index}"][1]["groupBox"].setMaximumSize(QtCore.QSize(260, 370))
        self.action_dict[f"Erode_{index}"][1]["groupBox"].setCheckable(True)
        self.action_dict[f"Erode_{index}"][1]["groupBox"].setObjectName(f"Erode_{index}_groupBox")
        self.action_dict[f"Erode_{index}"][1]["layoutWidget"].setGeometry(QtCore.QRect(10, 20, 242, 341))
        self.action_dict[f"Erode_{index}"][1]["layoutWidget"].setObjectName("layoutWidget_9")
        self.action_dict[f"Erode_{index}"][1]["verticalLayout_all"].setContentsMargins(0, 0, 0, 0)
        self.action_dict[f"Erode_{index}"][1]["verticalLayout_all"].setObjectName(f"Erode_{index}_verticalLayout_all")
        self.action_dict[f"Erode_{index}"][1]["horizontalLayout_size"].setContentsMargins(10, 10, 10, 10)
        self.action_dict[f"Erode_{index}"][1]["horizontalLayout_size"].setSpacing(2)
        self.action_dict[f"Erode_{index}"][1]["horizontalLayout_size"].setObjectName(f"Erode_{index}_horizontalLayout_size")
        self.action_dict[f"Erode_{index}"][1]["label_Size"].setMinimumSize(QtCore.QSize(50, 0))
        self.action_dict[f"Erode_{index}"][1]["label_Size"].setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f"Erode_{index}"][1]["label_Size"].setObjectName(f"Erode_{index}_label_Size")
        self.action_dict[f"Erode_{index}"][1]["horizontalLayout_size"].addWidget(self.action_dict[f"Erode_{index}"][1]["label_Size"])
        self.action_dict[f"Erode_{index}"][1]["horizontalLayout_sizeBoth"].setObjectName(f"Erode_{index}_horizontalLayout_sizeBoth")
        self.action_dict[f"Erode_{index}"][1]["lineEdit_size_0"].setObjectName(f"Erode_{index}_lineEdit_size_0")
        self.action_dict[f"Erode_{index}"][1]["lineEdit_size_0"].setText('3')
        self.action_dict[f"Erode_{index}"][1]["horizontalLayout_sizeBoth"].addWidget(self.action_dict[f"Erode_{index}"][1]["lineEdit_size_0"])
        self.action_dict[f"Erode_{index}"][1]["lineEdit_size_1"].setObjectName(f"Erode_{index}_lineEdit_size_1")
        self.action_dict[f"Erode_{index}"][1]["lineEdit_size_1"].setText('3')
        self.action_dict[f"Erode_{index}"][1]["horizontalLayout_sizeBoth"].addWidget(self.action_dict[f"Erode_{index}"][1]["lineEdit_size_1"])
        self.action_dict[f"Erode_{index}"][1]["horizontalLayout_size"].addLayout(self.action_dict[f"Erode_{index}"][1]["horizontalLayout_sizeBoth"])
        self.action_dict[f"Erode_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"Erode_{index}"][1]["horizontalLayout_size"])
        self.action_dict[f"Erode_{index}"][1]["horizontalLayout_kernel"].setContentsMargins(10, 10, 10, 10)
        self.action_dict[f"Erode_{index}"][1]["horizontalLayout_kernel"].setSpacing(2)
        self.action_dict[f"Erode_{index}"][1]["horizontalLayout_kernel"].setObjectName(f"Erode_{index}_horizontalLayout_kernel")
        self.action_dict[f"Erode_{index}"][1]["checkBox_sameValue"].setObjectName(f"Erode_{index}_checkBox_sameValue")
        self.action_dict[f"Erode_{index}"][1]["horizontalLayout_kernel"].addWidget(self.action_dict[f"Erode_{index}"][1]["checkBox_sameValue"])
        self.action_dict[f"Erode_{index}"][1]["horizontalLayout_sameValueEdit"].setObjectName(f"Erode_{index}_horizontalLayout_sameValueEdit")
        self.action_dict[f"Erode_{index}"][1]["lineEdit_sameValue"].setObjectName(f"Erode_{index}_lineEdit_sameValue")
        self.action_dict[f"Erode_{index}"][1]["horizontalLayout_sameValueEdit"].addWidget(self.action_dict[f"Erode_{index}"][1]["lineEdit_sameValue"])
        self.action_dict[f"Erode_{index}"][1]["horizontalLayout_kernel"].addLayout(self.action_dict[f"Erode_{index}"][1]["horizontalLayout_sameValueEdit"])
        self.action_dict[f"Erode_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"Erode_{index}"][1]["horizontalLayout_kernel"])
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setMinimumSize(QtCore.QSize(240, 240))
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setMaximumSize(QtCore.QSize(240, 240))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setFont(font)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setAutoFillBackground(False)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setStyleSheet("")
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setLineWidth(1)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setMidLineWidth(1)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setAutoScroll(True)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setAutoScrollMargin(16)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.AnyKeyPressed|QtWidgets.QAbstractItemView.EditTrigger.CurrentChanged|QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked|QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setAlternatingRowColors(False)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setTextElideMode(QtCore.Qt.TextElideMode.ElideMiddle)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setShowGrid(True)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setCornerButtonEnabled(True)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setRowCount(3)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setColumnCount(3)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setObjectName(f"Erode_{index}_tableWidget_kernel")
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].horizontalHeader().setVisible(False)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].horizontalHeader().setCascadingSectionResizes(False)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].horizontalHeader().setDefaultSectionSize(40)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].horizontalHeader().setMinimumSectionSize(40)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].horizontalHeader().setStretchLastSection(False)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].verticalHeader().setVisible(False)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].verticalHeader().setCascadingSectionResizes(False)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].verticalHeader().setDefaultSectionSize(40)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].verticalHeader().setMinimumSectionSize(40)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].verticalHeader().setSortIndicatorShown(False)
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].verticalHeader().setStretchLastSection(False)
        self.action_dict[f"Erode_{index}"][1]["verticalLayout_all"].addWidget(self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"])
        self.Elements_scrollArea_verticalLayout_main.addWidget(self.action_dict[f"Erode_{index}"][1]["groupBox"])
        
        self.action_dict[f"Erode_{index}"][2]['size_0'] = self.action_dict[f"Erode_{index}"][1]['lineEdit_size_0'].text()
        self.action_dict[f"Erode_{index}"][2]['size_1'] = self.action_dict[f"Erode_{index}"][1]['lineEdit_size_1'].text()
        self.add_action(index, 'Erode')
        self.kernel(f'Erode_{index}')
        kernel = self.action_dict[f'Erode_{index}'][2]['kernel']
        self.action_dict[f"Erode_{index}"].append(f'cv2.erode(image, {kernel})')
        
        self.action_dict[f"Erode_{index}"][1]["groupBox"].setTitle(f"Erode: {index} - On")
        self.action_dict[f"Erode_{index}"][1]["label_Size"].setText("Size")
        self.action_dict[f"Erode_{index}"][1]["checkBox_sameValue"].setText("Same Value")
        self.action_dict[f"Erode_{index}"][1]["tableWidget_kernel"].setSortingEnabled(False)
        
        self.action_dict[f'Erode_{index}'][1]['lineEdit_size_0'].textChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f'Erode_{index}'][1]['lineEdit_size_1'].textChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f'Erode_{index}'][1]['checkBox_sameValue'].clicked.connect(lambda: self.input_parameters(index))
        self.action_dict[f'Erode_{index}'][1]['lineEdit_sameValue'].textChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f'Erode_{index}'][1]['tableWidget_kernel'].cellChanged.connect(lambda: self.input_parameters(index))
        

    def dilate_add_action(self):        
        index = len(self.action_dict)
        exec(f"self.Dilate_{index}_groupBox = QtWidgets.QGroupBox(self.Elements_scrollAreaWidgetContents)")
        exec(f"self.Dilate_{index}_layoutWidget = QtWidgets.QWidget(self.Dilate_{index}_groupBox)")
        exec(f"self.Dilate_{index}_verticalLayout_all = QtWidgets.QVBoxLayout(self.Dilate_{index}_layoutWidget)")
        exec(f"self.Dilate_{index}_horizontalLayout_size = QtWidgets.QHBoxLayout()")
        exec(f"self.Dilate_{index}_label_Size = QtWidgets.QLabel(self.Dilate_{index}_layoutWidget)")
        exec(f"self.Dilate_{index}_horizontalLayout_sizeBoth = QtWidgets.QHBoxLayout()")
        exec(f"self.Dilate_{index}_lineEdit_size_0 = QtWidgets.QLineEdit(self.Dilate_{index}_layoutWidget)")
        exec(f"self.Dilate_{index}_lineEdit_size_1 = QtWidgets.QLineEdit(self.Dilate_{index}_layoutWidget)")
        exec(f"self.Dilate_{index}_horizontalLayout_kernel = QtWidgets.QHBoxLayout()")
        exec(f"self.Dilate_{index}_checkBox_sameValue = QtWidgets.QCheckBox(self.Dilate_{index}_layoutWidget)")
        exec(f"self.Dilate_{index}_horizontalLayout_sameValueEdit = QtWidgets.QHBoxLayout()")
        exec(f"self.Dilate_{index}_lineEdit_sameValue = QtWidgets.QLineEdit(self.Dilate_{index}_layoutWidget)")
        exec(f"self.Dilate_{index}_tableWidget_kernel = QtWidgets.QTableWidget(self.Dilate_{index}_layoutWidget)")

        self.action_dict.update({f'Dilate_{index}': 
            [index,
            {
                'groupBox': eval(f"self.Dilate_{index}_groupBox"),
                'layoutWidget': eval(f"self.Dilate_{index}_layoutWidget"),
                'verticalLayout_all': eval(f"self.Dilate_{index}_verticalLayout_all"),
                'horizontalLayout_size': eval(f"self.Dilate_{index}_horizontalLayout_size"),
                'label_Size': eval(f"self.Dilate_{index}_label_Size"),
                'horizontalLayout_sizeBoth': eval(f"self.Dilate_{index}_horizontalLayout_sizeBoth"),
                'lineEdit_size_0': eval(f"self.Dilate_{index}_lineEdit_size_0"),
                'lineEdit_size_1': eval(f"self.Dilate_{index}_lineEdit_size_1"),
                'horizontalLayout_kernel': eval(f"self.Dilate_{index}_horizontalLayout_kernel"),
                'checkBox_sameValue': eval(f"self.Dilate_{index}_checkBox_sameValue"),
                'horizontalLayout_sameValueEdit': eval(f"self.Dilate_{index}_horizontalLayout_sameValueEdit"),
                'lineEdit_sameValue': eval(f"self.Dilate_{index}_lineEdit_sameValue"),
                'tableWidget_kernel': eval(f"self.Dilate_{index}_tableWidget_kernel")
            },
            {
                'size_0': None,
                'size_1': None,
                'sameValue': None,
                'kernel': None      
            }]})
        
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.action_dict[f"Dilate_{index}"][1]["groupBox"].sizePolicy().hasHeightForWidth())
        self.action_dict[f"Dilate_{index}"][1]["groupBox"].setSizePolicy(sizePolicy)
        self.action_dict[f"Dilate_{index}"][1]["groupBox"].setMinimumSize(QtCore.QSize(260, 370))
        self.action_dict[f"Dilate_{index}"][1]["groupBox"].setMaximumSize(QtCore.QSize(260, 370))
        self.action_dict[f"Dilate_{index}"][1]["groupBox"].setCheckable(True)
        self.action_dict[f"Dilate_{index}"][1]["groupBox"].setObjectName(f"Dilate_{index}_groupBox")
        self.action_dict[f"Dilate_{index}"][1]["layoutWidget"].setGeometry(QtCore.QRect(10, 20, 242, 341))
        self.action_dict[f"Dilate_{index}"][1]["layoutWidget"].setObjectName("layoutWidget_10")
        self.action_dict[f"Dilate_{index}"][1]["verticalLayout_all"].setContentsMargins(0, 0, 0, 0)
        self.action_dict[f"Dilate_{index}"][1]["verticalLayout_all"].setObjectName(f"Dilate_{index}_verticalLayout_all")
        self.action_dict[f"Dilate_{index}"][1]["horizontalLayout_size"].setContentsMargins(10, 10, 10, 10)
        self.action_dict[f"Dilate_{index}"][1]["horizontalLayout_size"].setSpacing(2)
        self.action_dict[f"Dilate_{index}"][1]["horizontalLayout_size"].setObjectName(f"Dilate_{index}_horizontalLayout_size")
        self.action_dict[f"Dilate_{index}"][1]["label_Size"].setMinimumSize(QtCore.QSize(50, 0))
        self.action_dict[f"Dilate_{index}"][1]["label_Size"].setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f"Dilate_{index}"][1]["label_Size"].setObjectName(f"Dilate_{index}_label_Size")
        self.action_dict[f"Dilate_{index}"][1]["horizontalLayout_size"].addWidget(self.action_dict[f"Dilate_{index}"][1]["label_Size"])
        self.action_dict[f"Dilate_{index}"][1]["horizontalLayout_sizeBoth"].setObjectName(f"Dilate_{index}_horizontalLayout_sizeBoth")
        self.action_dict[f"Dilate_{index}"][1]["lineEdit_size_0"].setObjectName(f"Dilate_{index}_lineEdit_size_0")
        self.action_dict[f"Dilate_{index}"][1]["lineEdit_size_0"].setText('3')
        self.action_dict[f"Dilate_{index}"][1]["horizontalLayout_sizeBoth"].addWidget(self.action_dict[f"Dilate_{index}"][1]["lineEdit_size_0"])
        self.action_dict[f"Dilate_{index}"][1]["lineEdit_size_1"].setObjectName(f"Dilate_{index}_lineEdit_size_1")
        self.action_dict[f"Dilate_{index}"][1]["lineEdit_size_1"].setText('3')
        self.action_dict[f"Dilate_{index}"][1]["horizontalLayout_sizeBoth"].addWidget(self.action_dict[f"Dilate_{index}"][1]["lineEdit_size_1"])
        self.action_dict[f"Dilate_{index}"][1]["horizontalLayout_size"].addLayout(self.action_dict[f"Dilate_{index}"][1]["horizontalLayout_sizeBoth"])
        self.action_dict[f"Dilate_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"Dilate_{index}"][1]["horizontalLayout_size"])
        self.action_dict[f"Dilate_{index}"][1]["horizontalLayout_kernel"].setContentsMargins(10, 10, 10, 10)
        self.action_dict[f"Dilate_{index}"][1]["horizontalLayout_kernel"].setSpacing(2)
        self.action_dict[f"Dilate_{index}"][1]["horizontalLayout_kernel"].setObjectName(f"Dilate_{index}_horizontalLayout_kernel")
        self.action_dict[f"Dilate_{index}"][1]["checkBox_sameValue"].setObjectName(f"Dilate_{index}_checkBox_sameValue")
        self.action_dict[f"Dilate_{index}"][1]["horizontalLayout_kernel"].addWidget(self.action_dict[f"Dilate_{index}"][1]["checkBox_sameValue"])
        self.action_dict[f"Dilate_{index}"][1]["horizontalLayout_sameValueEdit"].setObjectName(f"Dilate_{index}_horizontalLayout_sameValueEdit")
        self.action_dict[f"Dilate_{index}"][1]["lineEdit_sameValue"].setObjectName(f"Dilate_{index}_lineEdit_sameValue")
        self.action_dict[f"Dilate_{index}"][1]["horizontalLayout_sameValueEdit"].addWidget(self.action_dict[f"Dilate_{index}"][1]["lineEdit_sameValue"])
        self.action_dict[f"Dilate_{index}"][1]["horizontalLayout_kernel"].addLayout(self.action_dict[f"Dilate_{index}"][1]["horizontalLayout_sameValueEdit"])
        self.action_dict[f"Dilate_{index}"][1]["verticalLayout_all"].addLayout(self.action_dict[f"Dilate_{index}"][1]["horizontalLayout_kernel"])
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setMinimumSize(QtCore.QSize(240, 240))
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setMaximumSize(QtCore.QSize(240, 240))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setFont(font)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setAutoFillBackground(False)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setStyleSheet("")
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setLineWidth(1)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setMidLineWidth(1)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setAutoScroll(True)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setAutoScrollMargin(16)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.AnyKeyPressed|QtWidgets.QAbstractItemView.EditTrigger.CurrentChanged|QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked|QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setAlternatingRowColors(False)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setTextElideMode(QtCore.Qt.TextElideMode.ElideMiddle)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setShowGrid(True)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setCornerButtonEnabled(True)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setRowCount(3)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setColumnCount(3)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setObjectName(f"Dilate_{index}_tableWidget_kernel")
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].horizontalHeader().setVisible(False)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].horizontalHeader().setCascadingSectionResizes(False)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].horizontalHeader().setDefaultSectionSize(40)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].horizontalHeader().setMinimumSectionSize(40)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].horizontalHeader().setStretchLastSection(False)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].verticalHeader().setVisible(False)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].verticalHeader().setCascadingSectionResizes(False)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].verticalHeader().setDefaultSectionSize(40)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].verticalHeader().setMinimumSectionSize(40)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].verticalHeader().setSortIndicatorShown(False)
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].verticalHeader().setStretchLastSection(False)
        self.action_dict[f"Dilate_{index}"][1]["verticalLayout_all"].addWidget(self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"])
        self.Elements_scrollArea_verticalLayout_main.addWidget(self.action_dict[f"Dilate_{index}"][1]["groupBox"])
        
        self.action_dict[f"Dilate_{index}"][2]['size_0'] = self.action_dict[f"Dilate_{index}"][1]['lineEdit_size_0'].text()
        self.action_dict[f"Dilate_{index}"][2]['size_1'] = self.action_dict[f"Dilate_{index}"][1]['lineEdit_size_1'].text()
        self.add_action(index, 'Dilate')
        self.kernel(f'Dilate_{index}')
        kernel = self.action_dict[f'Dilate_{index}'][2]['kernel']
        self.action_dict[f"Dilate_{index}"].append(f'cv2.dilate(image, {kernel})')
        
        self.action_dict[f"Dilate_{index}"][1]["groupBox"].setTitle(f"Dilate: {index} - On")
        self.action_dict[f"Dilate_{index}"][1]["label_Size"].setText("Size")
        self.action_dict[f"Dilate_{index}"][1]["checkBox_sameValue"].setText("Same Value")
        self.action_dict[f"Dilate_{index}"][1]["tableWidget_kernel"].setSortingEnabled(False)
        
        self.action_dict[f'Dilate_{index}'][1]['lineEdit_size_0'].textChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f'Dilate_{index}'][1]['lineEdit_size_1'].textChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f'Dilate_{index}'][1]['checkBox_sameValue'].clicked.connect(lambda: self.input_parameters(index))
        self.action_dict[f'Dilate_{index}'][1]['lineEdit_sameValue'].textChanged.connect(lambda: self.input_parameters(index))
        self.action_dict[f'Dilate_{index}'][1]['tableWidget_kernel'].cellChanged.connect(lambda: self.input_parameters(index))

    
    def cvtColor_change_values(self, name, filter, index):
        parse_filter = filter.split(".", 1)[1].split("(")[1].replace(")", "").split(", ")[1].split("_")[1]
        # print(index)
        # print(parse_filter)
        self.input_parameters(index, name, parse_filter)


    def gaussianBlur_change_values(self, name, filter, index):
        parse_filter = filter.split(".", 1)[1].split("(", 1)[1].replace("(", '').replace(")", '').split(", ")[1:]
        # print(index)
        # print(parse_filter)
        self.input_parameters(index, name, parse_filter)


    def medianBlur_change_values(self, name, filter, index):
        parse_filter = filter.split(".", 1)[1].split("(")[1].replace(")", '').split(", ")[1:]
        # print(index)
        # print(parse_filter)
        self.input_parameters(index, name, parse_filter)


    def threshold_change_values(self, name, filter, index):
        parse_filter = filter.split(".", 1)[1].split("(")[1].replace(")[1]", '').split(", ")[1:]
        # print(index)
        # print(parse_filter)
        self.input_parameters(index, name, parse_filter)


    def adaptiveThreshold_change_values(self, name, filter, index):
        parse_filter = filter.split(".", 1)[1].split("(")[1].replace(")", '').split(", ")[1:]
        # print(index)
        # print(parse_filter)
        self.input_parameters(index, name, parse_filter)


    def filter2D_change_values(self, name, filter, index):
        parse_filter = filter.split(".", 1)[1].split("(", 1)[1].split(", ", 2)[2].replace(")", '').replace("np.array(", '')
        # print(index)
        # print(parse_filter)
        self.input_parameters(index, name, parse_filter)


    def bilateralFilter_change_values(self, name, filter, index):
        parse_filter = filter.split(".", 1)[1].split("(")[1].replace(")", '').split(", ")[1:]
        print(index)
        print(parse_filter)
        self.input_parameters(index, name, parse_filter)


    def createCLAHE_change_values(self, name, filter, index):
        parse_filter = filter.split(".", 1)[1].split("(", 1)[1].replace("clipLimit=", '').replace("tileGridSize=", '').replace(")", '').replace("(", '').split(", ")
        print(index)
        print(parse_filter)
        self.input_parameters(index, name, parse_filter)


    def morphologyEx_change_values(self, name, filter, index):
        if "uint8" in filter:
            parse_filter = filter.split(".", 1)[1].split("(", 1)[1].replace(', "uint8"), iterations=1)', '').replace("np.array(", '').split(", ", 2)[1:]
        else:
            parse_filter = filter.split(".", 1)[1].split("(", 1)[1].replace('), iterations=1)', '').replace("np.array(", '').split(", ", 2)[1:]
        # print(index)
        # print(parse_filter)
        self.input_parameters(index, name, parse_filter)


    def erode_change_values(self, name, filter, index):
        parse_filter = filter.split(".", 1)[1].split("(", 1)[1].replace("np.array(", '').replace(")", '').split(", ", 1)[1:]
        self.input_parameters(index, name, parse_filter)


    def dilate_change_values(self, name, filter, index):
        parse_filter = filter.split(".", 1)[1].split("(", 1)[1].replace("np.array(", '').replace(")", '').split(", ", 1)[1:]
        self.input_parameters(index, name, parse_filter)


    def cvtColor_loadHelper(self, index, parameters):
        CurrentIndex = self.action_dict[f"cvtColor_{index}"][1]["comboBox_colorSpace"].findText(parameters)
        self.action_dict[f"cvtColor_{index}"][1]["comboBox_colorSpace"].setCurrentIndex(CurrentIndex)


    def GaussianBlur_loadHelper(self, index, parameters):
        self.action_dict[f'GaussianBlur_{index}'][1]['lineEdit_size_0'].setText(parameters[0])
        self.action_dict[f'GaussianBlur_{index}'][1]['lineEdit_size_1'].setText(parameters[1])
        CurrentIndex = self.action_dict[f'GaussianBlur_{index}'][1]["comboBox_borderType"].findText(parameters[2].split('.')[1])
        self.action_dict[f'GaussianBlur_{index}'][1]['comboBox_borderType'].setCurrentIndex(CurrentIndex)


    def medianBlur_loadHelper(self, index, parameters):
        self.action_dict[f'MedianBlur_{index}'][1]['lineEdit_size_0'].setText(parameters[0])
        self.action_dict[f'MedianBlur_{index}'][1]['lineEdit_size_1'].setText(parameters[0])


    def threshold_loadHelper(self, index, parameters):
        self.action_dict[f'Threshold_{index}'][1]['horizontalSlider_minValue'].setValue(int(parameters[0]))
        self.action_dict[f'Threshold_{index}'][1]['spinBox_minValue'].setValue(int(parameters[0]))
        self.action_dict[f'Threshold_{index}'][1]['horizontalSlider_maxValue'].setValue(int(parameters[1]))
        self.action_dict[f'Threshold_{index}'][1]['spinBox_maxValue'].setValue(int(parameters[1]))
        CurrentIndex = self.action_dict[f"Threshold_{index}"][1]["comboBox_thresholdType"].findText(parameters[2].split('.')[1])
        self.action_dict[f'Threshold_{index}'][1]['comboBox_thresholdType'].setCurrentIndex(CurrentIndex)

    
    def adaptiveThreshold_loadHelper(self, index, parameters):
        self.action_dict[f'AdaptiveThreshold_{index}'][1]['horizontalSlider_maxValue'].setValue(int(parameters[0]))
        self.action_dict[f'AdaptiveThreshold_{index}'][1]['spinBox_maxValue'].setValue(int(parameters[0]))
        CurrentIndex = self.action_dict[f"AdaptiveThreshold_{index}"][1]["comboBox_adaptiveThresholdType"].findText(parameters[1].split('.')[1])
        self.action_dict[f'AdaptiveThreshold_{index}'][1]['comboBox_adaptiveThresholdType'].setCurrentIndex(CurrentIndex)
        CurrentIndex = self.action_dict[f"AdaptiveThreshold_{index}"][1]["comboBox_thresholdType"].findText(parameters[2].split('.')[1])
        self.action_dict[f'AdaptiveThreshold_{index}'][1]['comboBox_thresholdType'].setCurrentIndex(CurrentIndex)
        self.action_dict[f'AdaptiveThreshold_{index}'][1]['horizontalSlider_blockSize'].setValue(int(parameters[3]))
        self.action_dict[f'AdaptiveThreshold_{index}'][1]['spinBox_blockSize'].setValue(int(parameters[3]))
        self.action_dict[f'AdaptiveThreshold_{index}'][1]['horizontalSlider_cConstantValue'].setValue(int(parameters[4]))
        self.action_dict[f'AdaptiveThreshold_{index}'][1]['spinBox_cConstantValue'].setValue(int(parameters[4]))
        

    def table_loadHelper(self, name, index, a):
        self.action_dict[f'{name}_{index}'][1]['tableWidget_kernel'].clear()
        b = a[:]
        same = []
        for i in range(len(a)):
            b[i] = set(a[i])   
        if len(a) > 1:     
            for i in range(len(b)-1):
                if b[i] != b[i+1]:
                    same.append(False)
                    continue
                same.append(True)
        elif len(b[0]) == 1:
            same.append(True)
        else:
            same.append(False)
        self.action_dict[f"{name}_{index}"][1]["lineEdit_size_0"].setText(f'{len(a)}')
        self.action_dict[f"{name}_{index}"][1]["lineEdit_size_1"].setText(f'{len(a[0])}')
        if False in same:
            self.action_dict[f'{name}_{index}'][1]['tableWidget_kernel'].setDisabled(False)
            self.action_dict[f'{name}_{index}'][1]['tableWidget_kernel'].setRowCount(len(a))
            self.action_dict[f'{name}_{index}'][1]['tableWidget_kernel'].setColumnCount(len(a[0]))
            for i in range(len(a)): 
                for j in range(len(a[i])): 
                    self.action_dict[f'{name}_{index}'][1]['tableWidget_kernel'].setItem(i, j, QtWidgets.QTableWidgetItem(f"{a[i][j]}"))
        else:
            self.action_dict[f'{name}_{index}'][1]['tableWidget_kernel'].setDisabled(True)
            self.action_dict[f'{name}_{index}'][1]['checkBox_sameValue'].setChecked(True)
            self.action_dict[f'{name}_{index}'][1]['lineEdit_sameValue'].setText(f'{a[0][0]}')
    
    
    def filter2D_loadHelper(self, index, parameters):
        self.table_loadHelper(name = "Filter2D",index = index, a = eval(parameters))
       

    def bilateralFilter_loadHelper(self, index, parameters):
        self.action_dict[f'BilateralFilter_{index}'][1]['horizontalSlider_dSize'].setValue(int(parameters[0]))
        self.action_dict[f'BilateralFilter_{index}'][1]['spinBox_dSize'].setValue(int(parameters[0]))
        self.action_dict[f'BilateralFilter_{index}'][1]['horizontalSlider_sigmaFirst'].setValue(int(parameters[1]))
        self.action_dict[f'BilateralFilter_{index}'][1]['spinBox_sigmaFirst'].setValue(int(parameters[1]))
        self.action_dict[f'BilateralFilter_{index}'][1]['horizontalSlider_sigmaSecond'].setValue(int(parameters[2]))
        self.action_dict[f'BilateralFilter_{index}'][1]['spinBox_sigmaSecond'].setValue(int(parameters[2]))


    def createCLAHE_loadHelper(self, index, parameters):
        self.action_dict[f'CLAHEFilter_{index}'][1]['horizontalSlider_clipLimit'].setValue(int(float(parameters[0])*10))
        self.action_dict[f'CLAHEFilter_{index}'][1]['doubleSpinBox_clipLimit'].setValue(float(parameters[0]))
        self.action_dict[f'CLAHEFilter_{index}'][1]['lineEdit_size_0'].setText(parameters[1])
        self.action_dict[f'CLAHEFilter_{index}'][1]['lineEdit_size_1'].setText(parameters[2])
        
        
    def morphologyEx_loadHelper(self, index, parameters):
        CurrentIndex = self.action_dict[f"Morphology_{index}"][1]["comboBox_functionType"].findText(parameters[0].split(".")[1])
        self.action_dict[f"Morphology_{index}"][1]["comboBox_functionType"].setCurrentIndex(CurrentIndex)
        self.table_loadHelper(name = "Morphology",index = index, a = eval(parameters[1]))
        
        
    def erode_loadHelper(self, index, parameters):
        self.table_loadHelper(name = "Erode",index = index, a = eval(parameters[0]))
        
        
    def dilate_loadHelper(self, index, parameters):
        self.table_loadHelper(name = "Dilate",index = index, a = eval(parameters[0]))


    def input_parameters(self, index, name = '', parameters = None):        
        loadingParameters = {
            "cvtColor":             [self.cvtColor_loadHelper], 
            "GaussianBlur":         [self.GaussianBlur_loadHelper],
            "medianBlur":           [self.medianBlur_loadHelper],
            "threshold":            [self.threshold_loadHelper],
            "adaptiveThreshold":    [self.adaptiveThreshold_loadHelper],
            "filter2D":             [self.filter2D_loadHelper],
            "bilateralFilter":      [self.bilateralFilter_loadHelper],
            "createCLAHE":          [self.createCLAHE_loadHelper],
            "morphologyEx":         [self.morphologyEx_loadHelper],
            "erode":                [self.erode_loadHelper],
            "dilate":               [self.dilate_loadHelper]
        }
        
        try:
            sender = self.MainWindow.sender()
        except Exception as e:
            print(f"Sender doesn't exist {e}")
        
        if not name == '':
            objectName = name
            objName = f"{name}_{index}"            
            print(loadingParameters[name][0].__call__(index, parameters))
        else:
            objectName = sender.objectName().split('_' , 1)[0]
            objName = f"{sender.objectName().split('_' , 1)[0]}_{index}"
        
        if objectName == 'cvtColor':
            
            colorSpace = self.action_dict[objName][1]['comboBox_colorSpace'].currentText()
            self.action_dict[objName][2]['colorSpace'] = colorSpace
            self.action_dict[objName][4] = f'cv2.cvtColor(image, cv2.COLOR_{colorSpace})'
            # self.action_dict[objName][4] = f'cv2.resize(image, (200, 200))'    
            
        elif objectName == 'GaussianBlur':
            
            size_0 = self.action_dict[objName][1]['lineEdit_size_0'].text()
            size_1 = self.action_dict[objName][1]['lineEdit_size_1'].text()
            borderType = self.action_dict[objName][1]['comboBox_borderType'].currentText()
            self.action_dict[objName][2]['size_0'] = size_0   
            self.action_dict[objName][2]['size_1'] = size_1   
            self.action_dict[objName][2]['borderType'] = borderType
            self.action_dict[objName][4] = f'cv2.GaussianBlur(image, ({size_0}, {size_1}), cv2.{borderType})'
            
        elif objectName == 'MedianBlur':
            
            size_0 = self.action_dict[objName][1]['lineEdit_size_0'].text()
            size_1 = self.action_dict[objName][1]['lineEdit_size_1'].text()
            self.action_dict[objName][2]['size_0'] = size_0
            self.action_dict[objName][2]['size_1'] = size_1
            self.action_dict[objName][4] = f'cv2.medianBlur(image, {size_0})'
        
        elif objectName == 'Threshold':
            
            minValue = self.action_dict[objName][1]['horizontalSlider_minValue'].value()
            maxValue = self.action_dict[objName][1]['horizontalSlider_maxValue'].value()
            thresholdType = self.action_dict[objName][1]['comboBox_thresholdType'].currentText()
            self.action_dict[objName][2]['minValue'] = minValue     
            self.action_dict[objName][2]['maxValue'] = maxValue     
            self.action_dict[objName][2]['thresholdType'] = thresholdType
            self.action_dict[objName][4] = f'cv2.threshold(image, {minValue}, {maxValue}, cv2.{thresholdType})[1]'
        
        elif objectName == 'AdaptiveThreshold':
            
            maxValue = self.action_dict[objName][1]['horizontalSlider_maxValue'].value()
            adaptiveThresholdType = self.action_dict[objName][1]['comboBox_adaptiveThresholdType'].currentText()
            thresholdType = self.action_dict[objName][1]['comboBox_thresholdType'].currentText()
            blockSize = self.action_dict[objName][1]['horizontalSlider_blockSize'].value()
            cConstantValue = self.action_dict[objName][1]['horizontalSlider_cConstantValue'].value()
            self.action_dict[objName][2]['maxValue'] =              maxValue
            self.action_dict[objName][2]['adaptiveThresholdType'] = adaptiveThresholdType
            self.action_dict[objName][2]['thresholdType'] =         thresholdType
            self.action_dict[objName][2]['blockSize'] =             blockSize
            self.action_dict[objName][2]['cConstantValue'] =        cConstantValue
            self.action_dict[objName][4] = f'cv2.adaptiveThreshold(image, {maxValue}, cv2.{adaptiveThresholdType}, cv2.{thresholdType}, {blockSize}, {cConstantValue})'
        
        elif objectName == 'Filter2D':
            
            size_0 = self.action_dict[objName][1]['lineEdit_size_0'].text()
            size_1 = self.action_dict[objName][1]['lineEdit_size_1'].text()
            sameValue = self.action_dict[objName][1]['lineEdit_sameValue'].text()
            self.action_dict[objName][2]['size_0'] = size_0
            self.action_dict[objName][2]['size_1'] = size_1
            if size_0 != '' and size_1 != '':
                self.action_dict[objName][1]['tableWidget_kernel'].setRowCount(int(size_0))
                self.action_dict[objName][1]['tableWidget_kernel'].setColumnCount(int(size_1))
            self.action_dict[objName][2]['sameValue'] = sameValue
            self.kernel(objName)
            kernel = self.action_dict[objName][2]['kernel']
            self.action_dict[objName][4] = f'cv2.filter2D(image, -1, np.array({kernel}))'
        
        elif objectName == 'BilateralFilter':
            
            dSize = self.action_dict[objName][1]['horizontalSlider_dSize'].value()
            sigmaFirst = self.action_dict[objName][1]['horizontalSlider_sigmaFirst'].value()
            sigmaSecond = self.action_dict[objName][1]['horizontalSlider_sigmaSecond'].value()
            self.action_dict[objName][2]['dSize'] =         dSize
            self.action_dict[objName][2]['sigmaFirst'] =    sigmaFirst
            self.action_dict[objName][2]['sigmaSecond'] =   sigmaSecond
            self.action_dict[objName][4] = f'cv2.bilateralFilter(image, {dSize}, {sigmaFirst}, {sigmaSecond})'
        
        elif objectName == 'CLAHEFilter':   
            
            clipLimit = self.action_dict[objName][1]['horizontalSlider_clipLimit'].value()/10
            size_0 = self.action_dict[objName][1]['lineEdit_size_0'].text()
            size_1 = self.action_dict[objName][1]['lineEdit_size_1'].text()
            self.action_dict[objName][2]['clipLimit'] = clipLimit
            self.action_dict[objName][2]['size_0'] =    size_0
            self.action_dict[objName][2]['size_1'] =    size_1
            self.action_dict[objName][4] = f'cv2.createCLAHE(clipLimit={clipLimit}, tileGridSize=({size_0}, {size_1}))'
        
        elif objectName == 'Morphology':
            
            size_0 = self.action_dict[objName][1]['lineEdit_size_0'].text()
            size_1 = self.action_dict[objName][1]['lineEdit_size_1'].text()
            sameValue = self.action_dict[objName][1]['lineEdit_sameValue'].text()
            functionType = self.action_dict[objName][1]['comboBox_functionType'].currentText()
            self.action_dict[objName][2]['size_0'] = size_0
            self.action_dict[objName][2]['size_1'] = size_1
            self.action_dict[objName][2]['functionType'] = functionType
            if size_0 != '' and size_1 != '': 
                self.action_dict[objName][1]['tableWidget_kernel'].setRowCount(int(size_0))
                self.action_dict[objName][1]['tableWidget_kernel'].setColumnCount(int(size_1))
            self.action_dict[objName][2]['sameValue'] = sameValue
            self.kernel(objName)
            kernel = self.action_dict[objName][2]['kernel']
            self.action_dict[objName][4] = f'cv2.morphologyEx(image, cv2.{functionType}, np.array({kernel}, "uint8"), iterations=1)'
        
        elif objectName == 'Erode':
            
            size_0 = self.action_dict[objName][1]['lineEdit_size_0'].text()
            size_1 = self.action_dict[objName][1]['lineEdit_size_1'].text()
            sameValue = self.action_dict[objName][1]['lineEdit_sameValue'].text()
            self.action_dict[objName][2]['size_0'] = size_0 
            self.action_dict[objName][2]['size_1'] = size_1
            if size_0 != '' and size_1 != '':
                self.action_dict[objName][1]['tableWidget_kernel'].setRowCount(int(size_0))
                self.action_dict[objName][1]['tableWidget_kernel'].setColumnCount(int(size_1))
            self.action_dict[objName][2]['sameValue'] = sameValue
            self.kernel(objName)
            kernel = self.action_dict[objName][2]['kernel']
            self.action_dict[objName][4] = f'cv2.erode(image, np.array({kernel}))'
        
        elif objectName == 'Dilate':
            
            size_0 = self.action_dict[objName][1]['lineEdit_size_0'].text()
            size_1 = self.action_dict[objName][1]['lineEdit_size_1'].text()
            sameValue = self.action_dict[objName][1]['lineEdit_sameValue'].text()
            self.action_dict[objName][2]['size_0'] = size_0
            self.action_dict[objName][2]['size_1'] = size_1
            if size_0 != '' and size_1 != '':
                self.action_dict[objName][1]['tableWidget_kernel'].setRowCount(int(size_0))
                self.action_dict[objName][1]['tableWidget_kernel'].setColumnCount(int(size_1))
            self.action_dict[objName][2]['sameValue'] = sameValue
            self.kernel(objName)
            kernel = self.action_dict[objName][2]['kernel']
            self.action_dict[objName][4] = f'cv2.dilate(image, np.array({kernel}))'
        
        self.imageFiltering()
    

    def kernel(self, object_name):
        if self.action_dict[object_name][2]['size_0'] != '' and self.action_dict[object_name][2]['size_1'] != '':
                if self.action_dict[object_name][1]['checkBox_sameValue'].isChecked() and self.action_dict[object_name][2]['sameValue'] != '':
                    self.action_dict[object_name][1]['tableWidget_kernel'].setDisabled(True)
                    self.action_dict[object_name][2]['sameValue_state'] = True
                    main_list = []
                    for i in range(int(self.action_dict[object_name][2]['size_0'])):
                        temp_list = []
                        for j in range(int(self.action_dict[object_name][2]['size_1'])):
                            temp_list.append(float(self.action_dict[object_name][2]['sameValue'].replace(",", ".")))
                        main_list.append(temp_list)
                    self.action_dict[object_name][2]['kernel'] = main_list
                else:
                    main_list = []
                    self.action_dict[object_name][1]['tableWidget_kernel'].setEnabled(True)
                    self.action_dict[object_name][2]['sameValue_state'] = False
                    for i in range(int(self.action_dict[object_name][2]['size_0'])):
                        temp_list = []
                        for j in range(int(self.action_dict[object_name][2]['size_1'])):
                            if self.action_dict[object_name][1]['tableWidget_kernel'].item(i, j) == None or self.action_dict[object_name][1]['tableWidget_kernel'].item(i, j).text() == '':
                                temp_list.append(.0)
                                continue
                            temp_list.append(float(self.action_dict[object_name][1]['tableWidget_kernel'].item(i, j).text().replace(",", "."))) 
                        main_list.append(temp_list)
                    self.action_dict[object_name][2]['kernel'] = main_list


    def setupActionsScrollArea(self):        
        self.Actions_scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.Actions_scrollArea.setGeometry(QtCore.QRect(200, 10, 190, 480))
        self.Actions_scrollArea.setMaximumSize(QtCore.QSize(190, 500))
        self.Actions_scrollArea.setWidgetResizable(True)
        self.Actions_scrollArea.setObjectName("Actions_scrollArea")
        self.Actions_scrollAreaWidgetContents = QtWidgets.QWidget()
        self.Actions_scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 188, 418))
        self.Actions_scrollAreaWidgetContents.setObjectName("Actions_scrollAreaWidgetContents")
        self.Actions_scrollArea_verticalLayout = QtWidgets.QVBoxLayout(self.Actions_scrollAreaWidgetContents)
        self.Actions_scrollArea_verticalLayout.setObjectName("Actions_scrollArea_verticalLayout")
        self.Actions_scrollArea.setWidget(self.Actions_scrollAreaWidgetContents)
        
    
    def add_action(self, index, name):        
        exec(f"self.action_{index}_groupBox = QtWidgets.QGroupBox(self.Actions_scrollAreaWidgetContents)")
        exec(f"self.action_{index}_widget = QtWidgets.QWidget(self.action_{index}_groupBox)")
        exec(f"self.action_{index}_horizontalLayout = QtWidgets.QHBoxLayout(self.action_{index}_widget)")
        exec(f"self.action_{index}_UPButton = QtWidgets.QPushButton(self.action_{index}_widget)")
        exec(f"self.action_{index}_DOWNButton = QtWidgets.QPushButton(self.action_{index}_widget)")
        exec(f"self.action_{index}_DELETEButton = QtWidgets.QPushButton(self.action_{index}_widget)")
        
        self.action_dict[f"{name}_{index}"].append(
        {
            'groupBox': eval(f"self.action_{index}_groupBox"),
            'widget': eval(f"self.action_{index}_widget"),
            'horizontalLayout': eval(f"self.action_{index}_horizontalLayout"),
            'UPButton': eval(f"self.action_{index}_UPButton"),
            'DOWNButton': eval(f"self.action_{index}_DOWNButton"),
            'DELETEButton': eval(f"self.action_{index}_DELETEButton")
        }
        )
        
        self.action_dict[f"{name}_{index}"][3]["groupBox"].setMinimumSize(QtCore.QSize(150, 60))
        self.action_dict[f"{name}_{index}"][3]["groupBox"].setMaximumSize(QtCore.QSize(150, 60))
        self.action_dict[f"{name}_{index}"][3]["groupBox"].setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_dict[f"{name}_{index}"][3]["groupBox"].setObjectName(f"action_{index}_groupBox")
        self.action_dict[f"{name}_{index}"][3]["groupBox"].setTitle(f"{name} : {index}")
        self.action_dict[f"{name}_{index}"][3]["widget"].setGeometry(QtCore.QRect(30, 20, 91, 27))
        self.action_dict[f"{name}_{index}"][3]["widget"].setObjectName(f"action_{index}_widget")
        self.action_dict[f"{name}_{index}"][3]["horizontalLayout"].setContentsMargins(0, 0, 0, 0)
        self.action_dict[f"{name}_{index}"][3]["horizontalLayout"].setObjectName(f"action_{index}_horizontalLayout")
        
        self.action_dict[f"{name}_{index}"][3]["UPButton"].setMinimumSize(QtCore.QSize(25, 25))
        self.action_dict[f"{name}_{index}"][3]["UPButton"].setMaximumSize(QtCore.QSize(25, 25))
        self.action_dict[f"{name}_{index}"][3]["UPButton"].setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("up_0.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.action_dict[f"{name}_{index}"][3]["UPButton"].setIcon(icon)
        self.action_dict[f"{name}_{index}"][3]["UPButton"].setIconSize(QtCore.QSize(18, 18))
        self.action_dict[f"{name}_{index}"][3]["UPButton"].setObjectName(f"action_{index}_UPButton")
        self.action_dict[f"{name}_{index}"][3]["horizontalLayout"].addWidget(self.action_dict[f"{name}_{index}"][3]["UPButton"])
        
        self.action_dict[f"{name}_{index}"][3]["DOWNButton"].setMinimumSize(QtCore.QSize(25, 25))
        self.action_dict[f"{name}_{index}"][3]["DOWNButton"].setMaximumSize(QtCore.QSize(25, 25))
        self.action_dict[f"{name}_{index}"][3]["DOWNButton"].setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("down_0.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.action_dict[f"{name}_{index}"][3]["DOWNButton"].setIcon(icon1)
        self.action_dict[f"{name}_{index}"][3]["DOWNButton"].setIconSize(QtCore.QSize(18, 18))
        self.action_dict[f"{name}_{index}"][3]["DOWNButton"].setObjectName(f"action_{index}_DOWNButton")
        self.action_dict[f"{name}_{index}"][3]["horizontalLayout"].addWidget(self.action_dict[f"{name}_{index}"][3]["DOWNButton"])
        
        self.action_dict[f"{name}_{index}"][3]["DELETEButton"].setMinimumSize(QtCore.QSize(25, 25))
        self.action_dict[f"{name}_{index}"][3]["DELETEButton"].setMaximumSize(QtCore.QSize(25, 25))
        self.action_dict[f"{name}_{index}"][3]["DELETEButton"].setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("cross.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.action_dict[f"{name}_{index}"][3]["DELETEButton"].setIcon(icon2)
        self.action_dict[f"{name}_{index}"][3]["DELETEButton"].setIconSize(QtCore.QSize(18, 18))
        self.action_dict[f"{name}_{index}"][3]["DELETEButton"].setObjectName(f"action_{index}_DELETEButton")
        self.action_dict[f"{name}_{index}"][3]["horizontalLayout"].addWidget(self.action_dict[f"{name}_{index}"][3]["DELETEButton"])
        
        self.action_dict[f"{name}_{index}"][3]["UPButton"].clicked.connect(lambda: self.UPAction(f"{name}_{index}"))
        self.action_dict[f"{name}_{index}"][3]["DOWNButton"].clicked.connect(lambda: self.DOWNAction(f"{name}_{index}"))
        self.action_dict[f"{name}_{index}"][3]["DELETEButton"].clicked.connect(lambda: self.DELETEAction(f"{name}_{index}"))
        
        self.Actions_scrollArea_verticalLayout.addWidget(self.action_dict[f"{name}_{index}"][3]["groupBox"])
    
    
    def UPAction(self, name):        
        items = list(self.action_dict.items())
        if self.action_dict[name][0] != 0:
            self.Elements_scrollArea_verticalLayout_main.removeWidget(self.action_dict[name][1]["groupBox"])
            self.Elements_scrollArea_verticalLayout_main.insertWidget(self.action_dict[name][0] - 1, self.action_dict[name][1]["groupBox"])
            self.Actions_scrollArea_verticalLayout.removeWidget(self.action_dict[name][3]["groupBox"])
            self.Actions_scrollArea_verticalLayout.insertWidget(self.action_dict[name][0] - 1, self.action_dict[name][3]["groupBox"])
            for item in items:
                if item[1][0] == self.action_dict[name][0] - 1:
                    temp_name = item[0]
                    break
            first_name = name.split("_")[0]
            second_name = temp_name.split("_")[0]
            self.action_dict[name][0], self.action_dict[temp_name][0] = self.action_dict[temp_name][0], self.action_dict[name][0]            
            self.action_dict[name][1]["groupBox"].setTitle(f"{first_name} : {self.action_dict[name][0]}")
            self.action_dict[name][3]["groupBox"].setTitle(f"{first_name} : {self.action_dict[name][0]}")
            self.action_dict[temp_name][1]["groupBox"].setTitle(f"{second_name} : {self.action_dict[temp_name][0]}")
            self.action_dict[temp_name][3]["groupBox"].setTitle(f"{second_name} : {self.action_dict[temp_name][0]}")
        
        self.imageFiltering()
        
    
    def DOWNAction(self, name):        
        items = list(self.action_dict.items())
        if self.action_dict[name][0] != len(self.action_dict) - 1:
            self.Elements_scrollArea_verticalLayout_main.removeWidget(self.action_dict[name][1]["groupBox"])
            self.Elements_scrollArea_verticalLayout_main.insertWidget(self.action_dict[name][0] + 1, self.action_dict[name][1]["groupBox"])
            self.Actions_scrollArea_verticalLayout.removeWidget(self.action_dict[name][3]["groupBox"])
            self.Actions_scrollArea_verticalLayout.insertWidget(self.action_dict[name][0] + 1, self.action_dict[name][3]["groupBox"])
            for item in items:
                if item[1][0] == self.action_dict[name][0] + 1:
                    temp_name = item[0]
                    break
            first_name = name.split("_")[0]
            second_name = temp_name.split("_")[0]
            self.action_dict[name][0], self.action_dict[temp_name][0] = self.action_dict[temp_name][0], self.action_dict[name][0]            
            self.action_dict[name][1]["groupBox"].setTitle(f"{first_name} : {self.action_dict[name][0]}")
            self.action_dict[name][3]["groupBox"].setTitle(f"{first_name} : {self.action_dict[name][0]}")
            self.action_dict[temp_name][1]["groupBox"].setTitle(f"{second_name} : {self.action_dict[temp_name][0]}")
            self.action_dict[temp_name][3]["groupBox"].setTitle(f"{second_name} : {self.action_dict[temp_name][0]}")
    
        self.imageFiltering()
    
    
    def DELETEAction(self, name):        
        for key in self.action_dict[name][1].keys():
            # print(key)
            if (key.split("_")[0] == 'formLayout' or 'horizontal' in key or 'vertical' in key) and 'Slider' not in key: #== 'horizontalLayout' or key == 'verticalLayout' or key == 'verticalLayout':
                self.Elements_scrollArea_verticalLayout_main.removeItem(self.action_dict[name][1][key])
            else:
                self.Elements_scrollArea_verticalLayout_main.removeWidget(self.action_dict[name][1][key])
            self.action_dict[name][1][key].deleteLater()
            self.action_dict[name][1][key] = None
        for key in self.action_dict[name][3].keys():
            # print(key)
            if (key.split("_")[0] == 'formLayout' or 'horizontal' in key or 'vertical' in key) and 'Slider' not in key:  #key.find('Layout') != 0: #== 'horizontalLayout':
                self.Actions_scrollArea_verticalLayout.removeItem(self.action_dict[name][3][key])
            else:
                self.Actions_scrollArea_verticalLayout.removeWidget(self.action_dict[name][3][key])
            self.action_dict[name][3][key].deleteLater()
            self.action_dict[name][3][key] = None
        self.action_dict.pop(name)
        for key in self.action_dict.keys():
            first_name = key.split("_")[0]
            self.action_dict[key][0] = self.Elements_scrollArea_verticalLayout_main.indexOf(self.action_dict[key][1]['groupBox'])
            self.action_dict[key][1]["groupBox"].setTitle(f"{first_name} : {self.action_dict[key][0]}")
            self.action_dict[key][3]["groupBox"].setTitle(f"{first_name} : {self.action_dict[key][0]}")
    
        self.imageFiltering()
    

    def setupElementsScrollArea(self):        
        self.Elements_scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.Elements_scrollArea.setGeometry(QtCore.QRect(400, 10, 300, 480))
        self.Elements_scrollArea.setMaximumSize(QtCore.QSize(300, 10000))
        self.Elements_scrollArea.setWidgetResizable(True)
        self.Elements_scrollArea.setObjectName("Elements_scrollArea")
        self.Elements_scrollAreaWidgetContents = QtWidgets.QWidget()
        self.Elements_scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 281, 2678))
        self.Elements_scrollAreaWidgetContents.setObjectName("Elements_scrollAreaWidgetContents")
        self.Elements_scrollArea_verticalLayout_main = QtWidgets.QVBoxLayout(self.Elements_scrollAreaWidgetContents)
        self.Elements_scrollArea_verticalLayout_main.setObjectName("Elements_scrollArea_verticalLayout_main")
        self.Elements_scrollArea.setWidget(self.Elements_scrollAreaWidgetContents)


    def setupImageLabelArea(self):        
        self.Image_label = QLabelClickable(self.centralwidget)
        self.Image_label.setGeometry(QtCore.QRect(710, 10, 680, 450))
        self.Image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.Image_label.setObjectName("Image_label")
        self.Image_label.clicked.connect(self.clickImageLabel)
        
    
    def clickImageLabel(self):        
        if hasattr(self, "image"):
            self.h = int(1080) #*(self.image.shape[0]/2160))
            self.w = int(1920) #*(self.image.shape[1]/3840))
            # sizes = [int(1920*(self.image.shape[1]/3840)), int(1080*(self.image.shape[0]/2160))]
            self.ui_ImageWindow = imageInsance()
            self.ui_ImageWindow.show()
            self.ui_ImageWindow.setImage(self.image, self.h, self.w)
        if hasattr(self, "list_of_filters"):
            self.drawImage()

   
    def imageFiltering(self):        
        self.list_of_filters = []
        temp_list_of_filters = list(self.action_dict.items())
        temp_list_of_filters.sort(key = lambda x: x[:][1][0])
        for item in temp_list_of_filters:
            self.list_of_filters.append(item[1][4])
        self.drawImage()

       
    def drawImage(self):        
        image = self.image
        try:
            for filter in self.list_of_filters:
                if "cv2.createCLAHE" in filter:
                    exec(f"clahe = {filter}")
                    exec(f"temp_image = clahe.apply(temp_image)")
                else:
                    exec(f"temp_image = {filter}")    
                image = eval("temp_image")
                
        except Exception as e:
            print(f" Impossible operation due to {e}")

        try:
            if hasattr(self, "ui_ImageWindow"):
                self.ui_ImageWindow.setImage(image, self.h, self.w)
            # print(f"image shape - {image.shape}")  
            if len(image.shape) == 2:
                lastH, lastW = image.shape
                lastCh = 1
                imageFormat = QtGui.QImage.Format.Format_Grayscale8
            else:
                lastH, lastW, lastCh = image.shape
                imageFormat = QtGui.QImage.Format.Format_BGR888
            bytesPline = lastCh * lastW
            imageInWindow = QtGui.QImage(image.data, image.shape[1], image.shape[0], bytesPline, imageFormat)
            # imageInWindow = imageInWindow.scaled(self.Image_label.width, self.Image_label.height, QtCore.Qt.KeepAspectRatio)
            
            pixmapImage = QtGui.QPixmap.fromImage(imageInWindow)
            pixmapImage = pixmapImage.scaled(self.Image_label.width(), self.Image_label.height(), aspectRatioMode = QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.Image_label.setPixmap(pixmapImage)
        except Exception as e:
            print(f"Image can't be displayed, 123131because {e}")


class imageInsance(QtWidgets.QWidget, Ui_imageWindow):
    
    
    def __init__(self, parent=None):
       super(imageInsance, self).__init__(parent)
       self.setupUi(self, h = ui.h, w = ui.w)


if __name__ == "__main__":
	import sys
	app = QtWidgets.QApplication(sys.argv)
	MainWindow = QtWidgets.QMainWindow()
	ui = Ui_MainWindow()
	ui.setupUi(MainWindow)
	MainWindow.show()
	sys.exit(app.exec())